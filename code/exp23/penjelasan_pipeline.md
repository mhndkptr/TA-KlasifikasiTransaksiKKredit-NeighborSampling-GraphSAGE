# Penjelasan Pipeline Eksperimen 23 (EXP23)
## *Pure End-to-End HeteroGraphSAGE Classification (XGBoost Ablation Protocol)*

---

## 1. Latar Belakang & Rasional Ilmiah

Pada eksperimen pendahulu (`EXP21`), implementasi pipeline menggunakan pendekatan hibrida:
1. `HeteroGraphSAGE` dilatih untuk menghasilkan representasi node transaksi berdimensi 128 (*embedding*).
2. Representasi 128 dimensi tersebut digabungkan (*concatenated*) dengan 24 fitur tabular asli.
3. Model ensemble pohon keputusan (`XGBoost`) dilatih di atas gabungan 152 fitur tersebut untuk menghasilkan keputusan fraud akhir.

### Anomali Temuan EXP21:
* Pada seluruh metode neighbor sampling (`uniform`, `topology`, `importance`), hasil metrik pada Test Set 60 hari terkunci seragam di angka F1 = 0.9397, Recall = 100%, Precision = 88.63%, dan AUCPR $\approx 0.92$.
* Pada **EXP21 v4**, ketika GNN mengalami *mode collapse/loss stagnation* di loss 1.147, **XGBoost tetap menghasilkan F1 0.9397 dan Recall 100%**.
* Hal ini membuktikan bahwa **24 fitur tabular asli transaksi mendominasi pohon keputusan XGBoost**, sehingga kontribusi murni dari GraphSAGE serta perbedaan kualitas representasi antara `uniform`, `topology`, dan `importance` menjadi tersamarkan (*masked*).

### Solusi Ilmiah EXP23:
Eksperimen 23 (`EXP23`) menerapkan **studi ablasi eliminasi total classifier downstream XGBoost**:
* Mengeliminasi Phase 2 (ekstraksi array numpy 152 dimensi) dan Phase 3 (pelatihan XGBoost).
* Menggunakan **Linear Classifier Head internal bawaan model HeteroGraphSAGE** ($p_i = \sigma(W h_i + b)$) untuk memprediksi probabilitas fraud secara langsung.
* Mengalibrasi ambang batas dinamis $\tau^*_{gnn}$ langsung pada output probabilitas sigmoid GNN di subset validasi.
* Mengisolasi (*disentangle*) efektivitas representasi struktural ketiga metode sampling (`uniform`, `topology`, `importance`) secara murni.

---

## 2. Tahapan Alur Eksekusi (2 Phases Ramping)

Pipeline dikoordinasikan dalam skrip `main.py` melalui 2 tahapan berurutan tanpa penyimpanan perantara:

```text
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. Data Preprocessing (Filter Year >= 2016, Label Encoding)             │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 2. Temporal Window Splitting (Train 300d, Val 30d, Gap 8d, Test 60d)    │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 3. Phase 1: Train HeteroGraphSAGE (2-Layer, 128-dim, Mean Aggr, 10 Ep)   │
│    - Weighted BCE Loss (pos_weight sqrt-damped ~27.06)                  │
│    - Adam (lr=1e-3, weight_decay=1e-4), Grad Clip 5.0, Warmup 0         │
│    - ReduceLROnPlateau Scheduler (monitor='val_auprc')                  │
│    - Best Checkpoint Saving (best_gnn.pt) & Auto-Restoration            │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 4. Phase 2: Direct Inductive Evaluation & Dynamic Threshold Calibration │
│    - Validation Loader: Prediksi langsung p_i = sigmoid(out)            │
│      └── Cari tau*_gnn = argmax F1_val(tau)                             │
│    - Test Loader (Inductive Graph: Train + Val + Test):                 │
│      └── Prediksi langsung p_i untuk transaksi uji (test nodes only)    │
│      └── Evaluasi Test @ tau*_gnn & @ 0.50                              │
│      └── Hitung AUCPR, ROC-AUC, Lift@1%, TP@500                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Rincian Teknis Tiap Fase

### Phase 1: Pelatihan GraphSAGE Encoder & Classifier Head
* **Arsitektur**: 2-layer `HeteroGraphSAGE`, `hidden_channels = 128`, agregator `mean`, `dropout = 0.20`, `BatchNorm1d`.
* **Classifier Head**:
  $$h_{\text{trans}} \in \mathbb{R}^{128}$$
  $$\text{logit}_i = W \cdot h_{\text{trans}} + b \quad (W \in \mathbb{R}^{1 \times 128})$$
  $$p_i = \sigma(\text{logit}_i) = \frac{1}{1 + e^{-\text{logit}_i}}$$
* **Loss Function**: `manual_weighted_bce` dengan pembobotan kelas teredam (*square-root*):
  $$\mathcal{L} = - \frac{1}{N} \sum_{i=1}^N \left[ w_{\text{pos}} y_i \log(p_i + \epsilon) + (1 - y_i) \log(1 - p_i + \epsilon) \right]$$
  di mana $w_{\text{pos}} = \sqrt{\frac{N_{\text{neg}}}{N_{\text{pos}}}} \approx 27.06$.
* **Restorasi Checkpoint Terbaik**: Bobot model dengan validation AUCPR tertinggi otomatis dipulihkan ke memori GPU sebelum memasuki evaluasi Phase 2.

### Phase 2: Kalibrasi Threshold Dinamis & Evaluasi Induktif Murni
* **Kalibrasi Threshold ($\tau^*_{gnn}$)**:
  Karena $w_{\text{pos}} \approx 27.06$ memberikan penalti besar pada False Negative, distribusi probabilitas sigmoid $p_i$ tergeser ke kanan. Ambang batas default $\tau = 0.50$ dapat menghasilkan False Positive yang berlebih.
  Oleh karena itu, grid search dilakukan pada Validation set:
  $$\tau^*_{gnn} = \arg\max_{\tau \in [0.01, 0.50]} \text{F1}_{\text{val}}(\tau)$$
* **Evaluasi Induktif Bebas Kebocoran Data (*Strict Anti-Leakage*)**:
  - Graf uji memuat interaksi historis Train + Val + Test, namun komputasi metrik hanya dilakukan pada node transaksi Test Set (`test_node_indices`).
  - Nilai $\tau^*_{gnn}$ diterapkan langsung out-of-sample ke Test Set.
  - Metrik yang dihitung mencakup: F1-Score (@0.5 & @ $\tau^*$), Recall, Precision, Accuracy, Specificity, G-Mean, AUCPR, ROC-AUC, Lift@1%, dan TP@500.

---

## 4. Keuntungan Komputasi EXP23

1. **Zero-Memory Overhead**: Tidak ada lagi array numpy raksasa berukuran 283.529 $\times$ 152 float32 yang dialokasikan di RAM.
2. **Eliminasi Latensi XGBoost**: Tidak ada proses pembangunan 100 pohon keputusan CPU/GPU dan komputasi split.
3. **Penyederhanaan Dependensi**: Seluruh inferensi dieksekusi 100% di dalam runtime PyTorch Geometric.
4. **Waktu Pipeline**: Waktu total per metode sampling turun dari $\approx 17$ menit ke $\approx 10$–$12$ menit di GPU Tesla T4.
