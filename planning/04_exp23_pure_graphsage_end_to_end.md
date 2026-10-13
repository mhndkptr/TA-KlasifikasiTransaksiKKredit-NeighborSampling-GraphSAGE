# Rencana Eksperimen 23 (EXP23): Pure End-to-End HeteroGraphSAGE Classification (Ablasi Eliminasi XGBoost Classifier)

---

## 1. Identitas Eksperimen

* **ID Eksperimen**: `EXP23`
* **Nama Rencana**: *Pure End-to-End HeteroGraphSAGE Classification (Standalone GNN Evaluation & XGBoost Ablation)*
* **Tipe Eksperimen**: Studi Ablasi Komponen (*Component Ablation Study & Architecture Disentanglement*)
* **Dasar Replikasi**: Identik dengan protokol data, arsitektur GNN, dan regularisasi konvergen pada **EXP21 v5 (*The Golden Balance*)**, namun **mengeliminasi secara total classifier downstream XGBoost dan tahap ekstraksi fitur perantara**.
* **Target Utama**:
  1. Mengukur performa klasifikasi murni dari representasi struktural graf dan *classifier head* linear internal bawaan `HeteroGraphSAGE` ($p_i = \sigma(W h_i + b)$).
  2. Mengisolasi (*disentangle*) dampak nyata dari ketiga strategi *neighbor sampling* (`uniform`, `topology`, `importance`) tanpa bias dominasi fitur tabular pohon keputusan XGBoost.
  3. Membangun pilar kedua (*Pure GNN Paradigm*) untuk tabel komparasi tri-paradigma utama pada **Bab 4 Skripsi**: `Tabular XGBoost` vs `Pure GraphSAGE (EXP23)` vs `Hybrid GraphSAGE + XGBoost (EXP21)`.
  4. Memangkas kompleksitas komputasi, penggunaan memori RAM (tanpa array numpy perantara berukuran ratusan MB), dan mempercepat waktu inferensi.

---

## 2. Latar Belakang & Motivasi Ilmiah

### 2.1. Temuan Kritis pada EXP21 (Fenomena Dominasi XGBoost & Masking Efek Sampling)
Pada seluruh iterasi EXP21 (v1 hingga v5):
* Hasil metrik pada Test Set 60 hari menunjukkan skor yang nyaris kembar untuk seluruh metode sampling:
  - **Uniform**: F1 = 0.9397, Recall = 100.0%, Precision = 88.63%, AUCPR = 0.9225, ROC-AUC = 0.9999
  - **Topology**: F1 = 0.9397, Recall = 100.0%, Precision = 88.63%, AUCPR = 0.9128, ROC-AUC = 0.9999
  - **Importance**: F1 = 0.9397, Recall = 100.0%, Precision = 88.63%, AUCPR = 0.9225, ROC-AUC = 0.9999
* Bukti paling mencolok terjadi pada **EXP21 v4**: saat representasi GNN mengalami *mode collapse* dan stagnasi loss di 1.147 (GNN gagal belajar), **XGBoost tetap menghasilkan F1 0.9397 dan Recall 100.0%**.
* Hal ini membuktikan secara empiris bahwa **24 fitur tabular transaksi asli sangat kuat dan mendominasi pohon keputusan XGBoost**, sehingga kontribusi murni dari GraphSAGE serta perbedaan kualitas representasi antara `uniform`, `topology`, dan `importance` menjadi tertutupi (*masked*).

### 2.2. Relevansi Fundamental Terhadap Topik Tugas Akhir
Judul Tugas Akhir berfokus pada:
> **"Klasifikasi Transaksi Kartu Kredit Menggunakan Neighbor Sampling dan GraphSAGE"**

Dalam sidang akademik dan penulisan laporan akhir, muncul pertanyaan metodologis yang krusial:
1. *Jika XGBoost sudah mampu memprediksi dengan sangat akurat dari fitur tabular, apa bukti empiris bahwa GraphSAGE benar-benar mampu mengenali fraud secara mandiri?*
2. *Bagaimana performa GraphSAGE murni jika diterapkan langsung sebagai sistem klasifikasi end-to-end tanpa bantuan ensemble pohon?*
3. *Bagaimana karakteristik sebenarnya dari metode sampling Topology-Aware dan Importance-Based pada level GraphSAGE murni?*

EXP23 dirancang secara spesifik untuk menjawab ketiga pertanyaan ilmiah tersebut dengan menghilangkan XGBoost sepenuhnya dari pipeline.

---

## 3. Hipotesis Penelitian

1. **Hipotesis Kelayakan Standalone (H1)**: `HeteroGraphSAGE` mampu berdiri sendiri sebagai *binary fraud classifier end-to-end* yang efektif dan stabil, menghasilkan AUCPR $> 0.70$ dan ROC-AUC $> 0.95$ pada test set masa depan (60 hari) melalui kalibrasi ambang batas dinamis $\tau^*_{gnn}$.
2. **Hipotesis Isolasi Kausalitas Sampling (H2)**: Tanpa XGBoost sebagai penyerap sinyal tabular, disparitas performa antara `uniform`, `topology`, dan `importance` akan terlihat lebih tegas (*unmasked*), memperlihatkan keunggulan sampling berbasis topologi derajat terhadap fenomena *hub-neighborhood explosion*.
3. **Hipotesis Efisiensi Inferensi (H3)**: Menghapus Phase 2 (ekstraksi array numpy 152 dimensi) dan Phase 3 (pelatihan XGBoost) akan mengurangi total runtime pipeline lebih dari 40% dan mengeliminasi overhead RAM secara signifikan.

---

## 4. Perbandingan Arsitektur Pipeline: EXP21 vs EXP23

```text
===================================================================================
Pipeline EXP21 (Hybrid: GraphSAGE + XGBoost)
===================================================================================
[Dataset 2016-2020] ──> [Rolling Window SHINE]
                              │
                              ▼
        [Phase 1: Train HeteroGraphSAGE (10 Epochs)]
                              │
                              ▼
        [Phase 2: Extract Embeddings (128-d) + Tabular (24-d)]  <── [Overhead RAM Masif]
                              │
                              ▼
        [Phase 3: Train XGBoost Classifier (100 Trees)]         <── [Mendominasi Sinyal]
                              │
                              ▼
        [Phase 4: Dynamic Threshold Calibration on XGBoost]
                              │
                              ▼
        [Test Evaluation on XGBoost Probabilities]

===================================================================================
Pipeline EXP23 (Pure End-to-End Standalone HeteroGraphSAGE)
===================================================================================
[Dataset 2016-2020] ──> [Rolling Window SHINE]
                              │
                              ▼
        [Phase 1: Train HeteroGraphSAGE (10 Epochs dengan Best Checkpointing)]
                              │
                              ▼
        [Phase 2: Direct Inductive GNN Evaluation & Threshold Calibration]
            ├── Validation Loader ──> Prediksi Langsung Probabilitas Sigmoid GNN
            │                         └── Cari tau*_gnn = argmax F1_val(tau)
            │
            └── Test Loader       ──> Prediksi Langsung Probabilitas Sigmoid GNN
                                      └── Evaluasi Test @ tau*_gnn & @ 0.50
                                      └── Hitung AUCPR, ROC-AUC, Lift@1%, TP@500
```

---

## 5. Desain & Spesifikasi Teknis EXP23

### 5.1. Komponen yang Dipertahankan (Identik dengan EXP21 v5)
Untuk menjamin prinsip *Controlled Experiment*, seluruh variabel data dan arsitektur GNN dikunci sama persis dengan EXP21 v5:
1. **Protokol Temporal Window (SHINE FRAUD II)**:
   - Tanggal Referensi ($T_0$): `2018-01-01`
   - Train Window: 300 hari (1 Jan 2018 – 28 Okt 2018, $\approx 1.400.000$ transaksi)
   - Validation Window: 30 hari (28 Okt 2018 – 27 Nov 2018, $\approx 140.000$ transaksi)
   - Embargo Gap: 8 hari (27 Nov 2018 – 5 Des 2018, dilewati / anti-leakage)
   - Test Window: 60 hari (5 Des 2018 – 3 Feb 2019, 283.529 transaksi, 421 fraud)
2. **Konstruksi Graf Heterogen (PyG HeteroData)**:
   - Node Types: `transaction`, `user`, `merchant`
   - Edge Relations: Bipartit multi-relasi antara user-transaction dan merchant-transaction
   - Fitur Node Transaksi: 24 fitur tabular standar (dinormalisasi)
   - Fitur Node User & Merchant: 2 fitur agregat terstandarisasi (`StandardScaler`)
3. **Arsitektur Model HeteroGraphSAGE**:
   - `num_layers`: 2
   - `hidden_channels`: 128
   - `aggr`: `'mean'`
   - `dropout`: 0.20
   - `BatchNorm1d` setelah setiap konvolusi
4. **Strategi Neighbor Sampling**:
   - `uniform`: Pembobotan seragam standar
   - `topology`: Pembobotan logaritmik derajat sumber $w_e = \log(1 + deg(src))$
   - `importance`: Pembobotan derajat teredam + sinyal nominal & resensi
   - `num_neighbors`: `[15, 10]`
   - `batch_size`: 1024
5. **Hiperparameter Pelatihan & Regularisasi Konvergensi (EXP21 v5 Golden Balance)**:
   - Optimizer: Adam (`lr = 0.001`, `weight_decay = 0.0001`)
   - Loss Function: `manual_weighted_bce` dengan strategi `loss_weight_strategy: "sqrt"` ($pos\_weight \approx 27.06$)
   - `label_smoothing`: `0.0` (Mencegah tabrakan matematis & stagnasi loss)
   - `grad_clip_norm`: `5.0`
   - `warmup_epochs`: `0`
   - `epochs`: 10
   - Scheduler: `ReduceLROnPlateau` (factor=0.5, patience=2, monitor='val_auprc')
   - Best Checkpoint: Menyimpan dan memulihkan bobot saat `val_auprc` mencapai titik tertinggi (`best_gnn.pt`)

### 5.2. Komponen yang Dihilangkan & Dimodifikasi

| Komponen | EXP21 (Hybrid) | EXP23 (Pure GNN) | Rationale Ilmiah |
| :--- | :--- | :--- | :--- |
| **Ekstraksi Fitur (Phase 2)** | Pre-alokasi numpy array 152-dim (24 tabular + 128 embedding) | **Dihapus Total (0 overhead)** | Tidak memerlukan penyimpanan perantara; inferensi langsung mini-batch via PyG loader |
| **Downstream Classifier (Phase 3)** | XGBoost Classifier (100 pohon, `max_depth=5`, `scale_pos_weight`) | **Dihapus Total** | Mengeliminasi bias tabular dan menguji kemampuan murni representasi graf |
| **Classifier Head** | Pohon keputusan XGBoost | `nn.Linear(128, 1)` + `torch.sigmoid` internal bawaan model | Klasifikasi biner langsung dari representasi node transaksi $h_{\text{trans}} \in \mathbb{R}^{128}$ |
| **Kalibrasi Threshold ($\tau^*$)** | Dicari dari probabilitas prediksi XGBoost pada data validasi | Dicari dari probabilitas prediksi **sigmoid GNN murni** pada data validasi | Mengatasi miskalibrasi akibat *class-weighted loss* pada output probabilitas GNN secara langsung |
| **Inferensi Data Uji** | Inferensi 2-tahap (GNN embed $\rightarrow$ XGBoost predict) | **Inferensi 1-tahap langsung** (GNN predict biner) | Mempercepat kecepatan prediksi uji dan mengurangi latensi sistem |

---

## 6. Protokol Evaluasi & Kalibrasi Threshold GNN Murni

### 6.1. Mengapa Kalibrasi Threshold Tetap Wajib untuk Pure GNN?
Ketika GNN dilatih menggunakan `weighted_bce` dengan denda kelas minoritas ($pos\_weight \approx 27.06$), fungsi aktivasi sigmoid tergeser ke arah probabilitas positif yang lebih tinggi. Pada threshold default $\tau = 0.50$, metrik Precision dan F1 sering kali tidak optimal.

Dalam EXP23, kalibrasi threshold dilakukan secara murni pada probabilitas output GNN:
1. Jalankan `best_gnn` pada `val_loader` (graf Train + Val, anti-leakage).
2. Dapatkan probabilitas kontinu $p_i^{val} \in [0, 1]$ untuk setiap transaksi validasi.
3. Lakukan grid search pada rentang $\tau \in [0.005, 0.50]$ dengan step $0.005$:
   $$\tau^*_{gnn} = \arg\max_{\tau} \text{F1}_{val}(\tau)$$
4. Kunci nilai $\tau^*_{gnn}$ dan terapkan langsung (*out-of-sample*) pada probabilitas transaksi uji $p_i^{test}$.

### 6.2. Metrik yang Dicatat & Dianalisis
1. **Metrik @ Threshold Default 0.50**: F1-Score, Precision, Recall, Accuracy.
2. **Metrik @ Optimal Threshold $\tau^*_{gnn}$**: F1-Score, F1-Macro, Precision, Recall, Accuracy, Specificity, G-Mean.
3. **Metrik Independen Threshold**: AUCPR (AUPRC), ROC-AUC.
4. **Metrik Operasional Perbankan**:
   - **Lift@1%**: Kelipatan peningkatan fraud detection pada top 1% alert sistem.
   - **TP@500 & Precision@500**: Jumlah dan akurasi fraud tertangkap dalam 500 transaksi paling mencurigakan harian.
5. **Metrik Komputasi**:
   - `gnn_training_time` (detik)
   - `gnn_val_tuning_time` (detik)
   - `gnn_test_inference_time` (detik)
   - `total_pipeline_time` (detik)

---

## 7. Struktur Direktori & Berkas Rencana

```text
code/exp23/
├── config.py                      # Parser konfigurasi YAML
├── config.reference.yaml          # Konfigurasi lokal CPU/GPU
├── config.kaggle.yaml             # Konfigurasi GPU Kaggle (T4/P100)
├── config.colab.yaml              # Konfigurasi Google Colab
├── data_preprocessing.py          # Modul rolling window temporal (2016-2020)
├── graph_builder.py               # Konstruksi HeteroData PyG
├── model.py                       # HeteroGraphSAGE + manual_weighted_bce (v5 Golden Balance)
├── samplers.py                    # Neighbor samplers (Uniform, Topology, Importance)
├── main.py                        # Pipeline ramping 2 fase (Pure GNN Train -> Direct Eval)
├── run.py                         # Eksekusi cepat baseline Uniform
├── run_all.py                     # Eksekusi berurutan 3 metode sampling + auto plot
├── visualize_results.py           # Visualisasi 16 chart ilmiah (disesuaikan tanpa XGBoost)
├── create_nb.py                   # Generator Kaggle Notebook
├── create_colab_nb.py             # Generator Colab Notebook
├── exp23_training_notebook.ipynb  # Berkas Jupyter Notebook siap upload ke Kaggle
├── exp23_colab_notebook.ipynb     # Berkas Jupyter Notebook siap buka di Colab
├── kernel-metadata.json           # Metadata push Kaggle API
├── dataset-metadata.json          # Metadata dataset Kaggle API
├── penjelasan_pipeline.md         # Penjelasan konseptual mendalam
├── LOCAL_RUN_GUIDE.md             # Panduan run lokal step-by-step
└── README.md                      # Dokumentasi komprehensif EXP23
```

### Penyesuaian `visualize_results.py` untuk Pure GNN:
* **Chart 1 s/d 14**: Tetap aktif penuh (komparasi metrik, kalibrasi threshold, confusion matrix, operasional perbankan TP@500, lift, PR-curve, ROC-curve, learning curves loss & akurasi GNN).
* **Chart 15 (Sebelumnya Konvergensi XGBoost)**: Digantikan dengan **Kurva Sensitivitas Kalibrasi Threshold Dinamis GNN** (menampilkan kurva evolusi Precision, Recall, dan F1-Score sepanjang $\tau \in [0.005, 0.50]$ pada Validation set vs Test set) atau **Diagram Reliabilitas Kalibrasi Probabilitas (ECE / Reliability Curve)**.
* **Chart 16 & 17**: Kurva konvergensi Loss dan Akurasi GNN per-step mini-batch iterasi dengan *Exponential Moving Average (EMA)*.

---

## 8. Format Output & Artefak Hasil

Hasil eksekusi per metode sampling akan tersimpan di `result/exp23/`:
1. `result/exp23/window_statistics.json`: Statistik partisi temporal.
2. `result/exp23/exp23_{method}_results.json`: Seluruh metrik numerik final (F1, Precision, Recall, AUCPR, ROC-AUC, TP@500, runtime).
3. `result/exp23/exp23_{method}_history.json`: Log riwayat pelatihan 10 epoch dan ribuan batch steps.
4. `result/exp23/exp23_{method}_predictions.npz`: Array numpy berisi `test_probs` (sigmoid murni) dan `test_labels` untuk 283.529 transaksi uji.
5. `model/exp23/exp23_{method}_best_gnn.pt`: Bobot model terbaik PyTorch (tidak ada berkas `.json` XGBoost).
6. `result/exp23/plots/`: 16 berkas visualisasi beresolusi tinggi 300 DPI.

---

## 9. Rencana Matriks Komparasi untuk Bab 4 Skripsi (Tri-Paradigm Ablation)

Hasil dari EXP23 akan melengkapi tabel utama skripsi yang membandingkan 3 paradigma:

| Paradigma | Metode Sampling | Model Klasifikasi Akhir | F1 @ $\tau^*$ | Recall | Precision | AUCPR | ROC-AUC | TP@500 | Waktu Pipeline |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Tabular Baseline** | N/A (Tanpa Graf) | XGBoost Murni (Tabular 24-d) | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | Sangat Cepat |
| **2. Pure GNN (EXP23)** | **Uniform** | **HeteroGraphSAGE Murni** | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | $\approx 12$ Menit |
| | **Topology-Aware** | **HeteroGraphSAGE Murni** | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | $\approx 12$ Menit |
| | **Importance-Based**| **HeteroGraphSAGE Murni** | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | $\approx 13$ Menit |
| **3. Hybrid (EXP21 v5)** | Uniform | GraphSAGE (128-d) + XGBoost | 0.9397 | 100.0% | 88.63% | 0.9225 | 0.9999 | 421 / 500 | $\approx 17$ Menit |
| | Topology-Aware | GraphSAGE (128-d) + XGBoost | 0.9397 | 100.0% | 88.63% | 0.9128 | 0.9999 | 421 / 500 | $\approx 17$ Menit |
| | Importance-Based | GraphSAGE (128-d) + XGBoost | 0.9397 | 100.0% | 88.63% | 0.9225 | 0.9999 | 421 / 500 | $\approx 18$ Menit |

---

## 10. Kriteria Keberhasilan (Success Criteria)

1. **Stabilitas Konvergensi Penuh**: Loss pelatihan dan validasi GNN konvergen mulus tanpa divergensi (*loss explosion*) maupun stagnasi (*loss stagnation* di 1.147).
2. **Evaluasi Murni Mandiri**: Seluruh metrik evaluasi (F1, Precision, Recall, AUCPR, ROC-AUC) dihasilkan 100% oleh bobot `best_gnn.pt`.
3. **Disentanglement Sampling**: Terlihat kontras performa yang jelas dan dapat dipertanggungjawabkan antara `uniform`, `topology`, dan `importance` pada representasi GNN murni.
4. **Efisiensi Komputasi**: Total waktu eksekusi skrip `main.py` per metode sampling di GPU Kaggle T4 berada di bawah 15 menit, dengan penghematan RAM $> 500$ MB dibanding EXP21.
