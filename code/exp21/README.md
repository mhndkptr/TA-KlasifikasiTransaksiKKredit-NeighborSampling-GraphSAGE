# Eksperimen 21 (EXP21): Temporal Window Alignment & Inductive Setup

Eksperimen 21 (`EXP21`) adalah implementasi dari perencanaan [02_exp21_temporal_window_alignment.md](../../planning/02_exp21_temporal_window_alignment.md) yang bertujuan **menyelaraskan protokol partisi data dan lingkungan induktif dengan metodologi paper SHINE (IEEE TKDE 2024)** pada dataset **FRAUD II** (IBM Synthetic Credit Card Transactions).

---

## 1. Motivasi & Target Ilmiah

Pada eksperimen exp16–exp19, data dibagi dengan rasio global 70/15/15 mencakup rentang waktu 1991 hingga 2020 (hampir 30 tahun). Hal ini memicu dua kendala besar:
1. **30-Year Concept Drift**: Pola fraud 1991–2015 didominasi transaksi *Swipe* dan *Online*, sedangkan periode uji (2018–2020) didominasi oleh transaksi *Chip*. Model menghafal pola usang yang tidak relevan.
2. **Kaggle Execution Timeout**: Memproses graf dengan 17 juta transaksi latih dan 3,65 juta transaksi uji memicu ledakan komputasi pada *feature extraction* (>2 jam di Kaggle), menyebabkan sesi terputus.

### Solusi Protokol EXP21:
- Membatasi horizon data ke **era modern 2016–2020**.
- Mengadopsi protokol **Rolling Window SHINE FRAUD II**:
  - **Train Window**: 300 hari transaksi.
  - **Validation Window**: 30 hari transaksi.
  - **Embargo Gap**: 8 hari transaksi (jeda laporan klaim fraud dunia nyata).
  - **Test Window**: 60 hari transaksi.
- Menerapkan **Kalibrasi Threshold Dinamis ($\tau^*$)** pada data validasi (temuan EXP20) untuk memulihkan F1-Score dan Recall.

---

## 2. Struktur Berkas Kode

```text
code/exp21/
├── config.py                      # Parser konfigurasi YAML
├── config.reference.yaml          # Konfigurasi eksekusi lokal
├── config.kaggle.yaml             # Konfigurasi eksekusi lingkungan Kaggle GPU
├── config.colab.yaml              # Konfigurasi eksekusi Google Colab
├── data_preprocessing.py          # Filter temporal 2016-2020 & rolling window generator
├── graph_builder.py               # Konstruksi HeteroData PyG & undersampling
├── model.py                       # Arsitektur 2-layer HeteroGraphSAGE & weighted BCE
├── samplers.py                    # Neighbor samplers (Uniform, Topology, Importance)
├── main.py                        # Pipeline end-to-end 4 fase
├── run.py                         # Skrip eksekusi cepat baseline Uniform
├── run_all.py                     # Skrip eksekusi seluruh metode sampling berurutan
├── create_nb.py                   # Generator Kaggle Notebook
├── create_colab_nb.py             # Generator Colab Notebook
├── exp21_training_notebook.ipynb  # Berkas Jupyter Notebook siap upload ke Kaggle
├── exp21_colab_notebook.ipynb     # Berkas Jupyter Notebook siap buka di Colab
├── kernel-metadata.json           # Metadata push Kaggle API
├── dataset-metadata.json          # Metadata dataset Kaggle API
├── penjelasan_pipeline.md         # Penjelasan konseptual dan teknis mendalam
├── LOCAL_RUN_GUIDE.md             # Panduan eksekusi lokal step-by-step
└── README.md                      # Dokumentasi ringkas eksperimen ini
```

---

## 3. Cara Menjalankan

### A. Di Kaggle (Direkomendasikan untuk GPU)
1. Buat notebook baru di Kaggle atau gunakan Kaggle CLI:
   ```bash
   kaggle kernels push -p code/exp21/
   ```
2. Atau salin konten `exp21_training_notebook.ipynb` langsung ke editor Kaggle Notebook, lalu jalankan seluruh sel (aktifkan GPU P100/T4).

### B. Di Komputer Lokal
Buka terminal PowerShell dan aktifkan environment:
```powershell
cd code/exp21
python main.py --config config.reference.yaml --sampling uniform
```

Untuk smoke test cepat:
```powershell
python main.py --config config.reference.yaml --sampling uniform --nrows 50000
```

---

## 4. Berkas Output & Metrik Hasil

Hasil eksekusi akan tersimpan secara otomatis pada:
- **`result/exp21/window_statistics.json`**: Rincian jumlah transaksi, jumlah fraud, fraud rate (%), unique users, unique merchants, dan total nodes per split.
- **`result/exp21/exp21_uniform_results.json`**: Metrik lengkap F1 @ 0.5, F1 @ $\tau^*$, Recall, Precision, AUCPR, ROC-AUC, Lift@1%, TP@500, dan waktu komputasi.
- **`model/exp21/exp21_uniform_best_gnn.pt`**: Bobot optimal model PyTorch HeteroGraphSAGE (checkpoint terbaik berbasis Validation AUCPR).
- **`model/exp21/exp21_uniform_gnn.pt`**: Bobot model PyTorch HeteroGraphSAGE (epoch terakhir).
- **`model/exp21/exp21_uniform_xgb.json`**: Model classifier pohon XGBoost.

---

## 5. Pembaruan Anti-Overfitting & Regularisasi GNN

Berdasarkan audit kurva konvergensi (Chart 12–14), model GNN sebelumnya memperlihatkan divergensi loss terbobot dan fluktuasi metrik pada epoch akhir. Untuk memitigasi overfitting dan miskalibrasi, diimplementasikan solusi berikut:
1. **Dropout ($p=0.2$)**: Disisipkan setelah aktivasi ReLU dan BatchNorm pada layer representasi `HeteroGraphSAGE`.
2. **Weight Decay ($10^{-4}$)**: Regularisasi $L_2$ diintegrasikan ke optimizer Adam.
3. **Peredaman Class Weight (`pos_weight`)**: Opsi `loss_weight_strategy: "sqrt"` meredam rasio bobot ekstrem ($\approx 732$ menjadi $\approx 27.0$), mencegah penalti meledak (*exploding BCE loss*) pada False Positive di validation set.
4. **Learning Rate Scheduler**: `ReduceLROnPlateau` (factor=0.5, patience=2) memantau `val_auprc` untuk memperhalus langkah pembaruan bobot di epoch-epoch akhir.
5. **Model Checkpointing & Restorasi**: Menyimpan state dict saat `val_auprc` mencapai titik tertinggi (`best_gnn.pt`) dan **memulihkan (restore)** bobot terbaik tersebut sebelum ekstraksi representasi ke XGBoost di Phase 2.
6. **Dual Loss Logging**: Menghitung dan mencatat Weighted BCE Loss serta Unweighted BCE Loss per-epoch untuk evaluasi visual murni.
7. **Per-Step Mini-Batch Iteration Logging & Smoothing**: Mencatat metrik langkah demi langkah (`gnn_step_history`) di setiap mini-batch update untuk menghasilkan kurva konvergensi mulus eksponensial (*textbook exponential decay*) dengan *Exponential Moving Average (EMA)* pada:
   - **Chart 16**: Kurva Konvergensi Loss GNN per-Step / per-Batch Iterasi.
   - **Chart 17**: Kurva Konvergensi Akurasi GNN per-Step / per-Batch Iterasi.

---

## 6. Pembaruan v4: Stabilisasi Sampling, Mitigasi Overfitting Dini, & Resolusi Paradoks Metrik

Berdasarkan analisis empiris menyeluruh pada hasil **EXP21 v3**, ditemukan 3 anomali:
1. **Anomali Stabilitas (Importance-Based Drop di Epoch 9)**: Akurasi anjlok ke 99.12% dan precision ke 6.41% akibat ledakan ~1.000 False Positive.
2. **Overfitting Terlalu Dini (Uniform & Topology-Aware)**: Validation loss berbalik naik (U-turn) di Epoch 2–5.
3. **Paradoks Metrik**: Val Loss membaik berkat suku fraud yang berbobot 27x, tetapi mengaburkan ledakan False Positive transaksi normal di threshold kaku 0.5.

### Solusi yang Diimplementasikan pada v4:
1. **Log-Degree Centrality Sampling (Sublinear Scaling)**:
   - Pada `apply_importance_based_sampling` dan `apply_topology_aware_sampling`, bobot linear $deg(src)$ diganti dengan transformasi sublinear $w_e = \log(1 + deg(src))$.
   - Meredam dominasi merchant super-hub ($20.000$ transaksi vs $50$ transaksi) dari rasio $400:1$ menjadi $\approx 2.3:1$, menstabilkan varians sampling dan mengeliminasi ledakan False Positive.
2. **Gradient Clipping (`grad_clip_norm = 1.0`)**:
   - Menjaga stabilitas update gradien pada training loop GNN dari spike lokal saat memproses batch subgraf super-hub.
3. **Label Smoothing Regularization (`label_smoothing = 0.05`)**:
   - Memberikan target terhalus $y_{smooth} = 0.95y + 0.025$ untuk mencegah logit overconfidence yang memicu penalti logaritmik raksasa pada unseen validation nodes.
4. **Linear LR Warmup (1 Epoch)**:
   - Learning rate dinaikkan bertahap dari $0.1 \times LR$ ke $LR$ pada epoch pertama untuk mencegah lompatan ke *sharp minima* lokal.
5. **Peningkatan Regularisasi**:
   - Dropout dinaikkan ke $0.3$ dan Weight Decay dinaikkan ke $5 \times 10^{-4}$.
6. **Logging Dual-Threshold & Metrik Terkalibrasi**:
   - Fungsi evaluasi GNN mencatat metrik pada threshold baseline $0.5$ dan threshold optimal terkalibrasi $\tau^*$ (`val_calibrated_accuracy`, `val_calibrated_f1`, dsb.), sehingga kurva performa mencerminkan kalibrasi probabilitas yang sebenarnya.
7. **Direktori Output Terisolasi**:
   - Seluruh model dan artefak hasil disimpan rapi pada `model/exp21/v4` dan `result/exp21/v4`.

---

## 7. Pembaruan v5: Resolusi Stagnasi Loss & *The Golden Balance*

Pada hasil evaluasi **EXP21 v4**, model mengalami fenomena *learning stagnation* (loss tertahan di 1.147 dan akurasi terkalibrasi kolaps ke 0.2%). Audit forensik mengungkap bahwa hal ini disebabkan oleh **tabrakan matematis (*fatal collision*) antara `label_smoothing: 0.05` dan `pos_weight: 27.06`**:
- Target non-fraud ($y=0$) dihaluskan menjadi $0.025$.
- Suku ini dikalikan $pos\_weight \approx 27.06$ sehingga menghasilkan penalti buatan $0.6765 \ln(p)$ pada $99.86\%$ data.
- Secara matematis, fungsi loss tersebut memaksa model memprediksi $p \approx 0.41$ untuk seluruh transaksi, mengunci loss di nilai minimum teoretis $\approx 1.147$ (*mode collapse*).
- Ditambah kenaikan agresif weight decay ($5 \times 10^{-4}$) dan dropout ($0.30$), model terjebak dan kehilangan kapasitas representasi graf.

### Solusi & Konfigurasi v5 (*The Golden Balance*):
1. **Eliminasi Total Label Smoothing (`label_smoothing: 0.0`)**:
   - Menghapus tabrakan matematis dengan `pos_weight`. Target kembali murni $0$ dan $1$, mengeliminasi stagnasi loss $1.147$ dan mengembalikan konvergensi loss sehat.
2. **Normalisasi Regularisasi Parameter**:
   - `dropout: 0.20` (dinormalisasi dari 0.30 untuk mempertahankan kapasitas representasi struktural).
   - `weight_decay: 0.0001` (dinormalisasi dari 0.0005 untuk mencegah parameter shrinkage ke nol).
3. **Pelonggaran Gradient Clipping (`grad_clip_norm: 5.0`)**:
   - Melindungi dari spike batch ekstrem tanpa memotong aliran gradien alami sinyal fraud.
4. **Pembelajaran Aktif Sejak Awal (`warmup_epochs: 0`)**:
   - Mengaktifkan laju pembelajaran nominal sejak batch pertama.
5. **Retensi Inovasi Sublinear Log-Degree Sampling**:
   - Tetap mempertahankan pembobotan $w_e = \log(1 + deg(src))$ dari v4 untuk meredam varians super-hub merchant.
6. **Direktori Output Terisolasi v5**:
   - Seluruh artefak hasil dan checkpoints model disimpan secara otomatis pada `model/exp21/v5` dan `result/exp21/v5`.

---

## 8. Pembaruan v6: Scaled GraphSAGE Architecture & Expanded Context Sampling

Pada konfigurasi **EXP21 v6**, dilakukan ekspansi kapasitas model dan perluasan jangkauan sampling subgraf sesuai arsitektur optimal terstandardisasi:

| Komponen / Hyperparameter | Konfigurasi v5 | **Konfigurasi v6 (Terkini)** | Rasional & Justifikasi |
| :--- | :---: | :---: | :--- |
| **Lapisan GraphSAGE** | 2 Layer | **2 Layer (Mean Aggregator)** | Tetap mempertahankan 2-hop message passing dengan Mean Aggregator untuk agregasi tetangga yang stabil. |
| **Hidden Dimension** | 128 | **256 (ReLU & BatchNorm)** | Kapasitas dimensi laten digandakan menjadi 256 untuk memproyeksikan pola topologi kompleks ke ruang representasi yang lebih ekspresif. |
| **Ukuran Sampel ($k$)** | [15, 10] | **25 & 10 (Hop-1 & Hop-2)** | Memperluas ekspansi sampling pada Hop-1 ($k_1=25$) untuk menangkap konteks lokal transaksi yang lebih komprehensif terhadap entitas User dan Merchant. |
| **Ukuran Batch** | 1.024 | **1.024 (Node Transaksi)** | 1.024 node transaksi per iterasi mini-batch untuk kestabilan estimasi gradien. |
| **Learning Rate & Optimizer** | 0.001 (Decay 0.5x / 2 ep) | **0.001 (Adam, Decay 0.5x / 5 epoch)** | Memberikan rentang waktu adaptasi bobot 5 epoch sebelum decay 0.5x diterapkan oleh LR Scheduler (`ReduceLROnPlateau`). |
| **Patience Early Stopping** | 0 (Fixed 10 Epochs) | **15 Epoch (Maks 100 Epochs)** | Early stopping aktif berbasis `val_auprc` dengan toleransi 15 epoch tanpa peningkatan sebelum memulihkan (*restore*) checkpoint bobot model terbaik. |
| **Direktori Output** | `v5/` | **`model/exp21/v6` & `result/exp21/v6`** | Direktori hasil dan bobot terisolasi penuh pada folder v6. |

