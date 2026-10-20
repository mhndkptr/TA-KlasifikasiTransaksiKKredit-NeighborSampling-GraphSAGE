# Eksperimen 21A (EXP21A): Anti-Leakage Preprocessing & Clean Attribute Pipeline

Repositori ini berisi implementasi kode untuk **Eksperimen 21A (EXP21A)**, yaitu eksperimen audit dan remediasi terhadap temuan hasil *too good to be true* (Recall 100%, F1 0.94) pada EXP21 berdasarkan saran pembimbing.

---

## 1. Motivasi & Temuan Kunci

Pada audit model XGBoost tersimpan di EXP21 (`model/exp21/v5/exp21_uniform_xgb.json`):
* Fitur `f4` (`Zip`) memiliki total split gain **30,7 juta**.
* Fitur `f2` (`Merchant City`) memiliki total split gain **9,25 juta**.
* Kedua fitur ID kuasi ini menyumbang **>99,9% keputusan pohon**, menenggelamkan 128 fitur representasi graf GraphSAGE. Pohon keputusan terbukti memotong `Zip <= 22832` dan sekadar menghafal klaster ZIP sintetis tempat data fraud IBM TabFormer berkumpul.
* Selain itu, penskalaan `StandardScaler` dan `LabelEncoder` di-fit pada seluruh data sebelum splitting (kebocoran global).

---

## 2. Solusi & Perubahan Utama pada EXP21A

1. **Eliminasi Fitur Penghafal (Quasi-Identifier)**:
   - Kolom `Zip` (~25.000+ kode) dan `Merchant City` (~10.000+ kota) **dihapus secara total** dari masukan model.
   - Fitur tabular bersih: `Amount`, `Use Chip`, `MCC`, `Errors?`, `Merchant State`.
   - Ditambahkan 4 fitur waktu siklik: `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`.
2. **Strict Out-of-Sample Preprocessing**:
   - `StandardScaler` Amount dan kamus kategori **hanya di-fit pada data Train Window** (300 hari).
   - Data Validation dan Test murni di-transform; kategori baru yang belum pernah muncul di Train dipetakan ke token `<oov>` (index 0).
3. **Pemberantasan Kebocoran Agregasi Entitas Graf**:
   - Scaler agregat User dan Merchant di-fit pada data Train dan ditransformasikan pada graf Validation/Test tanpa membocorkan statistik masa depan.
4. **Regularisasi Pohon XGBoost**:
   - `scale_pos_weight` diredam menggunakan `sqrt(neg/pos)` ($\approx 27.0$).
   - `max_depth` dibatasi ke 4 dan `min_child_weight` dinaikkan ke 10.
   - `colsample_bytree = 0.8` dan `subsample = 0.8` untuk membagi bobot keputusan ke berbagai fitur (termasuk GNN).
5. **Audit Feature Importance Otomatis**:
   - Skrip `main.py` secara otomatis menghitung akumulasi split gain untuk fitur Tabular vs fitur GraphSAGE, menghasilkan file `exp21a_{method}_feature_importance.json`.

---

## 3. Struktur Berkas

```text
code/exp21a/
├── config.reference.yaml         # Konfigurasi lokal
├── config.kaggle.yaml            # Konfigurasi Kaggle GPU
├── config.colab.yaml             # Konfigurasi Google Colab
├── config.py                     # Loader konfigurasi
├── data_preprocessing.py         # Strict anti-leakage splitting & preprocessing
├── graph_builder.py              # HeteroData builder dengan entity scaler guard
├── samplers.py                   # Uniform, Topology-Aware, Importance-Based
├── model.py                      # HeteroGraphSAGE & Class-Weighted BCE
├── main.py                       # Main pipeline end-to-end + feature importance audit
├── visualize_results.py          # Generator plot ilmiah 300 DPI
├── run.py                        # Single-run execution wrapper
├── run_all.py                    # Sequential runner 3 sampling + visualisasi
├── create_nb.py                  # Generator exp21a_training_notebook.ipynb (Kaggle)
├── create_colab_nb.py            # Generator exp21a_colab_notebook.ipynb (Colab)
├── exp21a_training_notebook.ipynb# Kaggle Notebook siap pakai
├── exp21a_colab_notebook.ipynb   # Colab Notebook siap pakai
├── kernel-metadata.json          # Metadata push Kaggle
├── dataset-metadata.json         # Metadata dataset code Kaggle
├── LOCAL_RUN_GUIDE.md            # Panduan lokal
├── penjelasan_pipeline.md        # Laporan teknis & komparasi metodologis
└── README.md                     # Ringkasan proyek
```

---

## 4. Cara Menjalankan

### Di Kaggle:
Unggah berkas `exp21a_training_notebook.ipynb` sebagai Kaggle Notebook dengan GPU Tesla T4 aktif dan pasang dataset IBM Synthetic Credit Card Transactions.

### Di Colab:
Buka berkas `exp21a_colab_notebook.ipynb` di Google Colab dengan runtime GPU T4 aktif.

### Di Komputer Lokal:
Ikuti panduan lengkap pada [LOCAL_RUN_GUIDE.md](file:///d:/Projects/Klasifikasi%20Transasi%20Kartu%20Kredit%20-%20GraphSAGE/tugas-akhir-klasifikasi-transaksi-kartu-kredit/code/exp21a/LOCAL_RUN_GUIDE.md).
