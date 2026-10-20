# Eksperimen 21B (EXP21B): Balanced Hyperparameter Tuning & Tree Regularization

Repositori ini berisi implementasi kode untuk **Eksperimen 21B (EXP21B)**, yaitu eksperimen eksplorasi dan optimasi terarah (*systematic hyperparameter tuning*) serta regularisasi pohon XGBoost pada pipeline hibrida GraphSAGE + XGBoost bebas kebocoran (*anti-leakage*).

Eksperimen ini dirancang secara khusus untuk merealisasikan arahan pembimbing pada dokumen [07_eksplorasi_data_splitting_atribut_tuning_exp21.md](file:///d:/Projects/Klasifikasi%20Transasi%20Kartu%20Kredit%20-%20GraphSAGE/tugas-akhir-klasifikasi-transaksi-kartu-kredit/planning/07_eksplorasi_data_splitting_atribut_tuning_exp21.md) bagian **Eksperimen B (EXP21-B)**.

---

## 1. Motivasi & Latar Belakang Ilmiah

Pada eksperimen EXP21, model menghasilkan metrik *too good to be true* (Recall 100%, F1 0.94) yang disebabkan oleh penghafalan ID kuasi (`Zip` dan `Merchant City`) dengan split gain >99,9%.

Setelah kebocoran tersebut dieliminasi pada EXP21A melalui penghapusan atribut quasi-identifier dan penerapan *Strict Out-of-Sample Preprocessing*, langkah krusial berikutnya adalah:
1. **Menghilangkan ketergantungan pada pembobotan ekstrem**: `scale_pos_weight = 732.18` (rasio penuh) memaksa pohon membuat daun-daun mikro yang sangat sensitif demi menghindari False Negative.
2. **Menemukan titik temu optimal antara Recall tinggi dan Precision yang sehat**: Menguji redaman bobot kelas (`scale_pos_weight`: 1.0, 10.0, 27.0, 50.0).
3. **Membatasi kompleksitas pohon (Anti-Memorization)**: Melalui regularisasi `max_depth` (3 s/d 5), `min_child_weight` (5 s/d 50), dan `colsample_bytree` (0.6 s/d 1.0) agar representasi struktural 128-d GraphSAGE mendapatkan ruang kontribusi yang signifikan.
4. **Kuantifikasi Kontribusi Murni GraphSAGE (Lift)**: Menguji secara komparatif *Tabular Only vs GraphSAGE Only vs Hybrid Tuned* untuk menjawab pertanyaan inti skripsi mengenai nilai tambah representasi graf.

---

## 2. Inovasi & Fitur Utama pada EXP21B

1. **Anti-Leakage Data Splitting & Preprocessing**:
   - Protokol temporal SHINE FRAUD II: Train (300d), Val (30d), Embargo Gap (8d), Test (60d).
   - Scaler dan kamus kategori **hanya di-fit pada Train Window**. Nilai baru pada Val/Test di-mapping ke `<oov>` (0).
   - Fitur bersih: `Amount`, `Use Chip`, `MCC`, `Errors?`, `Merchant State`, serta 4 fitur waktu siklik (`hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`).
2. **One-Time Graph Representation Extraction**:
   - Representasi induktif GraphSAGE 128-dimensi diekstrak **satu kali** ke dalam memori numpy untuk Train, Val, dan Test. Hal ini memungkinkan eksplorasi 108 kombinasi pohon berjalan sangat cepat (beberapa menit).
3. **Systematic Hyperparameter Tuning Engine**:
   - Grid parameter:
     - `scale_pos_weight`: `[1.0, 10.0, 27.0 (sqrt), 50.0]`
     - `max_depth`: `[3, 4, 5]`
     - `min_child_weight`: `[5, 20, 50]`
     - `colsample_bytree`: `[0.6, 0.8, 1.0]`
   - Pencarian threshold optimal $\tau^*$ secara dinamis pada Validation Set holdout.
   - Perekaman otomatis seluruh log eksplorasi ke file CSV dan JSON.
4. **Studi Ablasi Terstruktur untuk Bab 4 Skripsi**:
   - **Ablasi Damping SPW**: Menguji respon model terhadap variasi bobot kelas pada Test Set masa depan.
   - **Ablasi Subset Fitur**: Mengukur *Lift* yang diberikan GraphSAGE di atas baseline tabular murni.
5. **Audit Feature Importance & Visualisasi Publikasi 300 DPI**:
   - Pemetaan kontribusi Split Gain (Tabular vs GNN).
   - Heatmap interaksi parameter, kurva PR dan ROC, serta chart ablasi otomatis.

---

## 3. Struktur Berkas

```text
code/exp21b/
├── config.reference.yaml         # Konfigurasi lokal
├── config.kaggle.yaml            # Konfigurasi Kaggle GPU
├── config.colab.yaml             # Konfigurasi Google Colab
├── config.py                     # Loader konfigurasi YAML
├── data_preprocessing.py         # Strict anti-leakage splitting & preprocessing
├── graph_builder.py              # HeteroData builder dengan entity scaler guard
├── samplers.py                   # Uniform, Topology-Aware, Importance-Based
├── model.py                      # HeteroGraphSAGE & Class-Weighted BCE
├── main.py                       # Main pipeline end-to-end + tuning & ablation engine
├── visualize_results.py          # Generator plot ilmiah 300 DPI (heatmap, PR/ROC, feature importance)
├── run.py                        # Single-run execution wrapper
├── run_all.py                    # Sequential runner 3 sampling + visualisasi + summary report
├── create_nb.py                  # Generator exp21b_training_notebook.ipynb (Kaggle)
├── create_colab_nb.py            # Generator exp21b_colab_notebook.ipynb (Colab)
├── exp21b_training_notebook.ipynb# Kaggle Notebook siap pakai
├── exp21b_colab_notebook.ipynb   # Colab Notebook siap pakai
├── kernel-metadata.json          # Metadata push Kaggle
├── dataset-metadata.json         # Metadata dataset code Kaggle
├── LOCAL_RUN_GUIDE.md            # Panduan lengkap eksekusi lokal
├── penjelasan_pipeline.md        # Laporan teknis mendalam & tinjauan metodologis
└── README.md                     # Ringkasan proyek
```

---

## 4. Cara Menjalankan

### Di Kaggle:
Unggah berkas `exp21b_training_notebook.ipynb` sebagai Kaggle Notebook dengan GPU Tesla T4 aktif dan pasang dataset IBM Synthetic Credit Card Transactions.

### Di Colab:
Buka berkas `exp21b_colab_notebook.ipynb` di Google Colab dengan runtime GPU T4 aktif.

### Di Komputer Lokal:
Ikuti panduan lengkap pada [LOCAL_RUN_GUIDE.md](file:///d:/Projects/Klasifikasi%20Transasi%20Kartu%20Kredit%20-%20GraphSAGE/tugas-akhir-klasifikasi-transaksi-kartu-kredit/code/exp21b/LOCAL_RUN_GUIDE.md).
