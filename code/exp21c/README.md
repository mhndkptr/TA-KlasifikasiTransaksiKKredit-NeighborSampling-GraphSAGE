# Eksperimen 21C (EXP21C): Multi-Window Temporal Splitting & Walk-Forward Validation

Repositori ini berisi implementasi kode untuk **Eksperimen 21C (EXP21C)**, yaitu eksperimen validasi bergulir multi-horizon waktu (*Multi-Horizon Rolling Walk-Forward Validation*) pada pipeline hibrida GraphSAGE + XGBoost bebas kebocoran (*anti-leakage*).

Eksperimen ini dirancang secara khusus untuk merealisasikan arahan pembimbing pada dokumen [07_eksplorasi_data_splitting_atribut_tuning_exp21.md](file:///d:/Projects/Klasifikasi%20Transasi%20Kartu%20Kredit%20-%20GraphSAGE/tugas-akhir-klasifikasi-transaksi-kartu-kredit/planning/07_eksplorasi_data_splitting_atribut_tuning_exp21.md) bagian **Pilar I: Eksplorasi Pemrosesan Data & Data Splitting** dan **Eksperimen C (EXP21-C)**.

---

## 1. Motivasi & Latar Belakang Ilmiah

Pada eksperimen EXP21, model menghasilkan metrik *too good to be true* (Recall 100%, F1 0.94) yang disebabkan oleh penghafalan ID kuasi (`Zip` dan `Merchant City`) dengan split gain >99,9%.

Setelah kebocoran atribut diatasi pada **EXP21A** dan regularisasi pohon dioptimalkan pada **EXP21B**, pembimbing memberikan catatan kritis:
> *"Apakah model hanya beruntung pada satu jendela waktu evaluasi 60 hari (Desember 2018 s/d Februari 2019)? Bagaimana performa model jika diuji secara temporal di horizon masa depan yang berbeda pada era modern (2016–2020)?"*

Untuk menjawab pertanyaan ini secara tuntas dan ilmiah, **EXP21C** mengimplementasikan:
1. **Multi-Horizon Rolling Window (Walk-Forward Validation / SPLIT-2)**:
   - Menguji stabilitas model pada **3 jendela waktu bergulir independen**:
     - **Window 1 (2017–2018)**: Train Jan–Okt 2017, Test Des 2017–Feb 2018.
     - **Window 2 (2018–2019)**: Train Jan–Okt 2018, Test Des 2018–Feb 2019 (Horizon EXP21 baseline).
     - **Window 3 (2019–2020)**: Train Jan–Okt 2019, Test Des 2019–Feb 2020.
   - Melaporkan performa dalam format agregat ilmiah: Nilai Rata-rata $\pm$ Standar Deviasi ($\mu \pm \sigma$).
2. **Ablasi Durasi Embargo Gap (SPLIT-3)**:
   - Menguji variasi jeda verifikasi investigasi fraud: **0 hari** (tanpa gap), **8 hari** (standar paper SHINE), dan **30 hari** (siklus penagihan kartu kredit/chargeback bulanan).
3. **Studi Nilai Tambah Representasi GraphSAGE (Lift Analysis)**:
   - Membandingkan *Tabular Only vs GraphSAGE Only vs Hybrid* di setiap jendela waktu untuk membuktikan bahwa GraphSAGE memberikan kontribusi positif yang konsisten di berbagai era waktu.

---

## 2. Fitur & Keunggulan Metodologis EXP21C

* **Strict Anti-Leakage Preprocessing (SPLIT-1)**: Standarisasi `StandardScaler` dan kamus kategori hanya di-fit pada data Train masing-masing jendela waktu. Nilai baru pada Val/Test di-mapping ke `<oov>` (0).
* **Inductive Strictly-Past Graph Scaling (SPLIT-4)**: Scaler agregasi User dan Merchant di-fit strictly pada graf Train.
* **Fitur Bersih Bebas Quasi-Identifier**: Kolom `Zip` dan `Merchant City` dieksklusikan secara permanen; ditambahkan 4 fitur waktu siklik (`hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`).
* **Regularisasi Pohon XGBoost Optimal**: Mengadopsi konfigurasi seimbang hasil EXP21B (`scale_pos_weight = sqrt(neg/pos)`, `max_depth = 4`, `min_child_weight = 10`, `colsample_bytree = 0.8`).
* **Visualisasi Publikasi 300 DPI**: Menyertakan visualisasi perbandingan multi-window dengan error bar, kurva PR/ROC lintas era, dan kurva sensitivitas embargo gap.

---

## 3. Struktur Berkas

```text
code/exp21c/
├── config.reference.yaml         # Konfigurasi lokal (3 rolling windows, embargo gap, path)
├── config.kaggle.yaml            # Konfigurasi Kaggle GPU
├── config.colab.yaml             # Konfigurasi Google Colab
├── config.py                     # Loader konfigurasi YAML
├── data_preprocessing.py         # Multi-window slicing & strict anti-leakage preprocessing
├── graph_builder.py              # HeteroData builder dengan entity scaler guard
├── samplers.py                   # Uniform, Topology-Aware, Importance-Based
├── model.py                      # HeteroGraphSAGE & Class-Weighted BCE
├── main.py                       # Main pipeline multi-window walk-forward validation
├── visualize_results.py          # Generator plot ilmiah 300 DPI (multi-window, PR/ROC, lift)
├── run.py                        # Single-run execution wrapper
├── run_all.py                    # Sequential runner 3 sampling + ablasi + summary report
├── create_nb.py                  # Generator exp21c_training_notebook.ipynb (Kaggle)
├── create_colab_nb.py            # Generator exp21c_colab_notebook.ipynb (Colab)
├── exp21c_training_notebook.ipynb# Kaggle Notebook siap pakai
├── exp21c_colab_notebook.ipynb   # Colab Notebook siap pakai
├── kernel-metadata.json          # Metadata push Kaggle
├── dataset-metadata.json         # Metadata dataset code Kaggle
├── LOCAL_RUN_GUIDE.md            # Panduan lengkap eksekusi lokal
├── penjelasan_pipeline.md        # Laporan teknis mendalam & perancangan tabel Bab 4
└── README.md                     # Ringkasan proyek
```

---

## 4. Cara Menjalankan

### Di Kaggle:
Unggah berkas `exp21c_training_notebook.ipynb` sebagai Kaggle Notebook dengan GPU Tesla T4 aktif dan pasang dataset IBM Synthetic Credit Card Transactions.

### Di Colab:
Buka berkas `exp21c_colab_notebook.ipynb` di Google Colab dengan runtime GPU T4 aktif.

### Di Komputer Lokal:
Ikuti panduan lengkap pada [LOCAL_RUN_GUIDE.md](file:///d:/Projects/Klasifikasi%20Transasi%20Kartu%20Kredit%20-%20GraphSAGE/tugas-akhir-klasifikasi-transaksi-kartu-kredit/code/exp21c/LOCAL_RUN_GUIDE.md).
