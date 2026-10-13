# Eksperimen 23 (EXP23): Pure End-to-End HeteroGraphSAGE Classification (XGBoost Ablation)

Eksperimen 23 (`EXP23`) adalah implementasi dari perencanaan [04_exp23_pure_graphsage_end_to_end.md](../../planning/04_exp23_pure_graphsage_end_to_end.md) yang bertujuan **menguji kemampuan klasifikasi murni dari HeteroGraphSAGE secara mandiri (*stand-alone end-to-end classifier*) tanpa bantuan classifier downstream pohon XGBoost** pada dataset IBM Credit Card Synthetic (protokol **SHINE FRAUD II** 2016–2020).

---

## 1. Motivasi & Target Ilmiah

Pada eksperimen EXP21 (v1 hingga v5), model hybrid menggabungkan representasi graf 128-dimensi dengan 24 fitur tabular asli ke dalam XGBoost. Hal ini memicu dua kendala evaluasi akademik:
1. **Masking Sinyal Struktural oleh Fitur Tabular**: 24 fitur tabular mendominasi pohon keputusan XGBoost sehingga metrik pada Test Set seragam identik di seluruh metode sampling (F1 = 0.9397, Recall = 100%, Precision = 88.63%). Pada EXP21 v4, saat GNN mengalami *mode collapse*, XGBoost tetap menghasilkan F1 0.9397.
2. **Kebutuhan Studi Ablasi Tri-Paradigma Bab 4**: Untuk membuktikan keunggulan GraphSAGE secara ilmiah, laporan tugas akhir membutuhkan perbandingan 3 paradigma:
   - Paradigma 1: Tabular Murni (XGBoost saja).
   - **Paradigma 2: Pure GNN (HeteroGraphSAGE saja — EXP23)**.
   - Paradigma 3: Hybrid (GraphSAGE + XGBoost — EXP21).

### Solusi Protokol EXP23:
* **Eliminasi Total XGBoost & Ekstraksi Fitur**: Menghapus Phase 2 dan Phase 3.
* **Klasifikasi Linear Head Sigmoid Murni**: Menggunakan output probabilitas langsung dari layer linear internal model `HeteroGraphSAGE` ($p_i = \sigma(W h_i + b)$).
* **Kalibrasi Threshold Dinamis GNN**: Mencari ambang batas $\tau^*_{gnn} = \arg\max \text{F1}_{val}(\tau)$ langsung dari probabilitas validasi GNN.
* **Isolasi Efek Sampling (*Disentanglement*)**: Membandingkan kontribusi sesungguhnya dari `uniform`, `topology`, dan `importance` pada representasi struktural graf.

---

## 2. Struktur Berkas Kode

```text
code/exp23/
├── config.py                      # Parser konfigurasi YAML
├── config.reference.yaml          # Konfigurasi eksekusi lokal (CPU/GPU)
├── config.kaggle.yaml             # Konfigurasi eksekusi GPU Kaggle (T4/P100)
├── config.colab.yaml              # Konfigurasi eksekusi Google Colab
├── data_preprocessing.py          # Filter temporal 2016-2020 & rolling window generator
├── graph_builder.py               # Konstruksi HeteroData PyG & undersampling
├── model.py                       # HeteroGraphSAGE & class-weighted BCE loss (Golden Balance)
├── samplers.py                    # Neighbor samplers (Uniform, Topology, Importance)
├── main.py                        # Pipeline ramping 2 fase (Pure GNN Train -> Direct Eval)
├── run.py                         # Skrip eksekusi cepat baseline Uniform
├── run_all.py                     # Skrip eksekusi seluruh metode sampling berurutan + auto plot
├── visualize_results.py           # Generator 16 visualisasi ilmiah (Chart 1 - 17)
├── create_nb.py                   # Generator Kaggle Notebook
├── create_colab_nb.py             # Generator Colab Notebook
├── exp23_training_notebook.ipynb  # Berkas Jupyter Notebook siap upload ke Kaggle
├── exp23_colab_notebook.ipynb     # Berkas Jupyter Notebook siap buka di Colab
├── kernel-metadata.json           # Metadata push Kaggle API
├── dataset-metadata.json          # Metadata dataset Kaggle API
├── penjelasan_pipeline.md         # Penjelasan konseptual dan matematis mendalam
├── LOCAL_RUN_GUIDE.md             # Panduan eksekusi lokal step-by-step
└── README.md                      # Dokumentasi komprehensif eksperimen ini
```

---

## 3. Cara Menjalankan

### A. Di Kaggle (Direkomendasikan untuk GPU)
1. Push notebook langsung menggunakan Kaggle API CLI:
   ```bash
   kaggle kernels push -p code/exp23/
   ```
2. Atau salin konten `exp23_training_notebook.ipynb` ke notebook Kaggle dengan akselerator GPU P100/T4, lalu jalankan seluruh sel. Skrip `run_all.py` akan melatih ketiga varian sampling secara berurutan dan menghasilkan 16 chart ilmiah di `result/exp23/plots/`.

### B. Di Komputer Lokal
Buka terminal PowerShell:
```powershell
cd code/exp23
python main.py --config config.reference.yaml --sampling uniform
```

Untuk smoke test cepat:
```powershell
python main.py --config config.reference.yaml --sampling uniform --nrows 50000
```

---

## 4. Berkas Output & Metrik Hasil

Hasil eksekusi akan tersimpan secara otomatis pada:
* `result/exp23/window_statistics.json`: Rincian transaksi, fraud rate, unique users, unique merchants per partisi.
* `result/exp23/exp23_{method}_results.json`: Metrik lengkap F1 @ 0.5, F1 @ $\tau^*$, Recall, Precision, AUCPR, ROC-AUC, Lift@1%, TP@500, dan waktu komputasi.
* `result/exp23/exp23_{method}_history.json`: Log riwayat per-epoch dan per-step mini-batch iterasi.
* `result/exp23/exp23_{method}_predictions.npz`: Array probabilitas kontinu test set untuk kurva PR dan ROC.
* `model/exp23/exp23_{method}_best_gnn.pt`: Bobot model optimal PyTorch HeteroGraphSAGE (checkpoint terbaik).
* `result/exp23/plots/`: 16 galeri chart visualisasi ilmiah 300 DPI.
