# Eksperimen 21E (EXP21E): Pure End-to-End HeteroGraphSAGE
## *Implementasi Klasifikasi Murni GraphSAGE Selaras Bab 3.4 Proposal Tugas Akhir*

---

## 1. Deskripsi Eksperimen

Eksperimen **EXP21E** dirancang untuk menguji model **GraphSAGE secara murni end-to-end** persis seperti yang tertuang pada Bab 3.4 Proposal Tugas Akhir:
$$\hat{y}_v = \sigma(W \cdot h_v^{(2)} + b)$$

Seluruh protokol kontrol (*ceteris paribus*) dipertahankan sama persis dengan **EXP21D (The Golden Master)**, dengan satu-satunya variabel pembeda:
* **EXP21D (Hybrid)**: HeteroGraphSAGE (128-d) + Fitur Tabular (10-d) $\to$ Downstream XGBoost Classifier.
* **EXP21E (Pure GNN)**: HeteroGraphSAGE dengan *linear classifier head* internal $\to$ Sigmoid $\to$ Klasifikasi Biner Langsung (Tanpa XGBoost).

### Pilar Kontrol Identik dengan EXP21D:
1. **Strict Anti-Leakage Protocol**: Eliminasi total quasi-identifier `Zip` dan `Merchant City`, serta penskalaan/encoder kategori strictly fit hanya pada Train Window.
2. **Active Fraud Window Benchmark**: Partisi temporal rolling window 2018–2019 aktif (Train 300d, Val 30d, Embargo Gap 8d, Test 60d) sesuai standar SHINE (IEEE TKDE 2024).
3. **Neighbor Sampling Strategies**: Menguji 3 metode sampling secara sejajar: `Uniform`, `Topology-Aware`, dan `Importance-Based` ($k=[15, 10]$).
4. **Loss Function**: Weighted BCE Loss dengan damped square root ratio ($\text{pos\_weight} \approx 27.06$).
5. **Threshold Calibration**: Kalibrasi ambang batas optimal $\tau^*_{gnn}$ pada Validation Set.

---

## 2. Struktur Berkas di Direktori Ini

```text
code/exp21e/
├── config.reference.yaml         # Konfigurasi acuan lokal
├── config.kaggle.yaml            # Konfigurasi GPU Kaggle
├── config.colab.yaml             # Konfigurasi GPU Colab
├── config.py                     # Parser konfigurasi YAML
├── data_preprocessing.py         # Modul strict anti-leakage splitting & clean features
├── graph_builder.py              # Pembangun graf heterogen tripartit PyG
├── samplers.py                   # Implementasi Uniform, Topology-Aware, Importance-Based
├── model.py                      # Arsitektur HeteroGraphSAGE dengan linear head terintegrasi
├── main.py                       # Pipeline pelatihan murni GNN, kalibrasi threshold, & evaluasi
├── run_all.py                    # Eksekusi sekuensial 3 metode sampling & pembuatan tabel master
├── run.py                        # Shortcut eksekusi cepat
├── visualize_results.py          # Generator plot ilmiah 300 DPI (PR/ROC, CM, Sampling comparison)
├── create_nb.py                  # Generator notebook Kaggle mandiri
├── create_colab_nb.py            # Generator notebook Colab mandiri
├── dataset-metadata.json         # Metadata dataset kode Kaggle API
├── kernel-metadata.json          # Metadata kernel Kaggle API
├── LOCAL_RUN_GUIDE.md            # Panduan eksekusi lokal
└── README.md                     # Berkas dokumentasi ini
```

---

## 3. Cara Menjalankan

### Eksekusi Lokal:
```powershell
python main.py --config config.reference.yaml --sampling uniform --nrows 50000
```

### Eksekusi Penuh 3 Metode Sampling & Pembuatan Master Table:
```powershell
python run_all.py
```

### Eksekusi di Kaggle GPU:
Generate notebook Kaggle terlebih dahulu via:
```powershell
python create_nb.py
```
Lalu push kernel `exp21e_training_notebook.ipynb` menggunakan Kaggle API.
