# Eksperimen 22 (EXP22): Inovasi Agregator GNN: Elementwise Maximization (Max) vs Mean

Eksperimen 22 (`EXP22`) adalah implementasi dari perencanaan [03_exp22_gnn_aggregator_max_pooling.md](../../planning/03_exp22_gnn_aggregator_max_pooling.md) yang bertujuan **menguji pengaruh fungsi agregasi pesan `aggr='max'` (Elementwise Maximization) dibandingkan baseline `aggr='mean'` pada lapisan `SAGEConv` untuk mengatasi permasalahan *minority feature dilution* pada transaksi fraud**.

---

## 1. Motivasi & Target Ilmiah

Pada graf transaksi perbankan dengan ketimpangan kelas ekstrem (fraud rate ~0.15%), penggunaan agregator `mean` standar menyebabkan sinyal anomali transaksi fraud lenyap (*terlarut/diluted*) ke dalam rata-rata ribuan transaksi normal pada merchant atau user yang sama.

Mengacu pada paper **SHINE (IEEE TKDE 2024, Bagian III-D1, Persamaan 10 & 11)**, kami menerapkan:
* **`aggr='max'` (Elementwise Maximization)**: Mempertahankan aktivasi nilai ekstrem anomali yang dibawa oleh tetangga fraud tanpa teredam oleh tetangga normal mayoritas.
* **Arsitektur 2-Layer Ringan (128 dimensi)**: Menjawab kegagalan teknis `exp19` (yang mengalami *timeout* karena arsitektur 3-layer 256-dim pada 5,1 juta transaksi) dengan memanfaatkan protokol data bersih era modern (EXP21).
* **Evaluasi Separabilitas Embedding (t-SNE)**: Membuktikan secara visual dan empiris bahwa agregator `max` menghasilkan separasi klaster yang lebih tajam bagi classifier downstream (XGBoost).

---

## 2. Struktur Berkas Kode

```text
code/exp22/
├── config.py                      # Parser konfigurasi YAML
├── config.reference.yaml          # Konfigurasi eksekusi lokal
├── config.kaggle.yaml             # Konfigurasi eksekusi lingkungan Kaggle GPU
├── config.colab.yaml              # Konfigurasi eksekusi Google Colab
├── data_preprocessing.py          # Filter temporal 2016-2020 & rolling window generator
├── graph_builder.py               # Konstruksi HeteroData PyG & undersampling
├── model.py                       # HeteroGraphSAGE dengan dukungan SAGEConv aggr='max' / 'mean'
├── samplers.py                    # Neighbor samplers (Uniform, Topology, Importance)
├── visualize_tsne.py              # Visualisasi komparasi separabilitas embedding t-SNE
├── main.py                        # Pipeline end-to-end 4 fase dengan CLI --aggr
├── run.py                         # Skrip eksekusi cepat varian proposed (aggr='max')
├── run_all.py                     # Skrip eksekusi komparasi mean vs max + visualisasi t-SNE
├── create_nb.py                   # Generator Kaggle Notebook
├── create_colab_nb.py             # Generator Colab Notebook
├── exp22_training_notebook.ipynb  # Berkas Jupyter Notebook siap upload ke Kaggle
├── exp22_colab_notebook.ipynb     # Berkas Jupyter Notebook siap buka di Colab
├── kernel-metadata.json           # Metadata push Kaggle API
├── dataset-metadata.json          # Metadata dataset Kaggle API
├── penjelasan_pipeline.md         # Penjelasan konseptual dan matematis mendalam
├── LOCAL_RUN_GUIDE.md             # Panduan eksekusi lokal step-by-step
└── README.md                      # Dokumentasi eksperimen ini
```

---

## 3. Cara Menjalankan

### A. Di Kaggle (Direkomendasikan untuk GPU)
1. Buka notebook Kaggle atau gunakan Kaggle CLI:
   ```bash
   kaggle kernels push -p code/exp22/
   ```
2. Atau salin konten `exp22_training_notebook.ipynb` ke notebook Kaggle dengan akselerator GPU P100/T4, lalu jalankan seluruh sel. Skrip `run_all.py` akan melatih kedua varian (`mean` dan `max`), mengekstrak embedding, dan menghasilkan visualisasi perbandingan t-SNE.

### B. Di Komputer Lokal
Buka terminal PowerShell dan aktifkan environment:
```powershell
cd code/exp22

# Menjalankan varian proposed (Max Pooling):
python main.py --config config.reference.yaml --aggr max

# Atau menjalankan ablasi lengkap (Mean vs Max + t-SNE):
python run_all.py
```

Untuk smoke test cepat:
```powershell
python main.py --config config.reference.yaml --aggr max --nrows 50000
```

---

## 4. Berkas Output & Metrik Hasil

Hasil eksekusi akan tersimpan secara otomatis pada:
- **`result/exp22/exp22_mean_results.json`**: Metrik lengkap varian baseline `aggr='mean'`.
- **`result/exp22/exp22_max_results.json`**: Metrik lengkap varian proposed `aggr='max'`.
- **`result/exp22/exp22_mean_embeddings_sample.npz`**: Sampel representasi embedding test varian mean.
- **`result/exp22/exp22_max_embeddings_sample.npz`**: Sampel representasi embedding test varian max.
- **`result/exp22/embeddings_tsne_comparison.png`**: Grafik visual komparasi t-SNE separabilitas kelas (Mean vs Max).
- **`result/exp22/README.md`**: Rangkuman laporan analisis hasil resmi.
- **`model/exp22/`**: Model checkpoint PyTorch GNN dan pohon XGBoost.
