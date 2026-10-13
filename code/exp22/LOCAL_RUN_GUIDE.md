# Panduan Menjalankan Eksperimen 22 secara Lokal (Local Run Guide)

Dokumen ini memandu langkah-langkah menjalankan **Eksperimen 22 (EXP22: GNN Aggregator Innovation: Elementwise Maximization vs Mean)** di komputer lokal Anda (Windows PowerShell).

---

## 1. Lokasi Dataset

Pastikan file dataset berada di:
`../../dataset/credit_card_transactions-ibm_v2.csv`

Struktur folder:
```text
D:\Projects\Klasifikasi Transasi Kartu Kredit - GraphSAGE\tugas-akhir-klasifikasi-transaksi-kartu-kredit\
├── dataset\
│   └── credit_card_transactions-ibm_v2.csv
└── code\
    └── exp22\
        ├── main.py
        ├── visualize_tsne.py
        ├── config.reference.yaml
        └── ...
```

---

## 2. Menyiapkan Virtual Environment

Buka PowerShell di direktori `code`:

```powershell
cd "d:\Projects\Klasifikasi Transasi Kartu Kredit - GraphSAGE\tugas-akhir-klasifikasi-transaksi-kartu-kredit\code"

# Aktifkan virtual environment jika sudah ada
.\.venv\Scripts\activate

# Install dependensi yang dibutuhkan (jika belum terpasang)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric
pip install pandas numpy scikit-learn xgboost pyyaml matplotlib
```

---

## 3. Menjalankan Eksperimen

Masuk ke folder `exp22`:
```powershell
cd exp22
```

### Opsi A: Quick Test / Smoke Test (Rekomendasi Uji Coba Cepat)
Gunakan argumen `--nrows` untuk menguji fungsionalitas agregator Max pooling dan t-SNE secara instan tanpa memproses seluruh data:
```powershell
python main.py --config config.reference.yaml --aggr max --nrows 50000
```

### Opsi B: Eksekusi Model Usulan (Proposed Max Pooling)
Menjalankan model GraphSAGE dengan agregasi `aggr='max'` (Elementwise Maximization) pada protokol temporal penuh SHINE FRAUD II:
```powershell
python run.py
# atau:
python main.py --config config.reference.yaml --aggr max
```

### Opsi C: Eksekusi Baseline Mean Aggregator
Menjalankan model GraphSAGE pembanding dengan agregasi `aggr='mean'`:
```powershell
python main.py --config config.reference.yaml --aggr mean
```

### Opsi D: Eksekusi Seluruh Varian & Otomasi Visualisasi t-SNE (Rekomendasi Lengkap)
Menjalankan komparasi `mean` dan `max` secara berurutan, lalu secara otomatis memproduksi grafik perbandingan t-SNE:
```powershell
python run_all.py
```

### Opsi E: Menghasilkan Visualisasi t-SNE Secara Manual
Jika data embedding sample (`exp22_mean_embeddings_sample.npz` dan `exp22_max_embeddings_sample.npz`) telah dibuat:
```powershell
python visualize_tsne.py --result-dir ../../result/exp22
```

---

## 4. Memeriksa Hasil Output

Setelah eksekusi selesai, artefak hasil akan tersimpan di:
* **Metrik Evaluasi JSON:** 
  * `../../result/exp22/exp22_mean_results.json`
  * `../../result/exp22/exp22_max_results.json`
* **Plot t-SNE Resolusi Tinggi:** `../../result/exp22/embeddings_tsne_comparison.png`
* **Model Checkpoint:** `../../model/exp22/exp22_max_gnn.pt` dan `../../model/exp22/exp22_max_xgb.json`
