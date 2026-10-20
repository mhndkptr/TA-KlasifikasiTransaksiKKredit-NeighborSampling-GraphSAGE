# Panduan Menjalankan Eksperimen 21B secara Lokal (Local Run Guide)

Dokumen ini memandu langkah-langkah menjalankan **Eksperimen 21B (EXP21B: Balanced Hyperparameter Tuning & Tree Regularization)** di komputer lokal Anda (Windows PowerShell).

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
    └── exp21b\
        ├── main.py
        ├── config.reference.yaml
        └── ...
```

---

## 2. Menyiapkan Virtual Environment

Buka PowerShell di direktori `code`:

```powershell
cd "d:\Projects\Klasifikasi Transasi Kartu Kredit - GraphSAGE\tugas-akhir-klasifikasi-transaksi-kartu-kredit\code"

# Buat venv baru jika belum ada
python -m venv .venv

# Aktifkan virtual environment
.\.venv\Scripts\activate

# Install dependensi yang dibutuhkan
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric
pip install pandas numpy scikit-learn xgboost pyyaml matplotlib
```

---

## 3. Menjalankan Eksperimen

Masuk ke folder `exp21b`:
```powershell
cd exp21b
```

### Opsi A: Quick Smoke Test (Uji Coba Fungsionalitas Cepat)
Gunakan argumen `--nrows 50000` dan `--tuning-mode fast` untuk menguji alur pipeline tanpa memakan waktu lama:
```powershell
python main.py --config config.reference.yaml --sampling uniform --nrows 50000 --tuning-mode fast
```

### Opsi B: Focused Ablation Run (~14 Kombinasi Hiperparameter)
Menjalankan protokol penuh temporal SHINE FRAUD II dengan mode ablasi hiperparameter satu per satu (cepat dan komprehensif):
```powershell
python main.py --config config.reference.yaml --sampling uniform --tuning-mode ablation
```

### Opsi C: Full Grid Search Run (108 Kombinasi Hiperparameter)
Menjalankan pencarian grid penuh pada 4 variasi SPW $\times$ 3 kedalaman $\times$ 3 min child weight $\times$ 3 colsample:
```powershell
python main.py --config config.reference.yaml --sampling uniform --tuning-mode grid
```

### Opsi D: Eksekusi Seluruh Metode Sampling Sekaligus (Run All)
Menjalankan ketiga metode sampling (`uniform`, `topology`, `importance`) secara sekuensial, mengeksekusi visualisasi otomatis, dan menghasilkan file ringkasan komparasi:
```powershell
python run_all.py --tuning-mode ablation
```

### Opsi E: Generate Ulang Notebook Kaggle / Colab
Jika Anda melakukan modifikasi pada kode skrip Python:
```powershell
python create_nb.py; python create_colab_nb.py
```

---

## 4. Memeriksa Hasil Output

Setelah eksekusi selesai, artefak hasil akan tersimpan di:
* **Statistik Jendela Waktu:** `../../result/exp21b/window_statistics.json`
* **Log Lengkap Tuning CSV:** `../../result/exp21b/exp21b_uniform_tuning_results.csv`
* **Hasil Evaluasi Model Terbaik JSON:** `../../result/exp21b/exp21b_uniform_results.json`
* **Studi Ablasi SPW & Subset Fitur JSON:** `../../result/exp21b/exp21b_uniform_ablation_summary.json`
* **Audit Feature Importance JSON:** `../../result/exp21b/exp21b_uniform_feature_importance.json`
* **Laporan Ringkasan Komparasi Markdown:** `../../result/exp21b/exp21b_summary_report.md`
* **Plot Visualisasi Ilmiah 300 DPI:** `../../result/exp21b/plots/`
* **Model Checkpoint:** `../../model/exp21b/exp21b_uniform_best_gnn.pt` dan `../../model/exp21b/exp21b_uniform_best_xgb.json`
