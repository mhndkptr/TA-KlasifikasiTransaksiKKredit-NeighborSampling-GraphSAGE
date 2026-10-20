# Panduan Menjalankan Eksperimen 21A secara Lokal (Local Run Guide)

Dokumen ini memandu langkah-langkah menjalankan **Eksperimen 21A (EXP21A: Anti-Leakage & Clean Attribute Pipeline)** di komputer lokal Anda (Windows PowerShell).

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
    └── exp21a\
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

Masuk ke folder `exp21a`:
```powershell
cd exp21a
```

### Opsi A: Quick Smoke Test (Uji Coba Fungsionalitas Cepat)
Gunakan argumen `--nrows` untuk menguji alur pipeline tanpa memproses jutaan baris:
```powershell
python main.py --config config.reference.yaml --sampling uniform --nrows 50000
```

### Opsi B: Eksekusi Penuh Baseline Uniform
Menjalankan protokol penuh SHINE FRAUD II (300d train, 30d val, 8d gap, 60d test) dengan neighbor sampling Uniform pada fitur bersih (bebas Zip/City):
```powershell
python main.py --config config.reference.yaml --sampling uniform
```

### Opsi C: Eksekusi Seluruh Metode Sampling Sekaligus
Menjalankan ketiga metode sampling (`uniform`, `topology`, `importance`) secara sekuensial dan otomatis meng-generate grafik visualisasi:
```powershell
python run_all.py
```

### Opsi D: Generate Ulang Notebook Kaggle / Colab
Jika Anda melakukan modifikasi pada kode skrip Python:
```powershell
python create_nb.py; python create_colab_nb.py
```

---

## 4. Memeriksa Hasil Output

Setelah eksekusi selesai, artefak hasil akan tersimpan di:
* **Statistik Jendela Waktu:** `../../result/exp21a/window_statistics.json`
* **Metrik Evaluasi JSON:** `../../result/exp21a/exp21a_uniform_results.json`
* **Audit Feature Importance JSON:** `../../result/exp21a/exp21a_uniform_feature_importance.json`
* **Plot Visualisasi Ilmiah:** `../../result/exp21a/plots/`
* **Model Checkpoint:** `../../model/exp21a/exp21a_uniform_best_gnn.pt` dan `../../model/exp21a/exp21a_uniform_xgb.json`
