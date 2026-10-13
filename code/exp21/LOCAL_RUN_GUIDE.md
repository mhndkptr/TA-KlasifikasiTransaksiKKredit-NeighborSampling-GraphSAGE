# Panduan Menjalankan Eksperimen 21 secara Lokal (Local Run Guide)

Dokumen ini memandu langkah-langkah menjalankan **Eksperimen 21 (EXP21: Temporal Window Alignment & Inductive Setup)** di komputer lokal Anda (Windows PowerShell).

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
    └── exp21\
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
pip install pandas numpy scikit-learn xgboost pyyaml
```

---

## 3. Menjalankan Eksperimen

Masuk ke folder `exp21`:
```powershell
cd exp21
```

### Opsi A: Quick Test / Smoke Test (Rekomendasi Uji Coba Cepat)
Gunakan argumen `--nrows` untuk menguji fungsionalitas pipeline secara instan tanpa memproses seluruh data era 2016–2020:
```powershell
python main.py --config config.reference.yaml --sampling uniform --nrows 50000
```

### Opsi B: Eksekusi Penuh Baseline Uniform (Rekomendasi Utama EXP21)
Menjalankan protokol penuh SHINE FRAUD II (300 hari train, 30 hari val, 8 hari gap, 60 hari test) dengan neighbor sampling Uniform:
```powershell
python main.py --config config.reference.yaml --sampling uniform
```

### Opsi C: Override Tanggal Referensi ($T_0$)
Jika ingin menguji tanggal $T_0$ lain (misal awal tahun 2016):
```powershell
python main.py --config config.reference.yaml --sampling uniform --reference-date 2016-01-01
```

### Opsi D: Eksekusi Seluruh Metode Sampling Sekaligus
Menjalankan ketiga metode sampling (`uniform`, `topology`, `importance`) secara sekuensial:
```powershell
python run_all.py
```

---

## 4. Memeriksa Hasil Output

Setelah eksekusi selesai, artefak hasil akan tersimpan di:
* **Statistik Jendela Waktu:** `../../result/exp21/window_statistics.json`
* **Metrik Evaluasi JSON:** `../../result/exp21/exp21_uniform_results.json`
* **Model Checkpoint:** `../../model/exp21/exp21_uniform_gnn.pt` dan `../../model/exp21/exp21_uniform_xgb.json`
