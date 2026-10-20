# Panduan Menjalankan Eksperimen 21D secara Lokal (Local Run Guide)

Dokumen ini memandu langkah-langkah menjalankan **Eksperimen 21D (EXP21D: The Golden Master Hybrid Pipeline)** di komputer lokal Anda (Windows PowerShell).

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
    └── exp21d\
        ├── main.py
        ├── config.reference.yaml
        └── ...
```

---

## 2. Menyiapkan Virtual Environment

Buka PowerShell di direktori `code`:

```powershell
cd "d:\Projects\Klasifikasi Transasi Kartu Kredit - GraphSAGE\tugas-akhir-klasifikasi-transaksi-kartu-kredit\code"

# Aktifkan virtual environment
.\.venv\Scripts\activate
```

---

## 3. Menjalankan Eksperimen

Masuk ke folder `exp21d`:
```powershell
cd exp21d
```

### Opsi A: Quick Smoke Test (Uji Coba Cepat 50.000 Baris)
Gunakan argumen `--nrows` untuk menguji alur pipeline secara instan tanpa memproses jutaan baris:
```powershell
python main.py --config config.reference.yaml --sampling uniform --nrows 50000
```

### Opsi B: Eksekusi Penuh Baseline Uniform
Menjalankan protokol penuh Golden Master (Train 300d, Val 30d, Embargo Gap 8d, Test 60d pada Window 2018 aktif) dengan neighbor sampling Uniform pada fitur bersih:
```powershell
python main.py --config config.reference.yaml --sampling uniform
```

### Opsi C: Eksekusi Seluruh Metode Sampling & Studi Ablasi Sekaligus (Rekomendasi Utama)
Menjalankan ketiga metode sampling (`uniform`, `topology`, `importance`) secara sekuensial, mengeksekusi studi ablasi subset fitur (Tabular vs GNN vs Hybrid), dan otomatis meng-generate master table serta grafik visualisasi:
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
* **Statistik Jendela Waktu:** `../../result/exp21d/window_statistics.json`
* **Metrik Evaluasi JSON:** `../../result/exp21d/exp21d_uniform_results.json`
* **Tabel Master Komparasi CSV (Bab 4):** `../../result/exp21d/exp21d_master_comparison.csv`
* **Tabel Ablasi Nilai Tambah GNN CSV:** `../../result/exp21d/exp21d_feature_subsets_comparison.csv`
* **Audit Feature Importance JSON:** `../../result/exp21d/exp21d_uniform_feature_importance.json`
* **Ringkasan Laporan Markdown:** `../../result/exp21d/exp21d_summary_report.md`
* **Plot Visualisasi Ilmiah 300 DPI:** `../../result/exp21d/plots/`
* **Model Checkpoint:** `../../model/exp21d/exp21d_uniform_best_gnn.pt` dan `../../model/exp21d/exp21d_uniform_xgb.json`
