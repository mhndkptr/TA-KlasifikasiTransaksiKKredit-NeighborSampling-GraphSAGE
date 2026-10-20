# Panduan Menjalankan Eksperimen 21E secara Lokal (Local Run Guide)

Dokumen ini memandu langkah-langkah menjalankan **Eksperimen 21E (EXP21E: Pure End-to-End HeteroGraphSAGE)** di komputer lokal Anda (Windows PowerShell).

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
    └── exp21e\
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

Masuk ke folder `exp21e`:
```powershell
cd exp21e
```

### Opsi A: Quick Smoke Test (Uji Coba Cepat 50.000 Baris)
Gunakan argumen `--nrows` untuk menguji alur pipeline end-to-end secara instan:
```powershell
python main.py --config config.reference.yaml --sampling uniform --nrows 50000
```

### Opsi B: Eksekusi Penuh Baseline Uniform
Menjalankan klasifikasi murni GraphSAGE (Train 300d, Val 30d, Embargo Gap 8d, Test 60d pada Window 2018 aktif) dengan sampling Uniform:
```powershell
python main.py --config config.reference.yaml --sampling uniform
```

### Opsi C: Eksekusi Seluruh Metode Sampling Sekaligus (Rekomendasi Utama)
Menjalankan ketiga metode sampling (`uniform`, `topology`, `importance`) secara sekuensial, meng-generate master table CSV serta grafik visualisasi 300 DPI:
```powershell
python run_all.py
```

### Opsi D: Generate Ulang Notebook Kaggle / Colab
Jika Anda melakukan modifikasi pada skrip Python:
```powershell
python create_nb.py; python create_colab_nb.py
```

---

## 4. Memeriksa Hasil Output

Setelah eksekusi selesai, artefak hasil akan tersimpan di:
* **Statistik Jendela Waktu:** `../../result/exp21e/window_statistics.json`
* **Metrik Evaluasi JSON:** `../../result/exp21e/exp21e_uniform_results.json`
* **Tabel Master Komparasi CSV (Bab 4):** `../../result/exp21e/exp21e_master_comparison.csv`
* **Ringkasan Laporan Markdown:** `../../result/exp21e/exp21e_summary_report.md`
* **Plot Visualisasi Ilmiah 300 DPI:** `../../result/exp21e/plots/`
* **Model Checkpoint:** `../../model/exp21e/exp21e_uniform_best_gnn.pt`
