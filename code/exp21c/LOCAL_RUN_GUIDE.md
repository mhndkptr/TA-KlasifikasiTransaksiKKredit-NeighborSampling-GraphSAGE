# Panduan Menjalankan Eksperimen 21C secara Lokal (Local Run Guide)

Dokumen ini memandu langkah-langkah menjalankan **Eksperimen 21C (EXP21C: Multi-Window Temporal Splitting & Walk-Forward Validation)** di komputer lokal Anda (Windows PowerShell).

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
    └── exp21c\
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

Masuk ke folder `exp21c`:
```powershell
cd exp21c
```

### Opsi A: Quick Smoke Test (Uji Coba Fungsionalitas Cepat)
Gunakan argumen `--nrows 50000` dan `--windows window_2` untuk menguji alur pipeline pada satu horizon tanpa memakan waktu lama:
```powershell
python main.py --config config.reference.yaml --sampling uniform --nrows 50000 --windows window_2 --epochs 2
```

### Opsi B: Single Window Execution (Window Tertentu)
Menjalankan hanya pada Window 2 (Horizon EXP21 baseline):
```powershell
python main.py --config config.reference.yaml --sampling uniform --windows window_2
```

### Opsi C: Full Multi-Window Walk-Forward Execution (Window 1, 2, 3)
Menjalankan ketiga jendela waktu secara berurutan dan menghitung agregasi $\mu \pm \sigma$:
```powershell
python main.py --config config.reference.yaml --sampling uniform --windows all
```

### Opsi D: Menguji Metode Neighbor Sampling Tertentu
Pilihan sampling: `uniform`, `topology`, `importance`:
```powershell
# Topology-Aware Sampling
python main.py --config config.reference.yaml --sampling topology

# Importance-Based Sampling
python main.py --config config.reference.yaml --sampling importance
```

### Opsi E: Run All (Otomatisasi 3 Sampling + Visualisasi + Summary Markdown)
Menjalankan ketiga metode sampling, studi embargo gap, dan visualisasi secara berurutan:
```powershell
python run_all.py
```

---

## 4. Menghasilkan Visualisasi Hasil (Plots 300 DPI)

Setelah hasil eksekusi JSON dan CSV tersimpan di `../../result/exp21c`, Anda dapat menghasilkan grafik ilmiah secara mandiri:

```powershell
python visualize_results.py --result-dir ../../result/exp21c --output-dir ../../result/exp21c/plots
```

File gambar yang dihasilkan meliputi:
1. `exp21c_uniform_multi_window_performance.png` (Kinerja Walk-Forward Multi-Window dengan error bar)
2. `exp21c_uniform_cross_window_pr_roc.png` (Kurva PR & ROC antar jendela waktu)
3. `exp21c_uniform_embargo_gap_sensitivity.png` (Sensitivitas durasi jeda embargo)
4. `exp21c_uniform_gnn_lift_across_windows.png` (Nilai tambah representasi GraphSAGE)
5. `exp21c_uniform_gain_distribution_across_windows.png` (Distribusi gain keputusan GNN vs Tabular)

---

## 5. Output yang Dihasilkan

Setelah eksperimen selesai, folder `../../result/exp21c/` akan berisi:
* `exp21c_{sampling}_results.json` : Hasil lengkap metrik, threshold $\tau^*$, split gain, dan runtime.
* `exp21c_{sampling}_multi_window_summary.csv` : Tabel metrik per-window dan agregat $\mu \pm \sigma$.
* `exp21c_{sampling}_feature_subsets.csv` : Hasil ablasi Tabular Only vs GraphSAGE Only vs Hybrid.
* `exp21c_{sampling}_embargo_ablation.csv` : Hasil ablasi variasi gap 0, 8, dan 30 hari.
* `exp21c_{sampling}_predictions.npz` : Probabilitas kontinu dan label validasi/uji.
* `exp21c_summary_report.md` : Rangkuman laporan komparatif berformat tabel Markdown untuk Bab 4 Skripsi.
* `plots/` : Folder berisi grafik resolusi tinggi 300 DPI.
