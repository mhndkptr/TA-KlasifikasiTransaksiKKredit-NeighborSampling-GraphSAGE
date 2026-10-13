# Panduan Eksekusi Lokal EXP23: Pure End-to-End HeteroGraphSAGE

Panduan ini berisi instruksi langkah demi langkah untuk menjalankan eksperimen `EXP23` di komputer lokal Windows menggunakan PowerShell.

---

## 1. Persiapan Environment

Buka terminal PowerShell dan aktifkan virtual environment proyek:

```powershell
cd "d:\Projects\Klasifikasi Transasi Kartu Kredit - GraphSAGE\tugas-akhir-klasifikasi-transaksi-kartu-kredit"
.\code\.venv\Scripts\Activate.ps1
cd code\exp23
```

Pastikan dependensi dasar tersedia:
```powershell
python -c "import torch, torch_geometric; print('PyTorch:', torch.__version__, '| PyG:', torch_geometric.__version__)"
```

---

## 2. Smoke Test Cepat (Debug Mode)

Untuk memvalidasi bahwa seluruh modul dan fungsi inferensi GNN berjalan lancar tanpa error memori, jalankan smoke test dengan membatasi baris data (misal 50.000 baris):

```powershell
python main.py --config config.reference.yaml --sampling uniform --nrows 50000
```

---

## 3. Eksekusi Penuh (Single Sampling Method)

Jalankan salah satu metode neighbor sampling pada dataset penuh era modern 2016–2020:

```powershell
# 1. Baseline Uniform Sampling
python main.py --config config.reference.yaml --sampling uniform

# 2. Topology-Aware Sampling
python main.py --config config.reference.yaml --sampling topology

# 3. Importance-Based Sampling
python main.py --config config.reference.yaml --sampling importance
```

---

## 4. Eksekusi Otomatis Seluruh Metode (run_all.py)

Untuk mengeksekusi ketiga metode secara berurutan dan langsung menghasilkan 16 chart visualisasi hasil:

```powershell
python run_all.py
```

---

## 5. Menghasilkan Visualisasi Manual

Jika seluruh hasil JSON dan NPZ sudah terbentuk di `../../result/exp23`:

```powershell
python visualize_results.py --result-dir ../../result/exp23 --output-dir ../../result/exp23/plots
```

Semua visualisasi resolusi tinggi (300 DPI) akan tersimpan di `result/exp23/plots/`.
