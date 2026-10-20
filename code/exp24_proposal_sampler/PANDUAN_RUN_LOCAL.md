# Panduan Lengkap Eksekusi Lokal EXP24 di Windows 11 (GPU RTX 5060 Ti 16GB)

Dokumen ini berisi panduan *end-to-end* (mulai dari instalasi driver, environment Conda, PyTorch CUDA, dependensi PyG, hingga eksekusi eksperimen dan pemantauan telemetri per-epoch) yang dirancang khusus untuk spesifikasi perangkat keras Anda:
- **Sistem Operasi**: Windows 11 (64-bit)
- **GPU**: NVIDIA GeForce RTX 5060 Ti (16 GB VRAM)
- **CPU**: Intel Core i7 Gen-10 (8 Core / 16 Thread)
- **RAM**: 32 GB DDR4

---

## 1. Verifikasi Driver & GPU NVIDIA

Kartu grafis NVIDIA RTX 50-Series (arsitektur modern Blackwell) membutuhkan **CUDA 12.4 atau lebih tinggi**.

1. Buka terminal **PowerShell** (Run as Administrator jika perlu).
2. Periksa driver GPU:
   ```powershell
   nvidia-smi
   ```
   *Pastikan Driver Version $\ge 550.xx$ dan CUDA Version menunjukkan $\ge 12.4$. Jika belum, perbarui driver via GeForce Experience atau situs resmi NVIDIA.*

---

## 2. Pembuatan Virtual Environment Conda

Gunakan Anaconda/Miniconda yang telah terpasang di sistem Anda. Sebaiknya buat environment bersih dengan **Python 3.10** (versi paling stabil untuk kompilasi C++/CUDA PyTorch Geometric di Windows).

1. Buka PowerShell atau Anaconda Prompt:
   ```powershell
   conda create -n exp24_gpu python=3.10 -y
   ```
2. Aktifkan environment baru:
   ```powershell
   conda activate exp24_gpu
   ```
3. Verifikasi interpreter Python:
   ```powershell
   where.exe python
   ```
   *(Pastikan mengarah ke `C:\Users\<username>\anaconda3\envs\exp24_gpu\python.exe`)*

---

## 3. Instalasi PyTorch dengan Akselerasi CUDA 12.4

Jalankan instalasi PyTorch yang terkompilasi dengan **CUDA 12.4**:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Verifikasi Akselerasi Hardware GPU
Jalankan verifikasi cepat di PowerShell:
```powershell
python -c "import torch; print('PyTorch Version:', torch.__version__); print('CUDA Tersedia:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0)); print('VRAM Total:', round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2), 'GB')"
```
Output yang diharapkan:
```text
PyTorch Version: 2.4.x+cu124 (atau 2.5.x+cu124)
CUDA Tersedia: True
Device Name: NVIDIA GeForce RTX 5060 Ti
VRAM Total: 16.0 GB
```

---

## 4. Instalasi PyTorch Geometric (PyG) & Ekstensi Windows

Pada Windows 11, instalasi PyTorch Geometric dilakukan dengan menginstal pustaka inti dan binaries roda (*wheels*) yang cocok:

1. **Instalasi Pustaka Inti PyG**:
   ```powershell
   pip install torch_geometric
   ```

2. **Instalasi Ekstensi Terakselerasi C++/CUDA**:
   ```powershell
   pip install pyg_lib torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
   ```
   *(Catatan: Sesuaikan `torch-2.4.0` dengan versi PyTorch yang terpasang pada Langkah 3. Jika wheel belum tersedia untuk versi CUDA Anda, PyTorch Geometric versi 2.4+ sudah memiliki fallback native PyTorch C++ yang berjalan otomatis).*

3. **Instalasi Pustaka Pendukung Lainnya**:
   ```powershell
   pip install xgboost scikit-learn pandas numpy psutil pyyaml matplotlib tabulate
   ```

---

## 5. Verifikasi Dataset & Konfigurasi

Masuk ke folder eksperimen EXP24:
```powershell
cd "d:\Projects\Klasifikasi Transasi Kartu Kredit - GraphSAGE\tugas-akhir-klasifikasi-transaksi-kartu-kredit\code\exp24_proposal_sampler"
```

Pastikan dataset IBM Credit Card berada pada jalur relatif:
`../../dataset/credit_card_transactions-ibm_v2.csv`

Periksa apakah file dataset ada:
```powershell
Test-Path "../../dataset/credit_card_transactions-ibm_v2.csv"
```
Jika hasilnya `True`, dataset siap digunakan.

---

## 6. Uji Kelayakan & Validasi Awal (*Smoke Test*)

### Langkah 6.1: Validasi 8 Correctness Gates (Tanpa Dataset)
Uji apakah seluruh formulasi sampling proposal (Jaccard, ASA-GNN, Local PPR, zero label leakage) lolos uji kontrak:
```powershell
python validate_sampler.py
```
Output yang diharapkan:
```json
{
  "status": "passed",
  "checks": {
    "uniform_weights_are_strictly_one": true,
    "first_hop_remains_uniform": true,
    "zero_label_leakage_topology": true,
    "zero_label_leakage_importance": true,
    "topology_weights_bounded": true,
    "importance_weights_bounded": true,
    "neighbor_loader_uses_weight_attr": true,
    "topology_audit_passed": true,
    "importance_audit_passed": true
  }
}
[SUKSES] Seluruh 8 Correctness Gates EXP24 LULUS.
```

### Langkah 6.2: Smoke Test Cepat (Subset 30.000 Baris)
Jalankan uji coba cepat dengan pembatasan baris untuk memverifikasi alur forward/backward pada GPU Anda:
```powershell
python main.py --config config.reference.yaml --sampling topology --seed 42 --nrows 30000 --device cuda
```
*Jika proses selesai hingga evaluasi XGBoost dan membuat file CSV telemetri, maka sistem lokal Anda 100% siap untuk full run.*

---

## 7. Eksekusi Pelatihan Penuh (*Full Multi-Seed Run*)

Dengan **RTX 5060 Ti 16GB VRAM** dan **RAM 32GB**, komputer Anda mampu menjalankan seluruh dataset secara lokal tanpa downsampling graf (`nrows: null`).

### Menjalankan 5 Seed Berpasangan (15 Training Run)
Perintah ini akan menjalankan 3 metode sampling (`uniform`, `topology`, `importance`) secara otomatis untuk 5 seed (`42`, `100`, `2026`, `3407`, `8713`):

```powershell
python run_seed.py --config config.reference.yaml --seeds 42 100 2026 3407 8713 --device cuda
```

> [!TIP] Fitur Resume Otomatis
> Jika di tengah jalan proses terputus (misalnya listrik padam atau PC dimatikan), Anda cukup menjalankan kembali perintah di atas. Skrip `run_seed.py` secara otomatis mendeteksi run yang sudah selesai via `manifest.json` dan melanjutkan run yang belum selesai tanpa mengulang dari awal.

---

## 8. Pemantauan Real-Time & Pemeriksaan Telemetri Per-Epoch

Saat pelatihan berjalan di satu jendela terminal, Anda dapat memantau aktivitas sistem di terminal PowerShell kedua:

### 8.1 Pemantauan Hardware GPU
```powershell
nvidia-smi -l 2
```
Perhatikan metrik:
- **GPU-Util**: Akan berkisar antara 75% - 98% saat pelatihan GNN.
- **Memory-Usage**: Untuk fan-out `[15, 10]` dan batch size 1024, penggunaan VRAM berkisar antara **3,0 GB s.d. 5,5 GB** (sangat aman di dalam kapasitas 16 GB VRAM Anda).

### 8.2 Membaca Telemetri Per-Epoch CSV Secara Langsung
Skrip menyimpan berkas CSV telemetri secara *real-time* di:
`result/exp24_proposal_sampler/seed_42/exp24_topology_training_history.csv`

Anda dapat melihat 5 baris epoch terakhir secara langsung via PowerShell:
```powershell
Get-Content "..\..\result\exp24_proposal_sampler\seed_42\exp24_topology_training_history.csv" | Select-Object -Last 6
```
Kolom yang tersedia meliputi:
`epoch`, `train_loss`, `val_loss`, `val_auprc`, `val_f1_optimal`, `val_precision_optimal`, `val_recall_optimal`, `time_train_epoch_sec`, `time_sampling_overhead_sec`, `val_latency_ms_per_1k_tx`, `peak_gpu_vram_mb`.

---

## 9. Agregasi Hasil & Pembuatan Grafik

Setelah seluruh 15 run selesai, jalankan skrip agregasi untuk menghasilkan tabel komparasi dan plot grafik publikasi:

```powershell
python aggregate_exp24.py
```

### Hasil Artefak Akhir di `result/exp24_proposal_sampler/`:
1. **`exp24_multiseed_summary.csv`**: Rangkuman Mean, SD, dan Median metrik AUPRC, F1, Precision, Recall, FP, dan FN untuk ketiga metode.
2. **`exp24_paired_seed_deltas.csv`**: Analisis berpasangan (*paired delta*) serta skor Win/Tie/Loss antar-metode.
3. **`plots/exp24_paired_auprc_f1.png` & `.pdf`**: Grafik kurva AUPRC dan F1 per-seed dengan resolusi 300 DPI untuk Bab 4 Skripsi.

---

## 10. Tips Optimalisasi Performa Windows 11

1. **Pengaturan Power Windows**:
   - Pastikan power plan Windows 11 disetel ke **High Performance** atau **Best Performance** agar CPU Core i7 tidak melakukan throttling frekuensi selama pemrosesan data tabular.
2. **Kompabilitas Threading CPU**:
   - Skrip telah dikonfigurasi dengan `num_workers=0` pada DataLoader PyTorch Geometric. Hal ini disengaja untuk mencegah *deadlock* pada mekanisme `multiprocessing.spawn` bawaan OS Windows.
3. **Penyimpanan SSD**:
   - Karena dataset berukuran ~2,35 GB dan matriks fitur disimpan ke disk, pastikan drive `D:` adalah SSD NVMe / SATA agar loading data selesai dalam waktu < 20 detik.
