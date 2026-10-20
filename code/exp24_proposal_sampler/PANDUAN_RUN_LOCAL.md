# Panduan Lengkap Eksekusi Lokal EXP24 di Windows 11 (Anaconda Prompt + GPU RTX 5060 Ti 16GB)

Dokumen ini berisi panduan *end-to-end* (mulai dari setup Anaconda Prompt, konfigurasi PyTorch khusus arsitektur Blackwell **RTX 5060 Ti (`sm_120`)**, instalasi dependensi, hingga eksekusi eksperimen dan pemantauan telemetri per-epoch).

### Spesifikasi Target Sistem:
- **Terminal Operasional**: **Anaconda Prompt** (Windows CMD)
- **Sistem Operasi**: Windows 11 (64-bit)
- **GPU**: NVIDIA GeForce RTX 5060 Ti (16 GB VRAM, Compute Capability `sm_120` Blackwell)
- **CPU**: Intel Core i7 Gen-10 (8 Core / 16 Thread)
- **RAM**: 32 GB DDR4

---

## 1. Verifikasi Driver & GPU di Anaconda Prompt

1. Buka **Anaconda Prompt** dari menu Start Windows.
2. Periksa driver GPU:
   ```cmd
   nvidia-smi
   ```
   *Pastikan Driver Version $\ge 550.xx$ dan CUDA Version menunjukkan $\ge 12.4$ atau $\ge 12.8$.*

---

## 2. Pengaturan Virtual Environment (`mhp_exp24_gpu`)

Gunakan environment Conda Anda (misalnya `mhp_exp24_gpu`):

1. **Aktifkan environment**:
   ```cmd
   conda activate mhp_exp24_gpu
   ```
   *(Jika belum dibuat, buat dengan: `conda create -n mhp_exp24_gpu python=3.10 -y` lalu aktifkan).*

2. **Catatan Mengenai Dependency Resolver Warning (Gambar Error pip)**:
   > [!NOTE] Penjelasan Error pip pada Gambar
   > Jika saat menjalankan `pip` muncul pesan teks merah seperti:
   > `ERROR: pip's dependency resolver does not currently take into account all the packages... open-clip-torch requires huggingface-hub... tensorflow requires requests... timm requires pyyaml...`
   > **STATUS: AMAN UNTUK DIABAIKAN.**
   > Pesan tersebut hanyalah peringatan dari library vision/NLP lama (`open-clip-torch`, `tensorflow`, `timm`) yang ada di environment Anda. Proyek tugas akhir kita (GraphSAGE + XGBoost) **sama sekali tidak menggunakan** library tersebut, melainkan hanya menggunakan `torch`, `torch_geometric`, `xgboost`, `scikit-learn`, dan `pandas`.

---

## 3. Konfigurasi Khusus PyTorch untuk RTX 5060 Ti (`sm_120`)

### Latar Belakang Masalah `sm_120`
Jika muncul warning:
```text
UserWarning: NVIDIA GeForce RTX 5060 Ti with CUDA capability sm_120 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_50 sm_60 sm_61 sm_70 sm_75 sm_80 sm_86 sm_90.
```
Penyebabnya adalah PyTorch versi default hanya memuat binary kernel hingga `sm_90` (Hopper/Ada). Karena **RTX 5060 Ti** menggunakan arsitektur **Blackwell (`sm_120`)**, kita perlu memasang PyTorch yang mendukung CUDA 12.8 (`cu128`).

### Langkah Solusi Rekomendasi (Instalasi PyTorch `cu128`):

Di **Anaconda Prompt** (dengan `mhp_exp24_gpu` aktif):

1. **Copot versi PyTorch yang lama**:
   ```cmd
   pip uninstall torch torchvision torchaudio -y
   ```

2. **Pasang PyTorch dengan dukungan CUDA 12.8 / Blackwell (`sm_120`)**:
   ```cmd
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
   ```
   *(Jika pip mencari versi nightly/preview terbaru yang sudah memuat `sm_120` secara penuh):*
   ```cmd
   pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
   ```

3. **Verifikasi Kompatibilitas Hardware**:
   Jalankan perintah pengujian satu baris berikut di Anaconda Prompt:
   ```cmd
   python -c "import torch; print('CUDA Tersedia:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0)); print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2), 'GB'); print('Test Tensor GPU:', torch.randn(2, 2, device='cuda'))"
   ```
   *Jika `Test Tensor GPU` mencetak matriks tensor tanpa melempar UserWarning atau RuntimeError, GPU RTX 5060 Ti Anda 100% siap.*

---

> [!TIP] Solusi Cepat Tanpa Reinstall (Workaround Driver JIT)
> Jika Anda belum ingin menginstal ulang PyTorch dan ingin langsung mencoba menjalankan kode, Anda dapat memaksa driver NVIDIA melakukan kompilasi JIT ke `sm_120` saat runtime.
> Di **Anaconda Prompt**, jalankan:
> ```cmd
> set CUDA_FORCE_PTX_JIT=1
> ```
> *(Perintah ini berlaku untuk sesi terminal Anaconda Prompt saat ini).*

---

## 4. Instalasi PyTorch Geometric & Dependensi Lainnya

Pastikan seluruh pustaka pendukung terpasang di environment `mhp_exp24_gpu`:

```cmd
pip install torch_geometric
pip install xgboost scikit-learn pandas numpy psutil pyyaml matplotlib tabulate
```

---

## 5. Navigasi Direktori di Anaconda Prompt

Di command prompt Windows / Anaconda Prompt, untuk berpindah ke drive `D:`, ketik huruf drive `d:` terlebih dahulu, kemudian perintah `cd`:

```cmd
d:
cd "d:\Projects\Klasifikasi Transasi Kartu Kredit - GraphSAGE\tugas-akhir-klasifikasi-transaksi-kartu-kredit\code\exp24_proposal_sampler"
```

Pastikan dataset IBM Credit Card tersedia pada jalur relatif:
`../../dataset/credit_card_transactions-ibm_v2.csv`

---

## 6. Uji Coba Pra-Pelatihan (*Smoke Test*)

### Langkah 6.1: Validasi 8 Correctness Gates (Tanpa Dataset)
Uji apakah seluruh formulasi sampling proposal (Jaccard, ASA-GNN, Local PPR, zero label leakage) lolos uji kontrak:
```cmd
python validate_sampler.py
```
Output yang diharapkan:
```text
[SUKSES] Seluruh 8 Correctness Gates EXP24 LULUS.
```

### Langkah 6.2: Smoke Test Cepat (30.000 Baris pada GPU)
Jalankan uji coba cepat dengan pembatasan baris untuk memverifikasi alur forward/backward pada GPU RTX 5060 Ti Anda:
```cmd
python main.py --config config.reference.yaml --sampling topology --seed 42 --nrows 30000 --device cuda
```
*Jika proses berhasil menyelesaikan 20 epoch GNN dan downstream XGBoost, sistem lokal Anda siap untuk menjalankan pengujian penuh.*

---

## 7. Eksekusi Pelatihan Penuh (*Full Multi-Seed Run*)

Dengan **RTX 5060 Ti 16GB VRAM** dan **RAM 32GB**, PC Anda mampu menjalankan dataset penuh secara lokal tanpa batasan baris (`nrows: null`).

### Menjalankan 5 Seed Berpasangan (15 Training Run Lengkap)
Perintah ini mengeksekusi 3 metode sampling (`uniform`, `topology`, `importance`) untuk 5 seed berpasangan (`42`, `100`, `2026`, `3407`, `8713`):

```cmd
python run_seed.py --config config.reference.yaml --seeds 42 100 2026 3407 8713 --device cuda
```

> [!NOTE] Fitur Resume Otomatis
> Jika di tengah jalan proses terhenti (misalnya Anda menghentikannya dengan `Ctrl+C` atau PC dimatikan), cukup jalankan kembali perintah di atas. `run_seed.py` secara otomatis mendeteksi run yang sudah selesai (via manifest) dan melanjutkan run berikutnya tanpa mengulang dari awal.

---

## 8. Pemantauan Real-Time & Pemeriksaan Telemetri Per-Epoch

Buka jendela **Anaconda Prompt kedua** untuk memantau proses:

### 8.1 Pemantauan Hardware GPU
```cmd
nvidia-smi -l 2
```
- **GPU-Util**: Akan berada pada kisaran 75% - 98% saat pelatihan GNN.
- **Memory-Usage**: Penggunaan VRAM berkisar antara **3,0 GB s.d. 5,5 GB** (sangat longgar di dalam kapasitas 16 GB VRAM Anda).

### 8.2 Memeriksa Telemetri Per-Epoch CSV Secara Langsung
Skrip secara otomatis mencatat metrik dan waktu setiap epoch ke dalam berkas CSV:
`..\..\result\exp24_proposal_sampler\seed_42\exp24_topology_training_history.csv`

Di Anaconda Prompt kedua, Anda dapat membaca baris-baris epoch terbaru menggunakan perintah `type` atau PowerShell:
```cmd
powershell -Command "Get-Content '..\..\result\exp24_proposal_sampler\seed_42\exp24_topology_training_history.csv' | Select-Object -Last 5"
```
Kolom yang tercatat meliputi:
`epoch`, `train_loss`, `val_loss`, `val_auprc`, `val_f1_optimal`, `val_precision_optimal`, `val_recall_optimal`, `val_optimal_threshold`, `val_fp`, `val_fn`, `time_train_epoch_sec`, `time_sampling_overhead_sec`, `val_latency_ms_per_1k_tx`, `peak_gpu_vram_mb`.

---

## 9. Agregasi Hasil & Pembuatan Grafik

Setelah seluruh 15 run selesai, jalankan agregasi untuk membuat tabel komparasi skripsi dan grafik beresolusi 300 DPI:

```cmd
python aggregate_exp24.py
```

### Hasil yang Terbentuk di `result/exp24_proposal_sampler/`:
1. **`exp24_multiseed_summary.csv`**: Rangkuman Mean, SD, dan Median metrik AUPRC, F1, Precision, Recall, FP, dan FN untuk ketiga metode.
2. **`exp24_paired_seed_deltas.csv`**: Perbandingan langsung berpasangan (*paired delta*) serta skor Win/Tie/Loss.
3. **`plots/exp24_paired_auprc_f1.png` & `.pdf`**: Grafik kurva AUPRC dan F1 per-seed dengan resolusi 300 DPI untuk Bab 4 Skripsi.

---

## 10. Tips Tambahan untuk Windows 11

1. **Pengaturan Power Plan Windows**:
   - Buka Windows Settings $\to$ System $\to$ Power $\to$ ubah Power Mode ke **Best Performance**.
2. **Kompabilitas Multiprocessing Windows**:
   - Skrip telah dikunci dengan `num_workers=0` pada DataLoader PyTorch Geometric. Hal ini disengaja untuk mencegah *deadlock* pada mekanisme `multiprocessing.spawn` bawaan OS Windows.
3. **Penyimpanan Drive D:**:
   - Pastikan drive `D:` menggunakan SSD (NVMe atau SATA) agar pembacaan dataset 2,35 GB dan penulisan embedding berjalan dalam hitungan detik.
