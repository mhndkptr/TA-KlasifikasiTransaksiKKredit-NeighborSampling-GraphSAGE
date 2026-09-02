# Experiment 6 - Imbalance Calibration dan TQDM

EXP6 mempertahankan exact tensor neighbor sampler dari EXP5 dan memperbaiki
training/evaluasi berdasarkan `result/exp5_uniform_seed42.json`.

Hasil full-data terbaru memuat 20.886 fraud train, 4.417 fraud validation, dan
4.454 fraud test. EXP5 mencapai validation AUPRC 0,6679, tetapi test AUPRC turun
ke 0,0237. Threshold tetap 0,5 menghasilkan 179.533 false positive dan recall
0,0855. Karena ranking validation sudah kuat, EXP6 mempertahankan full
positive-class weight EXP5 sebagai kontrol dan memusatkan perbaikan pada
threshold F1 yang dipilih hanya dari validation set, gradient clipping, serta
pelaporan validation-test gap. AUPRC tetap dihitung dari probabilitas sehingga
tidak dipengaruhi threshold keputusan.

Evaluasi default memakai satu sampling pass deterministik. Validation full-data
berisi 3,66 juta node, sehingga tiga pass pada setiap epoch akan menambah biaya
evaluasi sekitar tiga kali tanpa bukti bahwa hasil EXP5 memerlukannya.

Satu progress bar induk aktif sejak pemilihan runtime, pembacaan dataset,
feature engineering, pembangunan graf, seluruh kombinasi eksperimen, hingga
ringkasan selesai ditulis. Progress bar turunan tersedia untuk epoch, batch
training, batch evaluasi, audit split, dan audit long-tail. Log informasional
per-batch/per-epoch dinonaktifkan agar progress bar tetap bersih. Blok informasi
hardware dan runtime tetap ditampilkan satu kali sebelum progress bar dimulai.

EXP6 bersifat GPU-bound dan menggunakan `experiment.device: cuda`. Tidak ada
fallback CPU: program berhenti dengan diagnostik interpreter, versi PyTorch,
dan CUDA build bila environment yang aktif tidak mendukung CUDA.

Pembacaan CSV, feature engineering, dan pembangunan graf tetap berlangsung di
CPU, sehingga penggunaan GPU dapat terlihat 0% selama tahap awal tersebut.
Model, fitur, label, CSR graph, neighbor sampling, forward, backward, dan
optimizer mulai memakai GPU ketika progress bar `Training` aktif. Pantau saat
training dengan `nvidia-smi -l 1`.

Gunakan `.venv` milik EXP6 karena environment EXP1 dapat berisi build PyTorch
CPU-only. Verifikasi sebelum training:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Jalankan satu eksperimen dari folder ini:

```powershell
.\.venv\Scripts\python.exe run.py --config config.yaml --strategy uniform --seed 42
```

Validasi dengan subset yang masih memuat fraud pada ketiga temporal split:

```powershell
.\.venv\Scripts\python.exe run.py --config config.yaml --strategy uniform --seed 42 --max-rows 100000
```

Untuk smoke test GPU yang lebih cepat, pertahankan `--max-rows 100000` agar
ketiga temporal split tetap memuat fraud, lalu ubah sementara `training.epochs`
menjadi `1` di `config.yaml`.

Jalankan semua strategi dan seed:

```powershell
.\.venv\Scripts\python.exe run.py --config config.yaml
```
