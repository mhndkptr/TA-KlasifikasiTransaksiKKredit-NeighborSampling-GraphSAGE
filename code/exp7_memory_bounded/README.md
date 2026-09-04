# Experiment 7 - Memory-Bounded Importance Sampling

EXP7 mempertahankan konfigurasi dan evaluasi EXP6, lalu membatasi temporary
tensor pada strategy `importance` agar local PPR tidak memenuhi VRAM.

## Evaluasi baseline EXP6

Artefak full-data yang tersedia baru seed 42. Karena itu, selisih berikut belum
bisa dianggap signifikan sebelum importance dan seed 43-46 selesai.

| Metrik | Uniform | Topology |
| --- | ---: | ---: |
| Best validation AUPRC | 0,664372 | 0,665948 |
| Test AUPRC | 0,023460 | 0,023791 |
| Test F1 | 0,041812 | 0,042358 |
| Test precision | 0,695652 | 0,769841 |
| Test recall | 0,021554 | 0,021778 |
| TP / FP | 96 / 42 | 97 / 29 |
| Peak VRAM | 6,410 GB | 7,755 GB |
| Durasi | 5,82 jam | 8,98 jam |

Topology menaikkan test AUPRC 1,41% dan F1 1,31% relatif terhadap uniform,
serta mengurangi 13 false positive pada threshold hasil kalibrasi. Biayanya
adalah peak VRAM 20,98% lebih tinggi dan durasi 54,27% lebih lama. Kedua
strategy memiliki validation-test AUPRC gap sekitar 0,64. Threshold validation
yang sangat tinggi (0,9967 dan 0,9986) juga hanya mempertahankan sekitar 2,2%
fraud pada temporal test. Jadi masalah dominan saat ini adalah generalisasi
temporal, bukan perbedaan sampler. Pada threshold tetap 0,5, topology mengurangi
false positive dari 176.122 menjadi 149.727 tetapi TP turun dari 364 menjadi
355.

Hasil full-data terbaru memuat 20.886 fraud train, 4.417 fraud validation, dan
4.454 fraud test. EXP5 mencapai validation AUPRC 0,6679, tetapi test AUPRC turun
ke 0,0237. Threshold tetap 0,5 menghasilkan 179.533 false positive dan recall
0,0855. Karena ranking validation sudah kuat, EXP7 mempertahankan full
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

EXP7 bersifat GPU-bound dan menggunakan `experiment.device: cuda`. Tidak ada
fallback CPU: program berhenti dengan diagnostik interpreter, versi PyTorch,
dan CUDA build bila environment yang aktif tidak mendukung CUDA.

## Cache preprocessing

`experiment.cache_graph: true` menyimpan hasil feature engineering, split,
label, dan edge graph sebagai tensor di `model/cache/`. Run pertama tetap
melakukan seluruh preprocessing; run strategy atau seed berikutnya dengan
dataset dan konfigurasi yang sama langsung memuat cache tersebut. Nama cache
memakai fingerprint berdasarkan path/ukuran/waktu-modifikasi dataset,
`max_rows`, temporal split, dan source preprocessing, sehingga konfigurasi yang
berbeda tidak memakai cache lama.

Cache full-data dapat berukuran beberapa GB. Untuk memaksa pembuatan ulang:

```powershell
..\exp6_gpu_tqdm\.venv\Scripts\python.exe run.py --config config.yaml --strategy uniform --seed 42 --rebuild-cache
```

Untuk membangun atau memvalidasi cache saja tanpa training:

```powershell
..\exp6_gpu_tqdm\.venv\Scripts\python.exe run.py --config config.yaml --preprocess-only
```

Pembacaan CSV, feature engineering, dan pembangunan graf tetap berlangsung di
CPU, sehingga penggunaan GPU dapat terlihat 0% selama tahap awal tersebut.
Model, fitur, label, CSR graph, neighbor sampling, forward, backward, dan
optimizer mulai memakai GPU ketika progress bar `Training` aktif. Pantau saat
training dengan `nvidia-smi -l 1`.

Strategy `importance` memakai local PPR yang sama secara matematis, tetapi
ekspansi walk dan kandidat diproses per `importance_edge_chunk_size`. Langkah
PPR terakhir mengambil mass melalui reverse adjacency (graf EXP7 dibangun
bidirectional), sehingga tidak lagi membentuk ekspansi root-node berukuran
beberapa GiB. Chunking ini hanya mengubah peak memory/kecepatan; fanout, bobot
importance, distribusi Gumbel-top-k, batch size, dan konfigurasi model tetap
sama dengan strategy lain. Nilai `batch_size: 4096` dipertahankan sesuai run
uniform dan topology yang sudah tersimpan.

Gunakan `.venv` milik EXP6 karena environment EXP1 dapat berisi build PyTorch
CPU-only. Verifikasi sebelum training:

```powershell
..\exp6_gpu_tqdm\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Jalankan satu eksperimen dari folder ini:

```powershell
..\exp6_gpu_tqdm\.venv\Scripts\python.exe run.py --config config.yaml --strategy importance --seed 42
```

Validasi dengan subset yang masih memuat fraud pada ketiga temporal split:

```powershell
..\exp6_gpu_tqdm\.venv\Scripts\python.exe run.py --config config.yaml --strategy importance --seed 42 --max-rows 100000
```

Untuk smoke test GPU yang lebih cepat, pertahankan `--max-rows 100000` agar
ketiga temporal split tetap memuat fraud, lalu ubah sementara `training.epochs`
menjadi `1` di `config.yaml`.

Jalankan semua strategi dan seed:

```powershell
..\exp6_gpu_tqdm\.venv\Scripts\python.exe run.py --config config.yaml
```
