# EXP13 — Kaggle T4 deployment and proposal-aligned baseline

EXP13 adalah lapisan eksperimen dan deployment yang memakai engine ilmiah
EXP12 yang sudah diuji, lalu mengunci konfigurasi tugas akhir ke pipeline
out-of-core dan NVIDIA Tesla T4. Source modular tetap berada di repository;
`build_kaggle_kernel.py` membungkus source itu secara deterministik menjadi satu
script yang diterima `kaggle kernels push`.

Run pertama telah dibuat sebagai kernel privat:

- `muhammadhendikaputra/graphsage-fraud-exp13-uniform-s42`
- strategy `uniform`, seed 42
- accelerator `NvidiaTeslaT4`

## Struktur

```text
exp13_try_kaggle/
|-- exp13/engine.py             # adapter + hash source engine EXP12
|-- config.yaml                # konfigurasi proposal-aligned EXP13
|-- run.py                     # entry point lokal
|-- build_kaggle_kernel.py     # deterministic single-file builder
|-- deploy_kaggle.py           # build + push aman melalui Kaggle CLI
|-- kaggle/
|   |-- kernel.py              # generated, tanpa credential
|   `-- kernel-metadata.json   # generated, T4 + dataset source
|-- tests/test_deployment.py   # invariant GPU/security/config
|-- PRIOR_EXPERIMENT_AUDIT.md
`-- VALIDATION.md
```

EXP13 sengaja tidak menyalin seluruh package EXP12. Builder memasukkan snapshot
source beserta SHA-256 setiap file ke `EXP13_SOURCE_MANIFEST.json`. Dengan pola
ini, perbaikan engine tetap mempunyai satu sumber kebenaran, tetapi setiap run
EXP13 masih dapat direproduksi dan diaudit secara tepat.

## Credential Kaggle

Credential lokal sudah berada pada `kaggle.json`. File itu diabaikan oleh
`.gitignore` pada level repository dan folder EXP13. Script hanya mengarahkan
`KAGGLE_CONFIG_DIR` ke folder ini. API key tidak pernah dimasukkan ke command
line, metadata, bundle, log, atau output JSON.

Jangan memindahkan `kaggle.json` ke folder `kaggle/`, karena seluruh isi folder
tersebut diunggah. Jika key pernah ter-commit atau dibagikan, segera revoke dan
buat token baru dari pengaturan akun Kaggle.

## Menjalankan

Dari root repository, gunakan interpreter virtual environment yang sama:

```powershell
$py = ".\code\exp6_gpu_tqdm\.venv\Scripts\python.exe"

# Unit test deployment dan keamanan
& $py -B -m unittest discover `
  -s .\code\exp13_try_kaggle\tests `
  -t .\code\exp13_try_kaggle -v

# Build saja; tidak memakai kuota GPU
& $py -B .\code\exp13_try_kaggle\deploy_kaggle.py `
  --strategy uniform --seed 42 --build-only

# Build, push, dan mulai kernel T4
& $py -B .\code\exp13_try_kaggle\deploy_kaggle.py `
  --strategy uniform --seed 42
```

Identifier resmi Kaggle adalah `NvidiaTeslaT4`, bukan `NvidiaGPUT4`.
EXP13 menetapkannya dua kali: `machine_shape` pada metadata dan flag
`--accelerator NvidiaTeslaT4` saat push. Kernel juga berhenti sebelum
preprocessing bila CUDA tidak tersedia atau nama GPU aktual tidak memuat `T4`.

Status, log, dan output:

```powershell
$env:KAGGLE_CONFIG_DIR = (Resolve-Path .\code\exp13_try_kaggle).Path
$kaggle = ".\code\exp6_gpu_tqdm\.venv\Scripts\kaggle.exe"
$ref = "muhammadhendikaputra/graphsage-fraud-exp13-uniform-s42"

& $kaggle kernels status $ref
& $kaggle kernels logs $ref
& $kaggle kernels output $ref -p .\result\kaggle\exp13_uniform_s42
```

Setelah baseline selesai sehat, buat kernel terpisah agar hasil tidak saling
menimpa:

```powershell
& $py -B .\code\exp13_try_kaggle\deploy_kaggle.py --strategy topology --seed 42
& $py -B .\code\exp13_try_kaggle\deploy_kaggle.py --strategy importance --seed 42
```

Untuk klaim tugas akhir, ulangi konfigurasi yang dibekukan pada sedikitnya seed
42–46. Jangan mengubah feature set, root sampling, loss, atau validation policy
di tengah perbandingan tiga sampler.

## Kontrak metodologis

- Split 70/15/15 dilakukan setelah external chronological sort. Timestamp yang
  sama tidak dipotong di dua split.
- Encoder, normalisasi, degree, fraud history, Jaccard, homophily, dan PPR hanya
  di-fit/dihitung dari prefix train.
- Validation/test adalah query roots terhadap snapshot graph train yang beku;
  edge maupun label held-out tidak memperbarui sampler.
- Training memakai distribusi root alami dan `pos_weight = N_neg/N_pos` yang
  dihitung dari train. Jangan sekaligus memakai balanced roots karena akan
  menggandakan koreksi imbalance.
- Model mengeluarkan logits dan memakai `BCEWithLogitsLoss`; Sigmoid hanya saat
  inferensi. Ini ekuivalen dengan output sigmoid secara probabilistik tetapi
  lebih stabil secara numerik.
- AUPRC pada validation selection memilih checkpoint. Bagian validation akhir
  yang tidak overlap dipakai untuk threshold F2. Test baru dievaluasi setelah
  keduanya terkunci.

## Memori Kaggle

Preset EXP13 memakai DuckDB external sort (`memory_limit=8GB`), chunk 250.000,
fitur proposal float16 pada memmap CPU, dan hanya mini-batch float32 yang masuk
GPU. Adjacency disimpan sebagai CSR, bukan list Python per node. Setelah run
berhasil, cache preprocessing reproducible (~beberapa GiB) dihapus dari output
Kaggle; checkpoint, metrics, history, konfigurasi, dan source manifest tetap
disimpan.

Preset ini sengaja memakai 15 fitur compact agar aman pada RAM Kaggle. Ia bukan
reproduksi feature-rich EXP12 yang memakai 220 fitur dan sekitar 9,99 GiB untuk
feature store saja. Karena itu, angka full-data EXP12 tidak boleh diklaim sebagai
hasil EXP13. Lihat `PRIOR_EXPERIMENT_AUDIT.md` untuk perbandingan dan batasnya.

## Definisi sampler yang perlu ditulis jujur

Pada heterograf `Transaction—User/Merchant`, degree literal setiap Transaction
selalu dua dan Jaccard neighborhood antar tipe literal bernilai nol. Implementasi
operasional memakai:

- uniform: weighted-without-replacement dengan semua bobot sama;
- topology: Jaccard himpunan transaksi historis kedua endpoint dan local
  homophily train-only yang leave-one-out;
- importance: projected transaction degree dan local PPR tiga langkah pada
  snapshot train.

Ini adalah adaptasi eksplisit terhadap struktur bipartit, bukan implementasi
literal metrik yang degenerat. Selain itu, dari fanout `[25, 10]`, root
Transaction hanya mempunyai dua endpoint; sampling yang benar-benar memangkas
adalah sisi entity ke transaksi historis dengan `k=25`.
