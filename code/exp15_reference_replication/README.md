# EXP15 — Replikasi Inductive Graph Representation Learning

EXP15 mengadaptasi pipeline referensi ke `dataset/credit_card_transactions-ibm_v2.csv`. Pipeline membangun graf client–transaction–merchant per rolling window, melatih HinSAGE dan FI-GRL, lalu membandingkan embedding, embedding + fitur transaksi, serta baseline fitur melalui classifier downstream.

Parameter referensi utama dipertahankan: window 17 hari bergeser 5 hari, holdout 5 hari, embedding 64, sampling HinSAGE `[2, 32]`, batch 50, 10 epoch, FI-GRL sketch 400, dan XGBoost 100 estimator. Preprocessing IBM dan aturan anchor kalender merupakan adaptasi yang dicatat di [PLAN.md](PLAN.md).

## Backend

`config.reference.yaml` menggunakan runtime sumber: StellarGraph, MATLAB Engine, dan XGBoost. `config.smoke.yaml` memakai backend kompatibilitas PyTorch, SciPy, dan sklearn agar wiring dapat diuji pada environment proyek. Artefak selalu menyimpan nama backend; hasil kompatibilitas tidak disebut sebagai hasil runtime referensi.

`config.compat.yaml` memakai dataset dan hyperparameter utama penuh dengan backend kompatibilitas. Preset ini berguna untuk audit engineering pada mesin saat ini, tetapi metriknya tetap dipisahkan di `result/exp15_compat/`.

Environment historis perlu dipisahkan karena StellarGraph 1.2.1 hanya mendukung Python di bawah 3.9. Buat environment dari `environment.reference.yml`, lalu pasang MATLAB Engine yang cocok dari instalasi MATLAB setempat.

## Menjalankan

Dari root repository:

```powershell
$python = 'python'

# Periksa dependency dan backend yang tersedia.
& $python .\code\exp15_reference_replication\run.py doctor

# Bekukan commit/hash sumber referensi.
& $python .\code\exp15_reference_replication\run.py audit-reference

# Scan CSV penuh, buat audit harian, dan kunci lima fold.
& $python .\code\exp15_reference_replication\run.py audit-data

# Jalankan seluruh fold eligible dan lima varian.
& $python .\code\exp15_reference_replication\run.py run

# Contoh satu fold/varian/seed.
& $python .\code\exp15_reference_replication\run.py run `
  --fold fold_1 --variant hinsage_plus_features --seed 42

# Bangun ulang ringkasan dari run lengkap.
& $python .\code\exp15_reference_replication\run.py summarize

# Test unit dan integration dengan backend kompatibilitas.
& $python -m unittest discover -s .\code\exp15_reference_replication\tests `
  -t .\code\exp15_reference_replication -v

# Smoke pada subset User0 dengan ketiga cabang utama kompatibilitas.
& $python .\code\exp15_reference_replication\run.py audit-data `
  --config .\code\exp15_reference_replication\config.smoke.yaml --no-progress
& $python .\code\exp15_reference_replication\run.py run `
  --config .\code\exp15_reference_replication\config.smoke.yaml --no-progress
```

`preprocess` saat ini identik dengan `audit-data`: CSV dipindai bertahap dan hanya dukungan harian serta manifest fold yang disimpan. Setiap `run` membaca kembali transaksi dalam window terpilih, sehingga graf 24 juta transaksi tidak pernah dibangun sekaligus.

Gunakan `--undersampling-rate 0.1` untuk rasio fraud:normal 1:10 pada graph training. Holdout tidak di-undersample. Nilai ini adalah pilihan eksplisit run; grid lengkap paper belum ditemukan pada artefak referensi dan tidak diklaim sebagai konfigurasi paper.

Setiap run menyimpan konfigurasi, environment, source manifest, sampled row IDs, pembagian internal HinSAGE, statistik graf, embedding, model classifier, prediksi, metrik, waktu, dan status. `result/exp15/summary.json` dan `runs.csv` hanya membaca run berstatus `complete`.

## Batas interpretasi

Graf inferensi berisi sampled train dan seluruh holdout lima hari, sama seperti notebook referensi. Ini adalah batch inductive, bukan inferensi online strictly-past. Preprocessing privat referensi tidak tersedia sehingga fitur IBM adalah adapter yang terdokumentasi. Hasil EXP12/EXP14 boleh menjadi konteks, tetapi bukan perbandingan terkontrol.
