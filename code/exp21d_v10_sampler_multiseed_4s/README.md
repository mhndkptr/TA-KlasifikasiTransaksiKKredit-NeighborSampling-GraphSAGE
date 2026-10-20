# EXP21D-v10 — Paired Multi-Seed Sampler Comparison (EXP 4S)

Pipeline ini memperluas corrected EXP 2A menjadi 15 paired runs:

```text
samplers = [uniform, topology, importance]
seeds    = [42, 100, 2026, 3407, 8713]
```

Satu paired block menjalankan tiga sampler dengan seed training dan evaluation seed yang sama. Seluruh hyperparameter lain dikunci. Tabular-only baseline hanya dihitung pada run Uniform, satu kali per seed.

## Batch pertama Kaggle

Notebook disiapkan untuk menjalankan tiga paired block secara berurutan dalam satu job:

```powershell
python run_seed.py --config config.kaggle.yaml --seeds 42 100 2026
```

Perintah tersebut menghasilkan sembilan training run: tiga sampler untuk setiap seed. Seed 3407 dan 8713 menjadi batch lanjutan. Resume aktif sehingga kombinasi seed-sampler dengan manifest dan indeks artefak lengkap akan dilewati saat job dijalankan ulang.

Batch lanjutan memakai kernel terpisah di `kaggle_extra_seeds`:

```powershell
python run_seed.py --config config.kaggle.extra.yaml --seeds 3407 8713
```

Kernel lanjutan menghasilkan enam training run dan menyimpan output ke `exp21d_v10_sampler_multiseed_4s_extra`, sehingga artefak batch tiga seed pertama tidak tertimpa.

## Artefak per run

- model GNN/XGBoost, history, sampler audit, resolved config, stdout log, manifest;
- raw validation/test probabilities dan labels;
- test predictions dengan transaction ID, source row, timestamp, dan hash user/merchant;
- runtime, peak RSS, peak GPU memory, prediction SHA-256, dan artifact index;
- validation-selected threshold tanpa tuning dari test.

## Agregasi

Setelah setiap paired block, agregator membaca seluruh seed yang tersedia dan menulis:

- per-run metrics termasuk Recall@K dan Precision@K untuk K 100/200/300/421/500/1000;
- mean, SD, median;
- paired seed delta dan win/tie/loss;
- status `partial` sampai 15 run lengkap;
- plot paired AUPRC/F1 dalam PNG/PDF.

Folder ini belum memiliki `kernel-metadata.json`, belum dipush, dan belum menjalankan training 15-run.
