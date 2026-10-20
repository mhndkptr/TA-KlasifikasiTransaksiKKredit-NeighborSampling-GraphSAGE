# EXP21D-v4 FairCtrl-1A

Implementasi **Eksperimen 1A** dari `planning/12_eksplorasi_mendalam_exp21d_hybrid_xgboost.md`:

> `128-d Tabular MLP Embedding + XGBoost` dibandingkan dengan `128-d GraphSAGE Embedding + XGBoost` pada protokol temporal, loss, dimensi laten, dan konfigurasi XGBoost yang setara.

## Cakupan eksperimen

Satu run melatih MLP tabular dua hidden-layer dengan embedding laten 128-d, lalu melatih tiga downstream XGBoost:

1. `10-d Clean Tabular + XGBoost`;
2. `128-d MLP Embedding + XGBoost`;
3. `10-d Tabular + 128-d MLP Embedding + XGBoost`.

Hasil 128-d MLP dibandingkan dengan metrik `GraphSAGE Only` untuk sampler Uniform, Topology, dan Importance dari `result/exp21d/v1`. Perbandingan baseline ini menggunakan agregat metrik lama karena run v1 belum menyimpan embedding dan probabilitas GraphSAGE-only. Run baru menyimpan seluruh matriks fitur dan probabilitas sehingga analisis setelah ini dapat dilakukan tanpa training ulang.

## Menjalankan

Lingkungan lokal:

```powershell
cd code/exp21d_v4_fairctrl_1a
python run.py --config config.reference.yaml
```

Smoke test (hasil hanya untuk validasi teknis):

```powershell
python run.py --config config.reference.yaml --smoke --skip-tsne
```

Di Kaggle, upload repository/kode dan artefak `result/exp21d/v1`, lalu jalankan:

```bash
python run.py --config config.kaggle.yaml
```

`run.py` mencari dataset dan direktori baseline secara otomatis di `/kaggle/input`.

## Artefak yang dipertahankan

Setiap run mendapat direktori unik di `result/exp21d_v4_fairctrl_1a/runs/<run_id>/`:

```text
manifest.json                 status, seed, git revision, fingerprint dataset
resolved_config.yaml          konfigurasi persis yang dieksekusi
environment.json              versi Python/library/CUDA
events.jsonl                  log append-only dari awal sampai selesai
training_history.csv/json     metrik setiap epoch
xgboost_training_history.json logloss tiap boosting round untuk tiga model
preprocessing_artifacts.json  scaler, encoder train-only, OOV audit
models/                       checkpoint MLP dan tiga model XGBoost
features/                     *.npy mmap-compatible: tabular, embedding, hybrid, label, ID
predictions.npz               probabilitas validation/test untuk semua model
metrics.json                  seluruh metrik dan kurva threshold
fair_128d_comparison.csv      tabel utama kontrol 128-d vs 128-d
mlp_feature_sets_comparison.csv
figures/                      PNG dan PDF 300 DPI beserta CSV sumber plot
artifact_index.json           daftar seluruh file dan ukurannya
code_snapshot/                snapshot kode/config run
```

Matriks `.npy` sengaja tidak dikompresi agar dapat dibuka dengan memory mapping dan dipakai ulang pada metrik atau classifier tambahan tanpa memenuhi RAM.

## Menambah metrik tanpa training ulang

```powershell
python reanalyze.py --run-dir <run_dir> --min-precision 0.25 --tag precision25
python visualize_results.py --run-dir <run_dir>
```

`reanalyze.py` hanya membaca `predictions.npz` dan menulis hasil baru ke `reanalysis/`. Visualisasi membaca artefak yang sama dan tidak menyentuh checkpoint training.

## Paralelisasi keseluruhan rencana

Dokumen eksplorasi memuat **15 work package eksperimen** dalam **6 pilar**: Pilar 1 = 3, Pilar 2 = 2, Pilar 3 = 2, Pilar 4 = 2, Pilar 5 = 3, dan Pilar 6 = 3. Kelima belas work package itu dieksekusi dalam 4 tahap; jumlah job aktual lebih besar karena grid dan multi-seed. Sebagian besar dapat diparalelkan setelah dependensinya tersedia:

- eksperimen post-hoc, grid XGBoost, classifier benchmark, bootstrap, dan visualisasi dapat berjalan paralel dari embedding/prediksi yang sama;
- sampler dan seed GNN dapat berjalan paralel jika setiap proses mendapat GPU atau jatah VRAM terpisah;
- eksperimen yang memakai checkpoint/embedding harus menunggu artefak upstream selesai;
- tiga XGBoost dalam satu run ini dijalankan berurutan untuk membatasi puncak RAM, sedangkan seed yang berbeda aman dijalankan sebagai job terpisah.

Untuk satu GPU Kaggle, jalankan training GNN/MLP berurutan. Paralel CPU cocok untuk reanalisis dan pembuatan figure setelah file fitur tersedia.
