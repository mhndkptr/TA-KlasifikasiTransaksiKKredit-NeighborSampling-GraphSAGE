# EXP21E-v4 — Causal Temporal Pilot

EXP21E-v4 adalah quality gate kedua sebelum eksperimen utama multi-seed V5. Versi ini
menjalankan tiga pilot (`uniform`, `topology`, dan `importance`) dengan seed 42 menggunakan
evaluasi daily-batch yang kausal. Metrik V4 tidak dimaksudkan sebagai hasil final paper.

## Protokol kausal

Untuk setiap hari kalender `d` pada validation dan test:

1. entity aggregate dan degree dihitung dari seluruh transaksi dengan timestamp sebelum awal hari `d`;
2. graph inference dibekukan selama seluruh transaksi hari `d` dinilai;
3. transaksi hari `d` hanya menjadi root dan tidak dapat menjadi candidate context bagi transaksi lain pada hari yang sama;
4. setelah inference selesai, seluruh transaksi hari `d` ditambahkan ke history;
5. label target tidak pernah dibaca oleh history state, aggregate, degree, atau sampling score.

Graph harian hanya mematerialisasi exact two-hop context yang relevan bagi root hari tersebut.
Global historical aggregate dan counterpart degree tetap dihitung dari seluruh history, sehingga
optimasi ini tidak mengubah semantics fan-out dua lapis `[15, 10]`.

Validation memakai daily protocol pada setiap epoch untuk pemilihan checkpoint. Setelah best
checkpoint dimuat ulang, threshold dipilih hanya dari validation. Graph test baru dibangun setelah
itu dan test diinferensikan tepat satu kali.

## Menjalankan validasi kausal

```powershell
python causal_validation.py --output ../../result/exp21e/v4/causal_validation_report.json
```

Validasi ini memeriksa cutoff, exclusion current-day context, label independence, aggregate
independence, update history setelah inference, root ordering, dan penolakan future history.

## Menjalankan smoke test lokal

```powershell
python main.py --config config.reference.yaml --sampling uniform --seed 42 --smoke
```

Smoke test membatasi data menjadi 20.000 baris per split, dua epoch, dan dua hari evaluasi per
split. Smoke hanya menguji integrasi.

## Menjalankan pilot lengkap

```powershell
python run_all.py --config config.reference.yaml --skip-install
```

Di Kaggle, notebook menjalankan `run_all.py` tanpa `--smoke`, sehingga ketiga strategi memakai
window 300/30/8/60 hari lengkap dan 20 epoch budget yang sama.

## Artefak

```text
result/exp21e/v4/
  causal_validation_report.json
  v4_pilot_summary.csv
  exp21e_v4_<window>_<strategy>_seed42_<config_hash>/
    manifest.json
    resolved_config.yaml
    environment.json
    window_statistics.json
    causal_temporal_audit.json
    sampler_audit.json
    metrics.json
    training_history.json
    predictions.csv.gz
    predictions.npz
    best_checkpoint.pt
    code_snapshot/
```

## Acceptance criteria menuju V5

- seluruh hari validation dan test lulus no-future audit;
- threshold hanya berasal dari validation;
- test dievaluasi sekali setelah checkpoint dan threshold dibekukan;
- transaction ID prediction identik dan dapat disejajarkan antar-sampler;
- ketiga pilot selesai dengan runtime dan peak GPU memory yang feasible;
- sampler audit menunjukkan neighborhood yang berbeda.

