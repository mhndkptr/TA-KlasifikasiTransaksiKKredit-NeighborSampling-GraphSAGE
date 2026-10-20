# EXP21D-v9 — Sampler Mechanism and Degree-Stratified Outcome (EXP 3S)

EXP 3S menganalisis hasil corrected EXP 2A tanpa training ulang. Input utamanya adalah probabilitas, validation-selected threshold, dan matriks fitur hibrida tiga sampler; dataset asli dipakai untuk merekonstruksi identitas baris, timestamp, user/merchant, dan degree.

## Kontrak alignment

EXP 2A memakai `sort_values("datetime")` dengan default quicksort. Implementasi ini mereproduksi urutan tersebut dan mewajibkan dua gate sebelum analisis:

1. label test harus identik untuk ketiga sampler;
2. sepuluh fitur tabular hasil rekonstruksi harus cocok dengan blok tabular matriks fitur EXP 2A.

Run dihentikan jika salah satu gate gagal. `source_row` disimpan sebagai identitas stabil untuk menghindari join berdasarkan posisi dari eksperimen lain.

## Analisis

- Strata dibekukan: low `<=15`, medium `16–200`, high `>200`.
- Metrics dihitung terpisah untuk user degree, merchant degree, dan max degree.
- Paired fraud overlap menyimpan seluruh delapan pola tangkapan tiga sampler.
- Blok embedding 128-d dianalisis dengan centroid distance dan Fisher per-dimension summary.
- PCA 2D memakai basis terpisah per sampler dan diberi label eksploratif.

## Artefak

- Row-level CSV.GZ dan NPZ berisi transaction/source-row ID, timestamp, hash entitas, degree, strata, label, probabilitas, threshold, dan prediksi.
- Degree-stratified metrics; fraud membership CSV/NPZ; latent diagnostics; PCA source data.
- Input SHA-256 manifest, resolved config, environment, event log, code snapshot, serta artifact index dengan SHA-256.
- Empat visualisasi PNG/PDF 300 DPI beserta source data.

## Menjalankan lokal

```powershell
python validate_mechanism.py
python main.py --config config.reference.yaml
```

Folder ini belum memiliki `kernel-metadata.json` dan belum dipush ke Kaggle.
