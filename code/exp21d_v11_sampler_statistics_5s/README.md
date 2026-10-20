# EXP21D-v11 — Statistical and Operational Analysis (EXP 5S)

EXP 5S menganalisis raw test predictions dari tiga seed awal EXP 4S (`42`, `100`, `2026`) tanpa training ulang. Dua seed tambahan tidak dibaca oleh pipeline utama ini.

## Analisis

- Paired seed delta, mean/SD/median, dan win/tie/loss.
- Hierarchical paired bootstrap: seed dan calendar-day block di-resample, dengan bobot bootstrap yang sama untuk ketiga sampler.
- McNemar exact per seed pada threshold yang dipilih dari validation set, untuk cohort all/fraud/nonfraud, disertai Holm adjustment.
- Recall@K dan Precision@K pada K `100, 200, 300, 421, 500, 1000`.
- Threshold sensitivity pada offset yang dipreregistrasikan dari validation-selected threshold; tidak ada pemilihan threshold baru dari test.
- CSV/JSON source data, bootstrap draws, figure PNG/PDF 300 DPI, input hashes, event log, manifest, dan artifact index.

## Menjalankan

```powershell
python validate_statistics.py
python main.py --config config.reference.yaml
```

Eksperimen ini berjalan lokal dan tidak memerlukan GPU atau Kaggle.
