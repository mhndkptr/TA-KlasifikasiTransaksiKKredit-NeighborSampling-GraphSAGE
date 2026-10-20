# EXP21E-v3 — Corrected Sampler and Reproducibility Gate

EXP21E-v3 adalah quality gate sebelum eksperimen temporal kausal dan multi-seed.
Versi ini tidak dimaksudkan sebagai hasil performa final paper.

## Perubahan utama

- `NeighborLoader` memakai `weight_attr="sampling_weight"` secara eksplisit.
- Candidate transaction pada hop kedua diberi skor berdasarkan degree counterpart entity.
- Bobot sampler selalu ditulis ulang sehingga tidak memakai state metode sebelumnya.
- Python, NumPy, PyTorch, CUDA, data loader, dan evaluation seed dicatat.
- Validation dan test memakai evaluation seed yang tetap.
- Threshold dipilih dari seluruh candidate threshold pada validation PR curve.
- Setiap run memiliki folder unik berdasarkan versi, window, strategy, seed, dan config hash.
- Hasil run menyimpan resolved config, environment, manifest, source snapshot, sampler audit,
  prediction IDs, history, metrics, serta checkpoint.
- `sampler_validation.py` menguji semantics bobot, distribusi sampling empiris PyG, dan
  reproducibility untuk fixed seed.

## Menjalankan validasi sampler lokal

```powershell
python sampler_validation.py --draws 10000 --seed 42 --output ../../result/exp21e/v3/sampler_validation_report.json
```

## Menjalankan satu smoke run

```powershell
python main.py --config config.reference.yaml --sampling uniform --seed 42 --smoke
```

## Menjalankan gate lengkap di Kaggle

```powershell
python create_nb.py
kaggle datasets create -p .
kaggle kernels push -p .
```

Notebook menjalankan validasi sampler terlebih dahulu. Tiga smoke runs baru dijalankan
jika validasi tersebut lulus.

## Struktur output

```text
result/exp21e/v3/
  sampler_validation_report.json
  v3_smoke_summary.csv
  exp21e_v3_<window>_<strategy>_seed<seed>_<config_hash>/
    manifest.json
    resolved_config.yaml
    environment.json
    window_statistics.json
    sampler_audit.json
    metrics.json
    training_history.json
    predictions.csv.gz
    predictions.npz
    best_checkpoint.pt
    code_snapshot/
```

## Batasan versi

V3 masih memakai graph per split untuk integration smoke test. Evaluasi causal daily batch
merupakan scope EXP21E-v4. Karena itu, metrik performa V3 harus diberi label
`smoke_only_not_final` atau `v3_gate_not_final`.

