# EXP21E-v5 — Primary Paired Multi-Seed Experiment

V5 menjalankan tiga causal GraphSAGE sampler (`uniform`, `topology`, `importance`) pada lima
paired seeds: 42, 100, 2026, 3407, dan 8713. Totalnya 15 run.

Protokol daily-batch, split 300/30/8/60 hari, preprocessing, model, loss, fan-out, threshold,
dan causal audit dikunci dari V4. Test tidak digunakan untuk pemilihan checkpoint atau sampler.

```powershell
python create_nb.py
kaggle kernels push -p .
```

Output agregat meliputi `v5_run_summary.csv`, `v5_strategy_summary.csv`, dan
`v5_statistical_report.json`. Strategy selection memakai mean validation AUPRC saja.
