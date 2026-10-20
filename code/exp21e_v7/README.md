# EXP21E-v7 — End-to-End Tabular MLP Baseline

Baseline non-graph yang menggunakan fitur transaksi leakage-safe dan protokol temporal yang sama dengan EXP21E-v5.

- Arsitektur: `Linear(128) -> BatchNorm -> ReLU -> Dropout(0.2)` sebanyak dua blok, lalu satu logit.
- Seed: `42, 100, 2026, 3407, 8713`.
- Checkpoint dipilih dengan validation AUPRC dan threshold dikalibrasi pada validation saja.
- Test dievaluasi satu kali per seed setelah checkpoint dan threshold dibekukan.
- Tidak menggunakan identitas user/merchant atau fitur turunan graph.

Kernel Kaggle ini terpisah dari V5 sehingga keduanya dapat berjalan secara paralel.
