# EXP21D-v7 — Eksperimen 2B Sampled Neighborhood Audit

Eksperimen ini mengukur perbedaan subgraf yang benar-benar dipilih oleh Uniform, Topology, dan Importance pada fan-out `[15, 10]`. Tidak ada training GNN atau XGBoost.

Kernel dijalankan sebagai notebook privat Kaggle dengan CPU karena eksperimen ini tidak melakukan training model.

## Desain audit

- Root populasi dipilih uniform tanpa pengembalian untuk statistik degree dan overlap yang tidak bias.
- Seluruh root fraud diaudit sebagai kohort terpisah agar estimasi sinyal fraud tidak bergantung pada sedikitnya fraud di sampel populasi.
- Setiap metode memakai root, repetition, dan random stream berpasangan yang sama.
- Hop pertama selalu memilih satu user dan satu merchant karena setiap transaksi memiliki tepat satu edge ke masing-masing entitas.
- Hop kedua mengambil maksimal 10 kandidat transaksi melalui user dan merchant.
- `transaction -> user` menggunakan degree merchant pasangan kandidat; `transaction -> merchant` menggunakan degree user pasangan kandidat.
- Uniform memakai bobot satu, Topology memakai inverse `log1p(degree)`, dan Importance memakai `log1p(degree)`, diikuti min-max global serta floor `0.1` seperti EXP 2A.
- Label tidak masuk fungsi sampling. Label baru digabungkan setelah edge dipilih untuk menghitung effective homophily/fraud density.
- Metrik utama homofili mengecualikan self-edge agar label root tidak menaikkan fraud density secara artifisial.

## Output

Setiap split menghasilkan:

- `selected_edges.npz`: edge-level sampling lengkap dengan method, repetition, root, kandidat, degree, bobot, dan label post-hoc;
- `local_jaccard.npz`: Jaccard per root, repetition, kanal, dan pasangan sampler;
- `root_cohorts.npz`: root audit, label, entity code, dan degree;
- CSV distribusi degree per hop/kanal (termasuk p90/p95/p99), Jaccard, dan effective homophily;
- empat visualisasi PNG/PDF 300 DPI serta source data;
- manifest, resolved config, log append-only, snapshot source, dan artifact index.

Artefak NPZ memungkinkan metrik, strata degree, atau visualisasi tambahan dihitung tanpa mengulang sampling.

## Menjalankan lokal

```powershell
python validate_audit.py
python main.py --config config.reference.yaml --seed 42
```

Audit lebih kecil dapat dibuat dengan:

```powershell
python main.py --config config.reference.yaml --population-roots 1000 --repetitions 2
```

Notebook lokal tersedia di `exp21d_v7_sampler_audit_2b.ipynb`.
