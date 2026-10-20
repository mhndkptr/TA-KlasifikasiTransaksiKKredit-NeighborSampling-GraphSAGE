# EXP21E-v8 — Mechanism Audit and Statistical Analysis

V8 adalah analisis post-hoc atas artefak final V5–V7. Versi ini tidak melakukan training atau
memilih ulang checkpoint/threshold.

Analisis yang dihasilkan:

- validasi kelengkapan dan alignment artefak V5, V6, dan V7;
- paired daily-block bootstrap terhadap Uniform;
- tabel utama GNN dan MLP;
- precision–recall serta cumulative fraud-capture curve;
- degree-quartile, super-hub, FP/FN, margin, dan case-switch analysis;
- intended sampling weight versus empirical inclusion probability;
- counterpart-degree distribution dan sampled-edge overlap;
- centroid distance serta Fisher ratio dari embedding 128-d checkpoint V5.

Kernel Kaggle memasang output kernel V5, kedua window V6, dan V7 sebagai `kernel_sources`.
Checkpoint hanya dibaca untuk inference ulang; optimizer tidak dibuat dan bobot tidak diperbarui.
Semua tabel, gambar, dan laporan JSON ditulis ke `/kaggle/working/result/exp21e/v8`.

Sebelum audit dijalankan, notebook mengeksekusi `install_dependencies.py`. Bootstrap ini membaca
versi PyTorch dan CUDA aktif, memilih wheel resmi PyG yang cocok, memasang `torch_geometric`,
`pyg_lib`, `torch_scatter`, dan `torch_sparse` bila belum tersedia, lalu menjalankan smoke test
weighted `NeighborLoader`. Diagnosis instalasi disimpan sebagai `dependency_report.json`.

Jalankan lokal dengan:

```powershell
python main.py --config config.reference.yaml
```

Bangun ulang notebook Kaggle dengan:

```powershell
python create_nb.py
```
