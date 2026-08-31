# Experiment 5 - Corrected GPU-Bound GraphSAGE

EXP5 mempertahankan keputusan utama EXP1, tetapi memindahkan fitur, label,
CSR graph, neighbor sampling, pembentukan subgraf, forward, backward, dan
optimizer ke CUDA.

Perbaikan terhadap EXP2/EXP4:

- threshold kembali fixed `0.5` seperti EXP1; threshold validation sekitar
  0.97-0.99 pada EXP3/EXP4 menghasilkan `TP=0` pada temporal test;
- sampling GPU menggunakan seluruh adjacency dan benar-benar tanpa
  penggantian, bukan candidate proposal yang dapat berulang;
- duplicate edges antar-hop dipertahankan seperti sampled multigraph EXP1;
- batch size 1024 dan FP32 dipakai agar training dynamics sebanding dengan
  EXP1;
- RNG evaluasi dipisahkan dari RNG training agar checkpoint reproducible.

## Menjalankan

```powershell
cd code/exp5_gpu_corrected
..\exp1_starter\.venv\Scripts\python.exe run.py --config config.yaml --strategy uniform --seed 42
```

Hasil ditulis sebagai `result/exp5_<strategy>_seed<seed>.json`, checkpoint
sebagai `model/exp5_<strategy>_seed<seed>.pt`, dan ringkasan sebagai
`result/exp5_summary.csv`.

Pantau dengan `nvidia-smi -l 1`. Naikkan batch size hanya setelah baseline
tervalidasi karena batch size mengubah jumlah optimizer step dan BatchNorm.

`max_rows: 200000` sama dengan artefak EXP1-EXP4 yang tersedia. Nilai `null`
memakai seluruh dataset dan membutuhkan RAM/VRAM jauh lebih besar. Strategy
`importance` menghitung local PPR eksak di GPU; turunkan
`ppr_root_chunk_size` jika terjadi CUDA out-of-memory.
