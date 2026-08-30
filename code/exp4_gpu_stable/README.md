# Experiment 4 - GPU-Bound dengan Stable Evaluation

EXP4 menggabungkan jalur GPU-bound dari EXP2 dengan perbaikan stabilitas dari
EXP3. Setelah preprocessing CSV di CPU, fitur node, label, adjacency CSR,
neighbor sampling, sampled subgraph, forward, backward, dan optimizer berjalan
di CUDA GPU.

Perbaikan stabilitas yang dibawa dari EXP3:

- RNG training dan evaluation dipisahkan;
- validation memakai seed dan urutan sampling yang tetap pada setiap epoch;
- probabilitas evaluation dirata-ratakan dari tiga sampling pass;
- early stopping memakai `min_delta` dan patience 20;
- threshold F1 dipilih hanya dari validation lalu dikunci untuk test;
- jumlah fraud setiap split, best epoch, learning rate, dan peak VRAM dicatat.

Sampler evaluation berbagi tensor CSR dan bobot edge GPU dengan sampler
training, tetapi mempunyai generator acak sendiri. Karena itu stabilisasi tidak
menggandakan penyimpanan graf di VRAM. EXP4 tetap memakai candidate sampling
Gumbel top-k aproksimatif milik EXP2, sehingga hasilnya dibandingkan langsung
dengan EXP2, bukan dengan sampler CPU eksak milik EXP1/EXP3.

## Menjalankan

Gunakan environment yang memiliki PyTorch CUDA dan dependensi EXP2:

```powershell
cd code/exp4_gpu_stable
python run.py --config config.yaml --strategy uniform --seed 42 --max-rows 10000
```

Untuk 200.000 baris sesuai konfigurasi:

```powershell
python run.py --config config.yaml --strategy uniform --seed 42
```

Untuk seluruh strategi dan seed:

```powershell
python run.py --config config.yaml
```

Pantau VRAM dengan `nvidia-smi -l 1`. Jika CUDA out-of-memory, turunkan
`training.batch_size`, `sampling.proposal_factor`, atau `experiment.max_rows`.
Nilai `evaluation.sampling_passes: 3` membuat evaluasi lebih lambat daripada
EXP2; ubah menjadi `1` untuk iterasi cepat, tetapi gunakan nilai yang sama saat
membandingkan strategi. Semua artefak memakai awalan `exp4_`.
