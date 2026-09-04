# Experiment 8 - Adaptive Batched Importance Sampling

EXP8 mempertahankan data, model, training, evaluasi, local-PPR eksak, dan
weighted Gumbel-top-k dari EXP7. Perubahannya hanya pada cara komputasi strategy
`importance`, untuk meningkatkan paralelisme GPU tanpa kembali mengalami OOM.

## Mengapa EXP7 importance lambat

Pada smoke test 100.000 baris, EXP7 importance membutuhkan 568,89 detik,
dibandingkan 8,10 detik untuk uniform. Perbaikan OOM EXP7 memproses local-PPR
satu root per giliran dan sering mengambil scalar CUDA dengan `.item()`. GPU
menjalankan banyak kernel kecil lalu menunggu loop Python, sehingga utilisasi
rata-rata hanya sekitar 30%.

## Optimasi EXP8

- Root ringan dikemas menjadi adaptive batch sampai batas
  `importance_edge_chunk_size` dan `importance_max_roots_per_batch`.
- Hub yang sendirian melampaui edge budget tetap diproses dalam slice, sehingga
  temporary tensor tidak tumbuh tanpa batas.
- Satu transfer degree vector ke CPU dipakai untuk merencanakan banyak chunk;
  sinkronisasi `.item()` per root dihilangkan.
- Langkah PPR terakhir mengambil mass melalui reverse adjacency. Ini eksak
  karena graph dibangun bidirectional dan menghindari ekspansi last-hop terbesar.
- Bobot `gamma * centrality + (1-gamma) * PPR` disimpan per CSR edge di GPU.
  Bobot tersebut statis; epoch dan evaluation berikutnya hanya membuat Gumbel
  noise baru. Cache juga digunakan bersama oleh seed berikutnya bila seluruh
  eksperimen dijalankan dalam satu proses.

Optimasi tersebut tidak mengubah fanout, nilai bobot importance, atau distribusi
sampling tanpa penggantian. `batch_size`, arsitektur, loss, scheduler, threshold,
dan seed policy tetap sama untuk ketiga strategy.

Cache bobot importance menambah kira-kira empat byte per directed edge ditambah
satu byte per node. Pada graph dengan sekitar 97 juta directed edge, tambahan
VRAM sekitar 0,39 GB. Nonaktifkan `cache_importance_weights` hanya bila ruang
VRAM tersebut tidak tersedia; hasil matematis tetap sama, tetapi PPR akan
dihitung ulang.

## Hasil smoke test CUDA

Pengujian 100.000 baris, importance seed 42, batch size 4096, dan 16 epoch
dilakukan pada RTX 3050 Laptop 4 GB:

| Pengukuran | EXP7 | EXP8 | Perubahan |
| --- | ---: | ---: | ---: |
| Durasi total | 568,89 detik | 8,78 detik | 64,8x lebih cepat |
| Inference | 5,744 detik | 0,058 detik | 99,6x lebih cepat |
| Peak VRAM | 0,0916 GB | 0,0930 GB | praktis sama |

Pada run tersebut terdapat 313 cache miss saat root pertama kali dihitung dan
86.457 cache hit. Nilai bobot telah diuji sama dengan implementasi exact EXP5.
Karena pengemasan root mengubah urutan konsumsi random number, satu seed tidak
harus menghasilkan realisasi neighbor yang identik dengan EXP7, tetapi
distribusi weighted sampling-nya tetap sama.

## Cache preprocessing

`experiment.cache_graph: true` menyimpan feature matrix, split, label, dan edge
graph sebagai tensor di `model/cache/`. EXP8 dapat memakai cache preprocessing
EXP7 bila fingerprint-nya sama. Fingerprint mencakup dataset, ukuran/waktu
modifikasi file, `max_rows`, temporal split, dan source preprocessing.

Bangun atau validasi cache tanpa training:

```powershell
..\exp6_gpu_tqdm\.venv\Scripts\python.exe run.py --config config.yaml --preprocess-only
```

Paksa cache dibangun ulang:

```powershell
..\exp6_gpu_tqdm\.venv\Scripts\python.exe run.py --config config.yaml --preprocess-only --rebuild-cache
```

## Menjalankan

Dari folder `code/exp8_adaptive_importance`:

```powershell
..\exp6_gpu_tqdm\.venv\Scripts\python.exe run.py --config config.yaml --strategy importance --seed 42
```

Smoke test 100.000 baris:

```powershell
..\exp6_gpu_tqdm\.venv\Scripts\python.exe run.py --config config.yaml --strategy importance --seed 42 --max-rows 100000
```

Hasil penelitian harus memakai full-data dan semua strategy/seed:

```powershell
..\exp6_gpu_tqdm\.venv\Scripts\python.exe run.py --config config.yaml
```

Output ditulis sebagai `result/exp8_<strategy>_seed<seed>.json`, checkpoint
sebagai `model/exp8_<strategy>_seed<seed>.pt`, dan ringkasan sebagai
`result/exp8_summary.csv`.

Jika full-data masih OOM, turunkan `importance_edge_chunk_size` secara bertahap,
misalnya dari 250.000 menjadi 125.000. Jangan mengubah `batch_size` hanya untuk
importance karena itu mengubah jumlah optimizer step dan perilaku BatchNorm.
