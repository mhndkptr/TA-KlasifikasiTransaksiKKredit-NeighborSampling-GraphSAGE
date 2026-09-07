# Catatan validasi EXP9

Tanggal pengujian: **7 September 2026**. Ini catatan pemeriksaan implementasi;
belum merupakan hasil eksperimen full-data atau bukti peningkatan terhadap EXP8.

Bagian awal mempertahankan catatan implementasi pertama. Hasil terbaru untuk
optimasi GPU, 21 tes, dan subset 1 juta baris terdapat pada bagian
**Validasi pembaruan GPU dan topology** di bawah.

## Pengujian otomatis

Perintah dari root repository:

```powershell
& .\code\exp6_gpu_tqdm\.venv\Scripts\python.exe -m unittest discover -s code/exp9_reference_sampling/tests -t code/exp9_reference_sampling -v
```

Hasil akhir: **12 tes lulus**, 1,768 detik waktu yang dilaporkan unittest.
Pengujian meliputi propagasi PPR dense independen, distribusi weighted sampling,
chunk hub, isolasi label, encoder/cache, kesetaraan inferensi cache dan blok,
checkpoint, metrik, serta training tiga strategi pada fixture CPU.

PyG mengeluarkan `DeprecationWarning` terkait `typing._eval_type` pada Python 3.13.
Peringatan berasal dari dependency dan tidak menyebabkan kegagalan pengujian.

## Smoke test GPU pada dataset asli

Perintah dari root repository:

```powershell
& .\code\exp6_gpu_tqdm\.venv\Scripts\python.exe .\code\exp9_reference_sampling\run.py --config .\code\exp9_reference_sampling\config.smoke.yaml --name exp9_smoke_verified --no-progress
```

Lingkungan yang dilaporkan proses:

| Komponen | Nilai |
|---|---|
| OS | Windows 11 |
| Python | 3.13.7 |
| PyTorch | 2.13.0+cu130 |
| PyTorch Geometric | 2.8.0.post1 |
| NumPy / pandas | 2.5.2 / 2.3.3 |
| scikit-learn / PyYAML | 1.9.0 / 6.0.3 |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| VRAM perangkat | 3,9995 GiB |
| CPU threads | 4 |
| Presisi model | FP32, float32 matmul precision `highest` |

Konfigurasi: prefix 100.000 baris CSV, seed 42, dua epoch, batch training 1.024,
hidden 256, dua lapisan, fanout 25/10, threshold utama 0,5, fitur disimpan di CPU.
Ketiga strategi berbagi comparison ID **`ad3e02d4d6f0a9ed`**.

| Split | Transaksi | Fraud | Awal | Akhir |
|---|---:|---:|---|---|
| Train | 70.000 | 51 | 1999-11-26 15:03 | 2015-05-17 22:12 |
| Validation | 15.000 | 32 | 2015-05-17 23:58 | 2017-10-01 10:48 |
| Test | 15.000 | 43 | 2017-10-01 12:21 | 2020-02-28 23:49 |

Graf mempunyai 2.178 entitas user/merchant. Penyusunan pertama memerlukan sekitar
1,48 detik pada smoke awal; verifikasi akhir memuat cache dalam sekitar 0,05 detik.
Angka preprocessing tersebut tidak dimasukkan ke durasi per strategi berikut.

| Strategi | Best val AP | Test AP | F1 @0,5 | Recall @0,5 | Durasi run (detik) | Peak VRAM (GiB) |
|---|---:|---:|---:|---:|---:|---:|
| Uniform | 0,008371 | 0,010952 | 0,008816 | 0,627907 | 2,665 | 0,041912 |
| Topology historical | 0,008026 | 0,010070 | 0,008793 | 0,627907 | 2,279 | 0,041912 |
| Importance | 0,007979 | 0,009957 | 0,008792 | 0,627907 | 3,678 | 0,041912 |

Ketiganya memilih checkpoint epoch pertama berdasarkan validation. Pada run ini,
topology dan importance **tidak mengungguli uniform** dalam AUPRC. Recall yang
tinggi disertai F1 sangat rendah; hasil tersebut tidak boleh dipresentasikan
sebagai keberhasilan deteksi fraud hanya karena recall terlihat tinggi.

| Strategi | Query test, ms/1.000 | Termasuk persiapan bobot/tabel/embedding, ms/1.000 |
|---|---:|---:|
| Uniform | 1,012 | 9,460 |
| Topology historical | 1,309 | 4,339 |
| Importance | 2,270 | 6,637 |

Angka ini adalah durasi seluruh test yang dinormalisasi, bukan tabel median
benchmark 1.000 query. Median lima pengulangan tersedia pada JSON hasil. Biaya
inisialisasi GPU, caching dan variasi durasi pada run pendek cukup berpengaruh;
tidak dilakukan klaim speedup antar-strategi dari satu pengukuran ini. Alokasi
VRAM tersebut adalah peak tensor PyTorch pada subset, bukan batas kebutuhan
full-data, penggunaan driver GPU, atau total RAM proses.

Artefak rinci berada di:

```text
outputs/smoke/result/exp9_smoke_verified_ad3e02d4d6f0a9ed_<strategy>_seed42/metrics.json
outputs/smoke/model/exp9_smoke_verified_ad3e02d4d6f0a9ed_<strategy>_seed42/best.pt
```

Ringkasan numerik yang dapat dilacak disertakan di
[`validation/smoke_summary.json`](validation/smoke_summary.json). File output
besar/checkpoint/cache berada di `outputs/` yang diabaikan Git. Waktu dan
probabilitas dapat sedikit berbeda saat dijalankan ulang karena reduksi CUDA dan
kondisi perangkat, meskipun seed tetap.

## Validasi pembaruan GPU dan topology

### Pemeriksaan otomatis terbaru

Perintah unittest sama seperti di atas. Hasil: **21 tes lulus dalam 3,648 detik**,
termasuk tes CUDA yang benar-benar dieksekusi, bukan dilewati. Tambahan pemeriksaan:

- Output, semua gradien parameter, statistik BatchNorm, dan dropout sama dalam
  toleransi numerik antara backend blocks dan factorized pada CPU/CUDA.
- Training CUDA juga bekerja saat fitur disimpan di CPU; history kosong tetap
  memiliki mean tetangga nol.
- Cache sampler CUDA mempertahankan tabel pada seed yang sama dan tidak mengubah
  bobot sumber. Pemindahan permutation sekali per epoch mempertahankan urutan CPU.
- Topology dengan label int8, 17.070.830 transaksi train dan total fraud 20.886
  tidak overflow. Bug awal direproduksi saat persiapan topology pada subset
  1 juta baris (608 fraud train), lalu diperbaiki dengan promosi tipe per chunk.
- Run lengkap dilewati; run terputus mendapat attempt baru dan checkpoint lama
  tetap utuh; lock melindungi overwrite; exception dicatat; attempt seed yang sama
  hanya dihitung satu kali pada ringkasan.

### Benchmark training dengan input yang sama

Perintah dari root repository:

```powershell
& .\code\exp6_gpu_tqdm\.venv\Scripts\python.exe code/exp9_reference_sampling/benchmark.py --config code/exp9_reference_sampling/config.gpu16gb.yaml --max-rows 1000000 --steps 200 --warmup 20 --repeats 3 --output code/exp9_reference_sampling/outputs/benchmark/gpu_optimization_1m.json
```

Lingkungan tetap RTX 3050 Laptop 4 GiB, Python 3.13.7, PyTorch 2.13.0+cu130,
PyG 2.8.0.post1, NumPy 2.5.2. Thread CPU 8. Dataset asli dibatasi ke 1 juta baris,
dengan 700.000 transaksi train dan 10.528 entitas. Topology historical,
batch 1.024, hidden 256, fanout 25/10, dropout 0,2, dan FP32 sama pada setiap varian.

| Varian | Waktu 200 batch, tiga repeat (detik) | Median (detik) | Transaksi/detik | Rasio median |
|---|---|---:|---:|---:|
| Blocks + Adam | 3,546 / 6,540 / 5,857 | 5,857 | 34.967 | 1,00x |
| Factorized + Adam | 2,716 / 4,509 / 4,407 | 4,407 | 46.467 | 1,33x |
| Factorized + fused Adam | 3,140 / 3,396 / 2,871 | 3,140 | 65.218 | 1,87x |

Root, tabel sampling dan state awal model sama; urutan varian dirotasi. Pengukuran
mencakup forward, loss, backward, clipping, finite check dan optimizer. Tidak
mencakup preprocessing, sampling, pembuatan mean, permutation atau validation.
Mean fitur mentah membutuhkan 0,088 detik di luar timer training. Semua varian
menggunakan root yang sudah berada pada GPU, sehingga ini tidak mengukur biaya
transfer root per batch pada versi lama. Backend blocks yang dipertahankan menjadi
baseline komputasi, bukan checkout penuh versi lama.

Variasi waktunya terlihat besar; rasio median bukan jaminan speedup setiap run.
Benchmark awal 100.000 baris juga dijalankan, kemudian pengukuran diperpanjang ke
1 juta baris untuk mengurangi ketergantungan pada interval yang sangat pendek.
Tidak ada benchmark pada RTX 5060 Ti dan tidak ada klaim utilisasi GPU 100%.

### Smoke test GPU 1 juta baris

```powershell
& .\code\exp6_gpu_tqdm\.venv\Scripts\python.exe code/exp9_reference_sampling/run.py --config code/exp9_reference_sampling/config.smoke.gpu.yaml --max-rows 1000000 --epochs 1 --name exp9_gpu_1m_verified --no-progress
```

Ketiga strategi selesai, satu epoch dan seed 42. Split: train 700.000/608 fraud,
validation 150.000/193 fraud, test 150.000/191 fraud. Backend factorized, fitur dan
indeks CUDA, fused Adam aktif. `comparison_id` ketiganya adalah
`d86bf4de62df21f9`. Status, keberadaan checkpoint, dan kecocokan source hash dengan
kode yang diserahkan telah diperiksa.
Perintah yang sama kemudian dijalankan ulang: ketiganya mencetak
`Run selesai ditemukan; dilewati`, tanpa menjalankan epoch baru.

| Strategi | Val AUPRC | Test AUPRC | Test F1 @0,5 | Durasi run (detik) |
|---|---:|---:|---:|---:|
| Uniform | 0,019148 | 0,004002 | 0,009155 | 6,886 |
| Topology | 0,041491 | 0,003664 | 0,008440 | 6,500 |
| Importance | 0,018397 | 0,004249 | 0,009480 | 10,076 |

Angka kualitas tersebut berasal dari **satu epoch untuk pemeriksaan pipeline**.
AUPRC/F1 rendah; tidak digunakan untuk menyatakan optimasi meningkatkan kualitas
deteksi atau memilih strategi. Durasi antarsmoke juga bukan perbandingan efisiensi
sampling yang terkontrol. Preprocessing sebelum `run_one` tidak masuk durasi run.

Artefak lengkap lokal berada di:

```text
outputs/benchmark/gpu_optimization_1m.json
outputs/smoke/result/exp9_gpu_1m_verified_d86bf4de62df21f9_<strategy>_seed42/
outputs/smoke/model/exp9_gpu_1m_verified_d86bf4de62df21f9_<strategy>_seed42/best.pt
```

Salinan benchmark beserta metadata, hash source, metrik, history, dan timing
smoke tersedia pada [validation/performance_summary.json](validation/performance_summary.json)
agar angka yang diringkas dapat diaudit tanpa memasukkan cache/checkpoint ke Git.
Kode preprocessing tidak berubah sehingga cache graf lama masih dapat digunakan;
source optimasi membuat identitas run training baru berbeda dari validasi awal.

## Pekerjaan evaluasi penelitian yang masih diperlukan

- Full-data, tiga strategi x lima seed, dengan aturan yang sama.
- Analisis mean/std, pasangan seed terhadap uniform, false positive dan long-tail.
- Ablation literal/historical, koreksi prior, dan batch size secara terpisah.
- Perbandingan historis dengan EXP8 yang menyatakan perbedaan protokol graf dan
  ruang lingkup latensi.

Smoke test ini memverifikasi bahwa pipeline berjalan pada data asli. Ia tidak
mengukur manfaat full-data, tidak menjamin GPU 4 GiB cukup untuk setiap ukuran
data, dan tidak menyediakan bukti statistik bahwa EXP9 meningkatkan EXP8.
