# EXP12 lokal: evaluasi hasil dan uji lanjutan berbasis konfigurasi

Preset lanjutan di direktori ini memakai `code/exp12_recent_context/run.py`,
preprocessing pandas, fitur contextual, graf train yang dibekukan, GPU lokal,
serta direktori hasil/model `exp12`. Ini **rencana uji berbasis konfigurasi**;
belum ada hasil training dari preset baru. EXP14 disisihkan untuk perubahan
implementasi atau protokol yang benar-benar baru.

## Temuan dari EXP12 full data

Run yang sedang dievaluasi adalah `comparison_id=c39b0e23c708fe40`, uniform,
seed 42, N:P 30. Dari 3.658.035 transaksi test dengan 4.454 fraud, AP test
0,216910, ROC-AUC 0,973375, precision 0,250614, recall 0,275034, F1
0,262256 pada threshold validasi 0,280506. Confusion matrix: TP 1.225,
FP 3.663, FN 3.229, TN 3.649.918. Akurasi 0,998116 kurang informatif
karena prevalensi fraud test hanya 0,001218.

Checkpoint dipilih pada epoch 5 (`recent_ap=0,190506`). Training berhenti
normal pada epoch 25 setelah 20 epoch tanpa perbaikan selection. Loss turun
dari 0,04855 ke 0,01504, tetapi AP selection epoch 25 hanya 0,12773.
AP full validation pada checkpoint 0,825605 dan maksimum teramati 0,867868
di epoch 19. Angka full validation didominasi rezim fraud online yang tidak
mirip dengan test; itu bukan kriteria pemilihan checkpoint. Selection terbaru
berisi 210 fraud (196 chip), calibration 169 fraud (153 chip), tanpa overlap.
Namun, tiga bin waktu pada selection masing-masing berisi 0, 3, dan 207 fraud;
AP per-bin dan keputusan checkpoint masih peka pada perubahan waktu.

Drift kanal sangat besar: fraud train Chip/Online/Swipe = 256/14.771/5.859,
validasi penuh = 673/3.450/294, sedangkan test = 3.907/128/419. Pada run
ini AP chip test
0,191543 dan recall chip 0,243921 (953 dari 3.907). Online AP 0,879855
tetapi hanya 128 kasus di test. Bin test terakhir berisi 185 fraud dari
731.607 transaksi, dengan AP 0,044790, precision 0,029310, recall 0,183784,
dan 1.126 FP. Penurunan AP terakhir sebagian mengikuti penurunan prevalensi
(AP lift masih 177 kali baseline), tetapi recall turun dan false alert naik.
Kualitas operasional pada periode terbaru perlu diuji dan dipantau terpisah.

Threshold F2 yang dikalibrasi pada validation menaikkan TP test dari 940
pada threshold 0,5 menjadi 1.225, tetapi FP naik dari 1.659 menjadi 3.663.
Tambahan 285 fraud terdeteksi memerlukan 2.004 alert salah tambahan. Ini
trade-off operasional; laporkan kurva precision-recall dan metrik pada batas
alert tetap sebelum memilih kebijakan threshold.

Perbandingan sampler yang adil tersedia pada cohort EXP12
`comparison_id=7bdae5fe466252c7` (fitur, N:P 50, split, seed 42 sama):

| Sampler | AP test | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Uniform | 0,207345 | 0,251586 | 0,267176 | 0,259146 |
| Topology | 0,220823 | 0,304762 | 0,251459 | 0,275557 |
| Importance | 0,224387 | 0,294223 | 0,258419 | 0,275161 |

Importance naik 8,22% AP relatif terhadap uniform pada satu seed, tetapi
belum ada ukuran variasi antar-seed. Sweep uniform N:P 10/20/25/30/50
menemukan AP test tertinggi pada 25 (0,248571), sedangkan AP selection
validasi tertinggi pada 30 (0,190506). Karena test telah dibaca, jangan
memilih 25 sebagai konfigurasi lanjutan berdasarkan angka test itu. Preset
lanjutan EXP12 memakai 30 sebagai keputusan eksploratif berdasarkan validation.

## Solusi dan urutan pelaksanaan

1. **Replikasi lokal multi-seed.** `config.n30_multiseed.yaml` membandingkan tiga sampler
   pada seed 42-46 dengan N:P 30 dan konfigurasi yang sama. Bandingkan AP
   selection, AP test eksploratif, AP/recall chip, FP, dan sebaran per seed.
   Jika selisih sampler lebih kecil daripada variasi seed, jangan klaim
   sampler berbobot lebih baik.
2. **Uji fokus chip tanpa mengganti banyak variabel.** `config.n30_channel_equal.yaml`
   menghilangkan bobot kanal; baseline memakai power 0,5/cap 4;
   `config.n30_chip_weight.yaml` memakai power 1/cap 8. Semua tetap N:P 30,
   uniform, lima seed. Pilih dari recent selection yang memuat 196 chip fraud,
   lalu nilai efeknya pada chip recall dan biaya FP. Bobot baru adalah
   hipotesis, bukan perbaikan yang telah terbukti.
3. **Ablasi graf full data.** `config.n30_self_only.yaml` menguji apakah graf
   menambah sinyal di atas 220 fitur contextual. Bukti EXP12 self-only
   0,119762 vs main 0,166574 berasal dari prefix satu juta baris dan tidak
   mewakili populasi penuh.
4. **Perbaikan validasi dan kalibrasi sesudah baseline.** Laporkan AP menurut
   bulan/kanal, recall pada FPR tetap, dan precision pada kuota alert tetap.
   Bila partisi terbaru masih menghasilkan checkpoint labil, gunakan beberapa
   origin waktu dengan validation-only untuk memilih epoch/hyperparameter.
   Threshold produksi harus dikalibrasi ulang memakai label terbaru yang
   benar-benar tersedia, dengan delay label dan budget alert yang ditentukan
   sebelumnya. Ini memerlukan perubahan kode/protokol terpisah; preset di sini
   belum mengimplementasikannya.
5. **Uji train yang lebih baru sebagai track terpisah.** Refit 80/5/15 atau
   rolling training dapat mengurangi jarak waktu train-test. Audit dulu jumlah
   fraud chip pada selection/calibration, kemudian bandingkan dengan 70/15/15
   sebagai protokol berbeda. Jangan mencampur skor kedua split dalam satu
   perbandingan sampler.

Test 2018-2020 sudah berulang kali dipakai untuk diagnosis. Semua perbandingan
lanjutan EXP12 pada test tersebut eksploratif; klaim akhir memerlukan holdout waktu
atau dataset baru yang belum dipakai memilih pendekatan.

## Menjalankan di mesin lokal

Dari root repository, dengan Python environment yang memuat PyTorch/PyG:

```powershell
$python = '.\code\exp6_gpu_tqdm\.venv\Scripts\python.exe'

# Periksa satu run terlebih dahulu; opsi ini membatasi strategi dan seed run.
& $python .\code\exp12_recent_context\run.py `
  --config .\code\exp12_recent_context\config.n30_multiseed.yaml --strategy uniform --seed 42

# Setelah run tunggal sehat, lanjutkan cohort yang sama.
& $python .\code\exp12_recent_context\run.py `
  --config .\code\exp12_recent_context\config.n30_multiseed.yaml

# Ablation: jalankan terpisah, lalu bandingkan pada seed yang sama.
& $python .\code\exp12_recent_context\run.py `
  --config .\code\exp12_recent_context\config.n30_channel_equal.yaml
& $python .\code\exp12_recent_context\run.py `
  --config .\code\exp12_recent_context\config.n30_chip_weight.yaml
& $python .\code\exp12_recent_context\run.py `
  --config .\code\exp12_recent_context\config.n30_self_only.yaml
```

Full-data EXP12 tercatat memakai sekitar 1,4-1,7 GiB peak VRAM pada RTX
5060 Ti 16 GiB. Mesin yang aktif saat dokumen ini ditulis terdeteksi RTX 3050
Laptop 4 GiB; kecukupan VRAM/RAM dan durasi preprocessing di mesin ini belum
dibuktikan. Jangan menafsirkan `--max-rows` prefix CSV sebagai evaluasi
representatif. Jalankan tes/smoke terpisah sebelum full data jika perlu.

Sumber angka: `result/exp12/exp12_recent_context_c39b0e23c708fe40_uniform_seed42/metrics.json`,
`result/exp12/exp12_recent_context_7bdae5fe466252c7_{uniform,topology,importance}_seed42/metrics.json`,
serta `result/exp12/summary.csv`.
