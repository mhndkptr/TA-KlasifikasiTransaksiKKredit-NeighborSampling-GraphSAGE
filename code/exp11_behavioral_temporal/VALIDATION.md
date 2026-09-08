# Validasi EXP11 — 9 September 2026

Lingkungan lokal: Windows 11, Python 3.13.7, PyTorch 2.13.0+cu130,
PyG 2.8.0.post1, NumPy 2.5.2, pandas 2.3.3, scikit-learn 1.9.0,
**RTX 3050 Laptop 4 GiB**. Bukan benchmark komputer training 16 GB.

## Pengujian implementasi

**36 tes lulus** dalam 5,99 detik, exit code **0**. [Log pengujian](validation/tests.log).
Suite mencakup:

- Fitur historis dibanding perhitungan lambat independen; card dipisahkan per
  user, window waktu sesuai, current/equal-timestamp peers tidak masuk histori.
- Mengubah semua label atau nilai transaksi masa depan tidak mengubah fitur
  perilaku sebelumnya; transform prefix konsisten dengan prefix hasil penuh.
- Train feature tensor tetap identik ketika nilai/label held-out diubah.
- Amount hilang, cold-start, kartu tidak tersedia tetap menghasilkan fitur finite.
- Bobot subtype memakai train saja, bobot negatif 1, mean bobot fraud 1.
- Selection/kalibrasi terpisah; window jarang fraud diperluas hanya ke belakang.
- Mengubah label kalibrasi tidak mengubah state model terpilih; mengubah semua
  label test juga tidak mengubah checkpoint maupun threshold.
- Early stopping menghasilkan `complete/early_stopping`; `KeyboardInterrupt`
  menghasilkan `interrupted`, dan lock dibersihkan.
- Kesetaraan factorized/block forward dan gradient CPU/CUDA, FP16 feature store
  dengan FP32 training, sampler, root balancing, cache, checkpoint, summary,
  skip/restart, dan tiga strategi pada fixture.

Log terakhir menangkap output runner langsung dan masih memuat warning
deprecation PyG/NumPy yang tidak menggagalkan tes. Fixture integrasi diperbesar dari 180 menjadi 360 baris
agar dua window validation baru masing-masing memiliki kedua kelas.
Hash seluruh modul `exp11/*.py` beserta sampling cocok dengan source manifest
ketiga artefak pilot; hasil berasal dari kode akhir yang diserahkan.

## Rencana pilot yang ditetapkan sebelum hasil

[validate_local.py](validate_local.py) menjalankan smoke lalu tiga konfigurasi
tetap secara berurutan. Argumen dan hash sumber dicatat di
[pilot_plan.json](validation/pilot_plan.json). Perintah dari root:

```powershell
& .\code\exp6_gpu_tqdm\.venv\Scripts\python.exe .\code\exp11_behavioral_temporal\validate_local.py --rows 1000000 --epochs 30
```

Ketiga pilot memakai satu juta baris **prefix CSV yang sama**, split kronologis
700.000/150.000/150.000, fraud train/val/test **608/193/191**, uniform,
seed 42, maksimal 30 epoch. Model dasar hidden 256, fanout 25/10, FP32,
feature store CPU float16. Ketiganya selesai 30 epoch (`max_epochs`).

Kontrol mengulang hasil pilot EXP10 sebelumnya secara tepat pada AP,
checkpoint epoch, threshold dan confusion matrix: AP **0,0048895425**, epoch
26, threshold **0,8356286883**, TP/FP **2/2.114**. Ini memeriksa kesetaraan
kontrol engine EXP11 terhadap hasil EXP10 yang tersimpan.

## Hasil pilot satu juta baris

| Metrik | Kontrol EXP10 | Tambah fitur saja | Paket utama EXP11 |
|---|---:|---:|---:|
| Comparison ID | `86b93d7499aa1296` | `6fcb3f02fa489c8a` | `4807fc0324d9a265` |
| Epoch terpilih | 26 | 30 | 26 |
| Fitur transaksi | 151 | 175 | 175 |
| Roots/epoch | 12.768 | 12.768 | 31.008 |
| Batch train | 4.096 | 4.096 | 2.048 |
| AP validation seluruhnya | 0,278605 | 0,714789 | 0,729801 |
| **AP test** | **0,004890** | **0,115679** | **0,062566** |
| ROC-AUC test | 0,837657 | 0,949918 | 0,904029 |
| F1 fraud, threshold validation | 0,001734 | 0,076555 | 0,126246 |
| Recall, threshold validation | 1,05% | 4,19% | 9,95% |
| Precision, threshold validation | 0,0945% | 44,44% | 17,27% |
| TP / 191 | 2 | 8 | 19 |
| FP | 2.114 | 10 | 91 |
| AP chip test | 0,005715 | 0,152453 | 0,088721 |
| Recall chip, threshold validation | 0,68% | 4,11% | 10,27% |
| F1 fraud @0,5 | 0,008483 | 0,156788 | 0,111857 |
| Recall @0,5 | 13,09% | 21,47% | 13,09% |
| FP @0,5 | 5.678 | 291 | 231 |
| Peak VRAM allocated GiB | 0,315 | 0,356 | 0,356 |
| Durasi run, tanpa preprocessing | 12,08 s | 13,83 s | 15,86 s |

**Temuan yang didukung:** 24 fitur perilaku meningkatkan ranking pada pilot
dengan kontrol budget/loss/model yang sama. AP chip ikut naik, bukan hanya
perubahan threshold. Paket utama meningkatkan F1 pada threshold kebijakannya,
tetapi **AP dan ROC-AUC lebih rendah daripada fitur-only**. Perubahan training
tambahan belum terbukti memperbaiki seluruh metrik. Jangan menyebut paket
utama unggul atas fitur-only tanpa menyatakan metrik dan protokolnya.

Recall utama tetap hanya **19/191**; ini masih lemah. Pada @0,5, fitur-only
juga memiliki F1/recall lebih tinggi daripada paket utama. Angka itu adalah
laporan threshold tetap yang sudah ditentukan, bukan alasan memilih threshold
baru setelah membaca test. Tidak dilakukan sweep parameter berdasarkan test.

Pilot memiliki hanya **8 fraud chip pada train**, berbanding **146 pada test**.
Ini membatasi keragaman contoh yang bisa dipelajari meskipun fitur bertambah.
Kode kelompok bobot pada artefak pilot dipetakan oleh `payment_channel_names`
di preprocessing report: 0=Swipe, 1=Online, 2=Chip. Paket utama memberi bobot
positif masing-masing 1,1171 / 0,8745 / 3,4981, dengan mean seluruh fraud 1.

Sumber lengkap: [pilot_summary.json](validation/pilot_summary.json),
[log kontrol](validation/control.log), [log fitur-only](validation/features_only.log),
[log utama](validation/main.log). Artefak run berada di
`result/exp11/exp11_pilot_1000000_{control,features_only,main}_<comparison_id>_uniform_seed42/`.

## Selection dan kalibrasi pada pilot

Fraud per kuartal validation: **99, 68, 0, 26**. Karena kuartal ketiga kosong,
window selection utama diperluas ke offset **60.685–112.500** (end eksklusif),
51.815 transaksi/25 fraud. AP pada window ini memilih model. Kalibrasi tetap
offset **112.500–150.000**, 37.500 transaksi/26 fraud. Tidak overlap.

Kontrol dan fitur-only mengikuti EXP10: seluruh validation untuk selection,
paruh terakhir untuk kalibrasi F1. Paket utama memakai window terbaru tadi
dan kalibrasi F2. Perbandingan threshold validation mengandung perubahan
kebijakan, dan jumlah fraud kalibrasi sangat kecil. Ketidakpastian threshold
masih besar; distribusi chip test berbeda dari validation.

## Smoke tiga strategi

Preset smoke satu juta baris, batch 256, tiga epoch; berbeda dari budget pilot.
Comparison ID `9e050cd1f71d35d2`. [Log smoke](validation/smoke.log).

| Strategi | AP test | F1 fraud | Recall | Epoch terpilih/akhir |
|---|---:|---:|---:|---:|
| Uniform | 0,083991 | 0,123648 | 20,94% | 2/3 |
| Topology | 0,057985 | 0,101877 | 19,90% | 2/3 |
| Importance | 0,061987 | 0,108808 | 21,99% | 2/3 |

Ketiganya `complete/max_epochs`; alur tiga strategi berjalan di GPU lokal.
Awal percobaan memakai 100.000 baris ditolak oleh guard karena tidak ada fraud
di bagian kalibrasi. Guard dipertahankan; dataset smoke diperbesar.

## Batas klaim

Belum dijalankan: training EXP11 full 24.386.900 baris, lima seed, ablation
no-behavior/self-only, dan recent-refit. Peningkatan pilot bukan bukti
keunggulan full-data. Prefix CSV hanya mencakup sebagian user/kartu dan tidak
mewakili populasi. Sampling roots, normalisasi, loss, checkpoint selection dan
threshold paket utama berubah sekaligus; fitur-only merupakan kontrol yang
lebih bersih untuk mengukur tambahan sinyal perilaku.

Counter perilaku menggunakan observasi held-out sebelumnya tanpa label;
bandingkan sebagai sistem dengan histori perilaku streaming yang tersedia.
Skor test sudah digunakan untuk audit eksperimen terdahulu: laporan ini
eksploratif, bukan validasi pada test baru yang belum pernah dilihat.
