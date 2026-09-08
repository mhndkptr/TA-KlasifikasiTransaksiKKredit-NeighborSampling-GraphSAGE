# Validasi EXP10 — 8 September 2026

Pengujian lokal pada Windows 11, Python 3.13.7, PyTorch 2.13.0+cu130,
PyG 2.8.0.post1, NumPy 2.5.2, pandas 2.3.3, scikit-learn 1.9.0,
**RTX 3050 Laptop 4 GiB**. Ini bukan pengukuran RTX 5060 Ti 16 GiB.

## Pemeriksaan implementasi

**28 tes lulus**, mencakup:

- parsing kronologis, dictionary/statistik train-only, missing error dan OOV;
- cache invalidation untuk split serta storage fitur;
- distribusi weighted sampling tanpa replacement, PPR dibanding dense reference,
  batas chunk hub dan histori yang tidak memakai held-out transactions;
- kesetaraan block/factorized forward, gradient, dropout dan BatchNorm di CPU
  serta CUDA, termasuk CUDA dengan feature store CPU float16;
- metrik macro-F1 dibanding scikit-learn, GMean dan AP-lift;
- semua fraud muncul sekali pada balanced-root epoch, unique roots, reproducibility,
  variasi negatif antar-epoch, dan penolakan double balancing;
- pemilihan temporal, self-only LayerNorm, serta konsistensi threshold antara
  checkpoint, metrics dan CSV;
- mengubah **semua label test** tetap menghasilkan model, checkpoint epoch,
  dan threshold yang sama pada fixture CPU;
- siklus tiga strategi, skip/restart run, lock dan de-duplication attempt.

Log tersedia di [tests.log](validation/tests.log). PyG memberi DeprecationWarning
`typing._eval_type` pada Python 3.13. Warning itu tidak menggagalkan suite.

## Audit dataset penuh

[exp9_audit.json](validation/exp9_audit.json) memuat agregat 24.386.900 transaksi,
label tahunan, serta distribusi fitur per kelas/split. Tidak dilakukan full-data
training EXP10. Lihat [ANALISIS_EXP9.md](ANALISIS_EXP9.md) untuk hasil audit.

## Smoke tiga strategi

Perintah: `config.smoke.yaml --no-progress`, 100.000 baris **prefix CSV**,
seed 42, tiga epoch, batch 256, hidden 256, GPU, fitur CPU float16.
Split train/val/test 70.000/15.000/15.000, fraud 51/32/43.
Comparison ID smoke `f1fe9f763923ac81`.

| Strategi | Val AP checkpoint | Test AP | Test ROC-AUC | F1/Recall threshold validation |
|---|---:|---:|---:|---:|
| Uniform | 0,102303 | 0,003626 | 0,511235 | 0 / 0 |
| Topology | 0,092476 | 0,003679 | 0,513105 | 0 / 0 |
| Importance | 0,100019 | 0,003537 | 0,508067 | 0 / 0 |

Alur, checkpoint dan artefak berhasil dibuat; deteksi belum baik. Smoke tidak
boleh disebut bukti peningkatan. Smoke ini dijalankan sebelum penambahan laporan
per kanal pembayaran; source manifest-nya berbeda dari pilot berikutnya.

## Pilot satu juta transaksi dengan kontrol

Kedua run menggunakan satu juta baris awal CSV yang sama, urutan waktu sama,
split 700.000/150.000/150.000, fraud 608/193/191, uniform neighbor sampling,
seed 42, hidden 256, fanout 25/10, batch training 4.096, maksimal 30 epoch,
dan CPU feature store. Epoch terpilih ditentukan oleh validation.

```powershell
& .\code\exp6_gpu_tqdm\.venv\Scripts\python.exe .\code\exp10_temporal_robust\run.py --config .\code\exp10_temporal_robust\config.gpu16gb.yaml --strategy uniform --seed 42 --max-rows 1000000 --epochs 30 --name exp10_pilot_1m --no-progress
& .\code\exp6_gpu_tqdm\.venv\Scripts\python.exe .\code\exp10_temporal_robust\run.py --config .\code\exp10_temporal_robust\config.legacy_control.yaml --strategy uniform --seed 42 --max-rows 1000000 --epochs 30 --name exp10_legacy_pilot_1m --no-progress
```

| Ukuran | Kontrol setelan EXP9 | EXP10 utama |
|---|---:|---:|
| Comparison ID | 536930f903a1662c | 2bf1b7b39ee22723 |
| Epoch dipilih | 30 | 26 |
| Val AP checkpoint | 0,487631 | 0,278605 |
| Test AP | 0,004089 | 0,004890 |
| Test ROC-AUC | 0,646780 | 0,837657 |
| Test F1 fraud @0,5 | 0,011322 | 0,008483 |
| Test Recall @0,5 | 0,392670 | 0,130890 |
| Test FP @0,5 | 12.983 | 5.678 |
| Test TP @0,5 | 75 | 25 |
| Test F1, validation-selected threshold | 0 | 0,001734 |
| Test Recall, validation-selected threshold | 0 | 0,010471 |
| Roots/epoch | 700.000 | 12.768 |
| Fitur transaksi | 9 | 151 |
| Ukuran tensor fitur | 36 MB | 302 MB |
| Durasi run, di luar preprocessing | 51,19 detik | 17,35 detik |
| Peak VRAM allocated | 0,093 GiB | 0,313 GiB |

Kontrol mengkalibrasi pada semua validation, EXP10 pada paruh terakhir sesuai
konfigurasi; baris threshold validation membandingkan dua **kebijakan berbeda**.
Baris @0,5 menyamakan angka threshold, tetapi skala skor juga berubah karena loss
dan root balancing. AP/ROC-AUC tidak bergantung threshold.

Test AP naik sekitar **19,6% relatif** dan ROC-AUC meningkat, tetapi **F1 dan
recall @0,5 turun**. Threshold validation EXP10 0,835629 hanya menemukan 2 dari
191 fraud. Pilot belum memenuhi tujuan deteksi yang baik dan bukan alasan untuk
mengklaim berhasil mengalahkan EXP9 full-data atau referensi paper.

Pilot EXP10 memiliki 26 fraud pada paruh validation untuk kalibrasi. Test tidak
memiliki fraud online, sedangkan 146/191 fraud adalah chip. Per-kanal, test chip
ROC-AUC 0,840771 dan AP 0,005715, tetapi recall 0,006849 pada threshold terpilih.
Perubahan pola masih menjadi masalah walaupun ranking global terlihat tinggi.

Hasil mentah berada di `result/exp10/exp10_pilot_1m_2bf1b7b39ee22723_uniform_seed42/`
dan `result/exp10/exp10_legacy_pilot_1m_536930f903a1662c_uniform_seed42/`.
Konfigurasi, source manifest, preprocessing dan metrik lengkap ikut disimpan.
`result/exp10/summary.csv` saat ini berisi **pilot satu juta baris**, bukan full data.

Kedua run memakai jumlah epoch yang sama tetapi budget update/roots berbeda.
Durasi tersebut bukan perbandingan efisiensi pada akurasi yang disamakan.
Prefix CSV juga bukan sampel populasi yang representatif. Ablation fitur-only,
balanced-only, LayerNorm, self-only, recent-refit dan lima seed full data
**belum dijalankan**. Klaim performa harus menunggu pengujian itu.
