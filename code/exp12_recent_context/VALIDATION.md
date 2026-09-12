# Validasi EXP12

Status 10 September 2026.

## Pemeriksaan yang selesai

- Audit artefak full-data EXP10/EXP11 menemukan enam run selesai, masing-masing
  tiga strategi pada seed 42.
- Audit streaming CSV cocok tepat dengan jumlah validation 3.658.035 dan 4.417
  fraud dari artefak EXP11.
- Audit membuktikan selection EXP11 hanya memuat 2 fraud chip; partisi selection
  dan calibration EXP12 memuat 196 dan 153 fraud chip tanpa overlap.
- 48 unit/integration tests lulus pada Python 3.13.7, PyTorch 2.13.0+cu130,
  PyG 2.8.0.post1, dan NVIDIA RTX 3050 Laptop 4 GB.
- Test mencakup strictly-past/equal-timestamp, isolasi future/label,
  partial-missing card, formula nominal conditional, validation partition,
  isolasi calibration/test dari checkpoint, semua preset, sampler, metrik,
  serta kesetaraan output/gradient backend factorized dan block di CPU/CUDA.
- Smoke satu juta baris, tiga strategi, tiga epoch selesai setelah guard minimum
  fraud subset ditetapkan ke 1.

Log awal smoke yang gagal dipertahankan sebagai bukti fail-closed: konfigurasi
utama minimum 25 fraud menolak prefix yang terlalu kecil. Preset smoke kemudian
secara eksplisit memakai minimum 1; konfigurasi full-data tetap 25.

## Pilot

`validate_local.py` menjalankan kontrol EXP11, feature-only, logic-only, paket
utama EXP12, dan self-only pada prefix satu juta baris, uniform seed 42, maksimum
30 epoch. Prefix ini tersusun per user/kartu dan bukan sampel populasi yang
representatif. Hasil di bawah harus dibaca sebagai pemeriksaan arah dan wiring,
bukan estimasi full-data.

| Preset | AP test | F1 fraud | Recall | Precision | Best epoch |
|---|---:|---:|---:|---:|---:|
| EXP11 control | 0,062566 | 0,126246 | 9,95% | 17,27% | 26 |
| Features only | 0,166574 | 0,205298 | 16,23% | 27,93% | 19 |
| Logic only | 0,046643 | 0,075377 | 7,85% | 7,25% | 30 |
| EXP12 main | 0,166574 | 0,205674 | 15,18% | 31,87% | 19 |
| Self-only | 0,119762 | 0,164134 | 14,14% | 19,57% | 16 |

Fitur baru menaikkan AP pilot dari 0,062566 menjadi 0,166574 (2,66×).
Main dan features-only memilih epoch yang sama sehingga ranking AP sama;
partition threshold EXP12 menukar sedikit recall untuk precision dan F1 yang
sedikit lebih tinggi. Self-only lebih rendah 0,046812 AP dari main, sehingga
konteks graf masih memberi kontribusi pada pilot ini.

Logic-only memburuk. Pada prefix satu juta, partisi interleaved terbaru hanya
memiliki 11 fraud untuk selection dan 15 untuk calibration; guard sengaja
diturunkan menjadi 1 agar wiring dapat diuji. Hasil itu mendukung kebijakan
fail-closed minimum 25 pada full-data dan tidak boleh dipakai untuk menyimpulkan
bahwa logic validation buruk pada populasi penuh, yang memiliki 210/169 fraud.

Smoke tiga epoch selesai untuk uniform/topology/importance dengan AP test
masing-masing 0,151156 / 0,142099 / 0,135347. Angka smoke tidak dipakai memilih
strategi karena budget training dan subset terlalu kecil.

## Validasi backend Kaggle out-of-core (12 September 2026)

- 48 unit/integration tests lulus, termasuk external chronological sort,
  split tie-safe, memmap, CSR train-only, ekspor `HeteroData`, dan invariansi
  bobot topology terhadap perubahan seluruh label validation/test.
- Smoke preprocessing pada 1.000.000 baris CSV asli berhasil selesai. Graf
  berisi 700.001 train, 150.000 validation, dan 149.999 test; seluruh boundary
  timestamp tidak overlap.
- Cache memmap dimuat ulang dalam 0,06 detik. Satu epoch uniform pada RTX 3050
  4 GB memproses 700.001 training roots dalam 6,1 detik (114.387 transaksi/detik)
  dengan sampler cache 10,7 MiB.
- `pos_weight` dihitung dari train, bukan dikunci ke 833. Pada prefix ini nilainya
  1.150,317 karena fraud rate train 0,0869%; full data akan mengikuti rasio train
  aktual.
- Smoke satu epoch hanya memvalidasi wiring dan tidak digunakan sebagai hasil
  ilmiah atau pembanding strategi.

## Yang belum divalidasi

- Full-data EXP12 belum dijalankan.
- Lima seed untuk tiga strategi belum dijalankan.
- Kebutuhan peak RAM preprocessing full-data belum diukur pada mesin ini.
- Test 2018–2020 telah dipakai berulang untuk diagnosis; evaluasi final tetap
  membutuhkan periode atau dataset baru.
