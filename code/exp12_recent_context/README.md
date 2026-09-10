# EXP12 — Recent behavioral context and drift-aligned validation

EXP12 menindaklanjuti full-data EXP11. AP test sudah naik dari **0,0285–0,0353**
di EXP10 menjadi **0,1539–0,1878**, tetapi checkpoint EXP11 ternyata dipilih
dari window yang hanya berisi 25 fraud: 20 online, 3 swipe, dan 2 chip. Pola
ini tidak mewakili test yang memiliki 3.907 fraud chip. Bukti dan perbandingan
repository ada di [ANALISIS_EXP11.md](ANALISIS_EXP11.md).

## Perubahan utama

- Mempertahankan 24 fitur EXP11 dan menambah 41 fitur strictly-past tanpa label:
  nominal relatif terhadap histori 7 hari, kebiasaan jam, intensitas 24 jam
  terhadap 7 hari, konteks kartu–kanal, kartu–MCC, user–merchant,
  merchant–kanal, serta novelty berbasis kartu.
- Nilai `Card` yang kosong kini masuk histori kartu-unknown per user. Satu nilai
  kosong tidak lagi membuat semua histori kartu disamakan dengan histori user.
- Seperempat validation terbaru menjadi populasi temporal yang relevan. Blok
  waktu tujuh hari bergantian dipakai untuk checkpoint dan threshold. Pembagian
  bergantung pada timestamp, bukan label, dan kedua subset tidak overlap.
- Jika partisi terbaru tidak memiliki fraud minimum, run gagal dengan jelas;
  pipeline tidak memperluas window ke rezim lama atau test.
- Penulisan artefak atomik memiliki retry terbatas untuk handle file sementara
  dari scanner/indexer Windows.

Audit full CSV menghasilkan 210 fraud untuk selection (196 chip) dan 169 fraud
untuk calibration (153 chip). EXP11 sebelumnya memakai 25 fraud untuk selection,
dengan hanya 2 fraud chip.

Arsitektur utama tetap dua layer mean GraphSAGE, hidden 256, LayerNorm,
dropout 0,2, fanout 25/10, tiga strategi sampling, dan split temporal 70/15/15.
Root balancing serta bobot subtype tetap dipelajari dari train saja.

## Menjalankan

Dari root repository:

```powershell
# Unit/integration tests CPU + CUDA
& .\code\exp6_gpu_tqdm\.venv\Scripts\python.exe -m unittest discover -s code/exp12_recent_context/tests -t code/exp12_recent_context -v

# Audit hasil EXP10/EXP11 dan window validation terhadap CSV penuh
& .\code\exp6_gpu_tqdm\.venv\Scripts\python.exe .\code\exp12_recent_context\audit_exp11.py --data .\dataset\credit_card_transactions-ibm_v2.csv

# Smoke tiga strategi, satu juta baris, tiga epoch
& .\code\exp6_gpu_tqdm\.venv\Scripts\python.exe .\code\exp12_recent_context\run.py --config .\code\exp12_recent_context\config.smoke.yaml --no-progress

# Full-data satu strategi terlebih dahulu
& .\code\exp6_gpu_tqdm\.venv\Scripts\python.exe .\code\exp12_recent_context\run.py --config .\code\exp12_recent_context\config.gpu16gb.yaml --strategy uniform --seed 42

# Tiga strategi × lima seed
& .\code\exp6_gpu_tqdm\.venv\Scripts\python.exe .\code\exp12_recent_context\run.py --config .\code\exp12_recent_context\config.gpu16gb.yaml
```

Preset full-data mempertahankan minimum 25 fraud pada masing-masing partisi
validation. `config.smoke.yaml` dan `validate_local.py` menurunkan guard menjadi
1 karena prefix satu juta baris hanya memvalidasi alur, bukan mengestimasi hasil
populasi.

## Kontrol dan ablation

| Preset | Perubahan yang diukur |
|---|---|
| `config.control_gpu.yaml` | Reproduksi pengaturan utama EXP11 pada engine EXP12 |
| `config.features_only.yaml` | Fitur context baru + partition validation EXP11 |
| `config.logic_only.yaml` | Fitur EXP11 + partition recent-interleaved EXP12 |
| `config.gpu16gb.yaml` | Fitur dan logic EXP12 |
| `config.self_only.yaml` | Paket utama tanpa pesan tetangga untuk mengukur kontribusi graf |
| `config.recent_refit.yaml` | Protokol terpisah 80/5/15; jangan dicampur dengan hasil utama |

Gunakan `validate_local.py` untuk menjalankan smoke dan kelima pilot dengan
urutan yang sudah ditetapkan. Semua hasil harus dilaporkan, termasuk ablation
yang lebih buruk. AP test pada test yang sudah berulang kali dianalisis bersifat
eksploratif; klaim akhir membutuhkan periode atau dataset baru.

Pilot satu juta baris uniform seed 42 menghasilkan AP **0,062566** pada kontrol
EXP11, **0,166574** pada features-only dan main, serta **0,119762** pada
self-only. Logic-only turun ke 0,046643 karena partisi terbaru pada prefix hanya
memiliki 11/15 fraud. Rincian F1, recall, precision, dan batas interpretasi ada
di [VALIDATION.md](VALIDATION.md).

## Memori dan artefak

Encoder context memiliki 65 fitur perilaku, dibanding 24 pada EXP11. Tambahan
41 kolom float16 membutuhkan sekitar **1,86 GiB** untuk 24.386.900 transaksi,
di luar fitur dasar, tensor graf, DataFrame, dan indeks temporer preprocessing.
Fitur disimpan pada CPU dan dikonversi FP32 per batch.

Output utama: `result/exp12/`; cache/checkpoint: `model/exp12/`. Audit dan log
validasi tersimpan di `code/exp12_recent_context/validation/`. Checkpoint memuat
skema fitur, kebijakan validation, threshold, dan identitas source/data.
