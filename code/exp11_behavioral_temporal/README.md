# EXP11 — Fitur perilaku dan validation terbaru

Menindaklanjuti hasil full-data EXP10: AP test 0,0285–0,0353 dan recall
5,14–6,53%. Ketiga run **selesai melalui early stopping**, bukan interrupt.
Penjelasan beserta angka: [ANALISIS_EXP10.md](ANALISIS_EXP10.md).

## Perubahan

- 24 fitur riwayat user/kartu: intensitas transaksi, jeda, penyimpangan nominal,
  kebaruan/kelangkaan merchant, MCC, kanal, dan lokasi.
- Riwayat hanya membaca observasi dengan waktu **lebih awal**, tanpa label.
  Transaksi simultan tidak saling membaca. Card dikunci bersama User.
- Robust amount, one-hot, missing/OOV, dan fit statistik train-only dipertahankan.
- LayerNorm; 50 normal/fraud per epoch; bobot subtype fraud dari training saja.
- Checkpoint menggunakan AP validation terbaru. Window default val[50%:75%]
  diperluas mundur bila target 25 fraud belum tercapai; tidak melewati batas
  75%. Kalibrasi F2 khusus val[75%:100%]. Test tidak menentukan checkpoint/threshold.
- Status menyebut `early_stopping` atau `max_epochs`; interrupt sebenarnya
  tetap `interrupted`. Minimum 20 epoch dan patience 20 pada preset utama.

GraphSAGE tetap dua layer mean, hidden 256, dropout 0,2, fanout 25/10, FP32,
dan tiga strategi uniform/topology/importance. Split utama tetap **70/15/15**.
Graf tetap membeku pada train; counter perilaku menerima observasi sebelumnya
pada validation/test. Ini simulasi streaming fitur tanpa label, bukan pembaruan
graf dengan label test. Graf train belum strictly causal per transaksi.

## Menjalankan dari root repository

```powershell
# Smoke alur: prefix satu juta baris, tiga strategi, seed 42, tiga epoch.
& .\code\exp6_gpu_tqdm\.venv\Scripts\python.exe .\code\exp11_behavioral_temporal\run.py --config .\code\exp11_behavioral_temporal\config.smoke.yaml --no-progress

# Pilot full-data satu strategi pada komputer training.
& .\code\exp6_gpu_tqdm\.venv\Scripts\python.exe .\code\exp11_behavioral_temporal\run.py --config .\code\exp11_behavioral_temporal\config.gpu16gb.yaml --strategy uniform --seed 42

# Tiga strategi x lima seed.
& .\code\exp6_gpu_tqdm\.venv\Scripts\python.exe .\code\exp11_behavioral_temporal\run.py --config .\code\exp11_behavioral_temporal\config.gpu16gb.yaml
```

Jika environment sudah aktif dan berada di folder ini:
`python run.py --config config.gpu16gb.yaml --strategy uniform --seed 42`.
Path dataset/output diselesaikan relatif terhadap YAML, bukan cwd.
Dependensi ada di [requirements.txt](requirements.txt), sama dengan EXP10.

Smoke satu juta baris dipakai karena prefix 100.000 baris tidak memiliki fraud
di seperempat terakhir validation. Tidak ada fallback diam-diam ke test.
Prefix CSV berurutan per user/kartu dan bukan sampel populasi yang representatif.

## Kontrol dan ablation

| Preset | Tujuan |
|---|---|
| `config.exp10_control.yaml` | Mengulang setelan ilmiah EXP10 GPU pada engine EXP11 |
| `config.features_only.yaml` | Kontrol + 24 fitur perilaku; budget/loss/model tetap |
| `config.gpu16gb.yaml` | Paket utama EXP11 |
| `config.no_behavior.yaml` | Paket utama tanpa 24 fitur perilaku |
| `config.self_only.yaml` | Jalur self-only LayerNorm untuk mengukur kontribusi graf |
| `config.recent_refit.yaml` | Protokol terpisah 80/5/15; test 15% terakhir tetap |

Untuk pilot berukuran sama, gunakan `--max-rows 1000000 --epochs 30 --strategy
uniform --seed 42`. Tetapkan protokol dengan validation dan laporkan semua
ablation, termasuk yang lebih buruk. F2 mengubah kompromi precision/recall;
bandingkan AP dan angka FP juga, bukan hanya recall.

## Memori, output, dan restart

Fitur disimpan float16 di CPU; batch dikonversi ke FP32 sebelum agregasi/model.
24 fitur baru menambah sekitar **1,09 GiB** untuk 24.386.900 transaksi, di luar
fitur lama, DataFrame, indeks histori, dan tensor graf. Preprocessing melakukan
sort/search vektor per kelompok dan masih membutuhkan RAM besar. Preset GPU
16 GB bukan jaminan full-data muat pada RAM host yang kecil.

Output: `result/exp11/`, checkpoint/cache: `model/exp11/`; smoke terpisah.
`metrics.json` memuat metrik/per kanal/per periode, `training.selection_window`,
bobot subtype, dan kebijakan perilaku. `history.json` merekam selection score,
roots, learning rate dan timing; `status.json` merekam alasan berhenti.
`best.pt` menyimpan state model, encoder, schema, threshold, kebijakan fitur,
dan identitas sumber/data. Histori mentah/counter yang dibutuhkan untuk serving
harus direkonstruksi dari observasi sebelumnya; checkpoint bukan layanan online.

Attempt selesai dilewati. Attempt gagal/interrupted dimulai ulang di folder
baru, **bukan resume optimizer**. EXP10 dan hasilnya tidak diubah.

## Pengujian dan audit

```powershell
& .\code\exp6_gpu_tqdm\.venv\Scripts\python.exe -m unittest discover -s code/exp11_behavioral_temporal/tests -t code/exp11_behavioral_temporal -v
& .\code\exp6_gpu_tqdm\.venv\Scripts\python.exe .\code\exp11_behavioral_temporal\audit_exp10.py
```

Hasil pengujian/pilot dan batas klaim: [VALIDATION.md](VALIDATION.md).

Pilot lokal satu juta baris: AP test kontrol **0,004890**, fitur-only
**0,115679**, paket utama **0,062566**. F1 pada threshold validation masing-masing
**0,001734 / 0,076555 / 0,126246**. Jadi fitur-only lebih baik pada ranking,
sementara paket utama lebih baik pada F1 dengan kebijakan threshold-nya.
Recall utama masih 9,95%; full-data dan multi-seed belum diuji.
