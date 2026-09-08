# EXP10 — Temporal robustness dan evaluasi yang sebanding

EXP10 menindaklanjuti AP validation EXP9 sekitar 0,65 yang turun menjadi 0,025
pada test. Audit full CSV menemukan perubahan fraud online menjadi fraud chip.
Penjelasan angka dan ketujuh referensi ada di [ANALISIS_EXP9.md](ANALISIS_EXP9.md).
EXP9 serta hasil aslinya dipertahankan.

## Menjalankan

Dari root repository, gunakan Python environment yang memiliki dependensi
[requirements.txt](requirements.txt). Environment proyek yang sudah tersedia
dapat dipakai tanpa instalasi ulang:

```powershell
# Uji alur: 100.000 baris awal CSV, tiga strategi, tiga epoch, seed 42.
& .\code\exp6_gpu_tqdm\.venv\Scripts\python.exe .\code\exp10_temporal_robust\run.py --config .\code\exp10_temporal_robust\config.smoke.yaml --no-progress

# Full data, uniform terlebih dahulu pada komputer training 16 GB.
& .\code\exp6_gpu_tqdm\.venv\Scripts\python.exe .\code\exp10_temporal_robust\run.py --config .\code\exp10_temporal_robust\config.gpu16gb.yaml --strategy uniform --seed 42

# Ketiga strategi x lima seed, setelah pilot dan protokol ditetapkan.
& .\code\exp6_gpu_tqdm\.venv\Scripts\python.exe .\code\exp10_temporal_robust\run.py --config .\code\exp10_temporal_robust\config.gpu16gb.yaml
```

Jika berada di `code/exp10_temporal_robust` dan environment sudah aktif, perintah
pendeknya `python run.py --config config.gpu16gb.yaml --strategy uniform --seed 42`.
Path CSV/cache/result di YAML diselesaikan relatif terhadap file YAML, sehingga
perintah tidak bergantung pada current working directory.

## Perubahan utama

- One-hot untuk `Use Chip`, `MCC`, `Errors?`; missing dan OOV eksplisit.
  Seluruh vocabulary dan statistik dipelajari hanya dari training.
- Amount signed-log, median/IQR, clip ±10; waktu siklik; indikator online,
  refund, error, dan kelangkaan lokasi. Kolom `Is Fraud?` tidak menjadi fitur.
- Semua fraud train sekali per epoch dan maksimal 20 normal/fraud dipilih
  tanpa replacement. Semua strategi tetap menggunakan seluruh graf train.
  Root balancing berbeda dari neighbor sampling dan diterapkan sama pada ketiganya.
- BCE `pos_weight=1` setelah root balancing; konfigurasi menolak inverse-frequency
  weight bersamaan dengan balanced roots. Weight decay 0,0001.
- Checkpoint memakai geometric mean AP-lift dari tiga blok waktu validation:
  `exp(mean(log(AP_block / prevalence_block)))`. Blok satu kelas dicatat dan
  dikecualikan; bila semua blok tidak layak, proses dihentikan. AP global tetap
  dilaporkan. Mode `selection_metric: ap` tersedia sebagai ablation.
- Threshold memaksimalkan F1 fraud pada paruh terakhir validation. Semua artefak
  utama memakai threshold itu. Kalibrasi dihentikan jika segmen tidak punya dua kelas.
  Test hanya dievaluasi sesudah checkpoint dan threshold ditetapkan.

GraphSAGE default tetap dua layer mean, hidden 256, BatchNorm, dropout 0,2,
fanout 25/10, Adam dan komputasi FP32. Freeze histori sama seperti EXP9:
train memakai snapshot seluruh train, query val/test tidak dimasukkan ke histori.
Ini bukan graf strictly causal untuk setiap transaksi di dalam periode training.

## Memori dan biaya

One-hot menambah dimensi fitur. `storage_dtype: float16` menghemat penyimpanan
fitur, tetapi setiap batch dikonversi ke FP32 **sebelum** agregasi/model. Ini
kuantisasi fitur, bukan AMP. `float32` tersedia untuk kontrol presisi.
Preset 16 GB sengaja menyimpan fitur di **CPU**, batch train 4.096 dan eval 8.192.
Jumlah byte fitur dilaporkan oleh preprocessing. Preprocessing full CSV tetap
membutuhkan RAM besar; preset bukan jaminan muat untuk setiap skema/dataset.

Balanced roots mengurangi jumlah update per epoch. Pada full IBM, 438.606
roots/epoch dibanding 17.070.830 pada uniform-root EXP9. Bandingkan jumlah roots,
update, waktu dan kualitas; jumlah epoch sama bukan budget komputasi sama.
Angka latency query memakai embedding entitas yang telah dihitung; lihat biaya
preparation terpisah sebelum menafsirkan sebagai performa sistem produksi.

## Ablation yang disediakan

Gunakan `--strategy uniform --seed 42` untuk pilot setiap konfigurasi, kemudian
tetapkan protokol menggunakan validation sebelum menjalankan perbandingan seed.

| Konfigurasi | Yang diukur |
|---|---|
| `config.legacy_control.yaml` | Fitur, loss, root sampling, model, checkpoint AP dan threshold utama 0,5 setara scientific settings EXP9 GPU; runtime feature store CPU |
| `config.features_only.yaml` | Hanya robust features + storage float16 terhadap legacy control |
| `config.balanced_only.yaml` | Tambah balanced roots/weight 1 terhadap features-only |
| `config.gpu16gb.yaml` | Paket utama: tambah regularisasi dan validasi beberapa periode/threshold terbaru |
| `config.layernorm.yaml` | Ganti BatchNorm dengan LayerNorm terhadap paket utama |
| `config.self_only.yaml` | Pasangan LayerNorm dengan kontribusi tetangga nol untuk mengukur nilai tambah graf |
| `config.recent_refit.yaml` | **Protokol berbeda** 80/5/15, lebih banyak label/histori baru, test 15% akhir tetap sama |

Self-only setara jalur MLP dua layer dengan bias dan LayerNorm; masih membangun
graf untuk memakai pipeline audit yang sama. Latency-nya bukan benchmark MLP
minimal. Tidak boleh mencampur hasilnya dalam klaim tiga strategi GraphSAGE.
Recent-refit adalah sekali refit sebelum test, belum rolling training per bulan.

## Output dan cara membaca

Run utama berada di `result/exp10/<name>_<comparison_id>_<strategy>_seed42/`.
Smoke berada di `result/exp10_smoke/`; model/cache memakai direktori terpisah.

- `metrics.json`: AP, ROC-AUC, F1 fraud, F1-macro, GMean, specificity/FPR,
  prevalensi, AP-lift, alert rate; fixed 0,5 dan validation threshold; metrik
  val/test per periode dan kanal; quantile skor serta probabilitas jenuh.
- `history.json`: loss, global val AP, selection score, AP-lift setiap blok,
  roots/epoch, checkpoint, learning rate, dan timing.
- `summary.csv`: mean/std/count per strategi dan comparison ID, termasuk
  `roc_auc_mean`, `f1_macro_mean`, `gmean_mean`, `decision_threshold_mean`.
  `f1_mean` tetap **F1 fraud**. `auprc_mean` adalah AP, bukan ROC-AUC.
- `runs.csv`: nilai tiap seed termasuk threshold policy.
- Checkpoint: model, encoder, config, fingerprint, epoch, metric selection dan threshold.
  `best_val_auprc` adalah AP checkpoint terpilih; `max_observed_val_auprc` menyimpan
  AP tertinggi di semua epoch, yang dapat berbeda pada selection temporal.

Comparison ID membedakan seluruh pengaturan ilmiah, sumber kode, data dan
lingkungan. Satu seed menghasilkan std kosong. Attempt ulang memakai folder baru;
ringkasan memakai attempt selesai terakhir per seed. Ini restart, bukan resume optimizer.

## Audit dan pengujian

```powershell
& .\code\exp6_gpu_tqdm\.venv\Scripts\python.exe -m unittest discover -s code/exp10_temporal_robust/tests -t code/exp10_temporal_robust -v
& .\code\exp6_gpu_tqdm\.venv\Scripts\python.exe .\code\exp10_temporal_robust\audit_exp9.py --data .\dataset\credit_card_transactions-ibm_v2.csv --output .\code\exp10_temporal_robust\validation\exp9_audit.json
```

Audit tanggal cutoff dalam `audit_exp9.py` khusus full-data EXP9 yang diberikan.
Jangan memakai cutoff itu untuk subset/split lain. Audit mengeluarkan agregat,
tanpa menyalin transaksi individu. Status pengujian: [VALIDATION.md](VALIDATION.md).
Kenaikan AP full-data belum dijamin; samakan protokol/multi-seed dan periksa
performa fraud chip sebelum menyimpulkan perbaikan generalisasi.
