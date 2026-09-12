# Audit eksperimen sebelumnya untuk keputusan EXP13

Audit ini membaca artefak lokal EXP9–EXP12 tanpa mengubahnya. Tujuannya adalah
memisahkan temuan yang layak diteruskan dari hasil yang masih eksploratif.

## Hasil yang paling dapat dibandingkan

Cohort full-data EXP12 dengan `comparison_id=7bdae5fe466252c7` memakai
konfigurasi yang sama untuk ketiga sampler: contextual encoder, LayerNorm,
balanced roots 50 normal per fraud, seed 42.

| Sampler | Test AUPRC | Precision | Recall | F1 | Peak VRAM | Durasi |
|---|---:|---:|---:|---:|---:|---:|
| Uniform | 0,207345 | 0,251586 | 0,267176 | 0,259146 | 1,436 GiB | 199 s |
| Topology | 0,220823 | 0,304762 | 0,251459 | 0,275557 | 1,691 GiB | 217 s |
| Importance | 0,224387 | 0,294223 | 0,258419 | 0,275161 | 1,691 GiB | 208 s |

Topology dan importance masing-masing naik sekitar 6,50% dan 8,22% AUPRC
terhadap uniform pada seed tunggal. Ini sinyal yang menjanjikan, bukan bukti
signifikan; EXP13 harus memakai beberapa seed dengan konfigurasi identik.

## Kelebihan eksperimen sebelumnya

- AUPRC naik besar dari EXP9 (~0,025), EXP10 (0,0285–0,0353), EXP11
  (0,1539–0,1878), hingga cohort fair EXP12 (0,2073–0,2244).
- Pilot feature ablation menunjukkan sinyal contextual: control 0,062566 menjadi
  features-only 0,166574 pada prefix satu juta baris.
- Pilot self-only 0,119762 lebih rendah dari main 0,166574, sehingga graph
  context memberi sinyal tambahan pada pilot tersebut.
- Pipeline sudah menyimpan config, history, metrics, threshold policy, timing,
  per-channel metrics, dan source manifest.
- Objective balanced-root lama tidak melakukan double weighting: `pos_weight=1`.

## Kekurangan dan risiko interpretasi

1. Semua full run hanya seed 42 dan dijalankan pada RTX 5060 Ti, bukan Kaggle
   T4. Belum ada distribusi antar-seed maupun benchmark hardware target.
2. Kenaikan lintas EXP9–EXP12 confounded oleh perubahan fitur, loss, root ratio,
   normalization, dan validation policy. Kenaikan itu tidak boleh seluruhnya
   diatribusikan ke sampler.
3. Sweep uniform N:P 10/20/25/30/50 memberi test AUPRC terbaik pada 25
   (0,248571), tetapi validation selection terbaik justru N:P=30
   (`selection_score=0,190506`). Memilih 25 karena melihat test adalah test-set
   selection leakage. Bila track balanced dilanjutkan, freeze 30 dari validation.
4. Fraud sangat drift menurut kanal. Train fraud Chip/Online/Swipe adalah
   256/14.771/5.859, sedangkan test menjadi 3.907/128/419. Pada cohort fair,
   chip AP uniform 0,221266 justru melebihi importance 0,211916 dan topology
   0,210409. Custom sampler terutama memperbaiki precision/FP overall, bukan
   recall subtype utama.
5. Nama `best_val_auprc` lama menyimpan global validation AP sekitar 0,85,
   sedangkan checkpoint sebenarnya dipilih oleh recent-selection AP sekitar
   0,16–0,18. EXP13 perlu membedakan global validation AP, selection AP, dan
   test AP secara eksplisit dalam laporan.
6. Full run pandas lama mempunyai timestamp tie pada kedua batas split. EXP13
   memakai boundary tie-safe; jumlah baris bisa sedikit berbeda dan bukan
   reproduksi bit-identik.
7. Snapshot graph train aman terhadap held-out leakage, tetapi root train lama
   masih dapat melihat transaksi train yang terjadi lebih akhir. Protokol itu
   adalah frozen end-of-train snapshot, bukan causal event-time training. Untuk
   klaim deployment real-time diperlukan rolling/causal sampler.
8. Fanout `[25,10]` tidak berarti dua operasi pemangkasan independen. Transaction
   selalu mempunyai dua endpoint; fanout 10 pada hop tersebut adalah no-op.
9. Full EXP12 terbaik memakai 220 fitur float16 (~9,99 GiB) sebelum adjacency,
   DataFrame, dan workspace. Preset out-of-core EXP13 hanya 15 fitur (~698 MiB)
   dan belum memiliki bukti full-data. Fitur compact juga memakai ordinal code
   untuk `Use Chip`/MCC dan biner untuk error, sehingga lebih lemah secara
   representasi daripada contextual encoder.
10. Full importance EXP12 lama memakai default literal transaction degree yang
    konstan. `projected_transaction` pada EXP13 adalah perbaikan masuk akal,
    tetapi harus diuji melawan ablation literal sebelum klaim centrality.

## Keputusan untuk EXP13

- Jalankan track proposal-aligned lebih dulu: compact out-of-core features,
  BatchNorm, root alami, dynamic train-only `pos_weight`, batch 1.024, dan T4.
- Jangan membandingkan nilainya langsung dengan EXP12 feature-rich seolah hanya
  sampler/hardware yang berubah.
- Setelah uniform sehat, jalankan topology dan importance pada config, seed,
  split, feature schema, dan threshold policy yang sama.
- Tambahkan seed 42–46, rolling-origin backtest, AP per kanal/bulan,
  precision@alert-budget, recall@fixed-FPR, serta baseline MLP/LightGBM dan
  self-only full-data.
- Karena test 2018–2020 sudah berkali-kali dianalisis, nyatakan semua hasil saat
  ini eksploratif atau kunci external/new-period holdout untuk klaim final.

Sumber angka utama: `result/exp12/summary.csv`, `result/exp12/runs.csv`, dan
masing-masing `metrics.json` pada folder dengan comparison ID di atas.
