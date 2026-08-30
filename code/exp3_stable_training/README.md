# Experiment 3 - Stable Training dan Evaluation

Experiment 3 diturunkan dari `exp1_starter` untuk memperbaiki reliabilitas
early stopping pada AUPRC. Arsitektur GraphSAGE, split temporal, fitur, loss,
dan tiga strategi sampling tetap sama agar hasil masih dapat dibandingkan.

## Mengapa EXP1 dapat berhenti pada epoch 29?

Early stopping berarti AUPRC validation terbaik tidak terlampaui selama jumlah
epoch yang ditentukan. Ini normal dan checkpoint terbaik tetap dipakai. Pada
EXP1 ada sumber noise tambahan: satu `WeightedSampler` yang stateful dipakai
bersama oleh training, validation, dan test. Tetangga validation berubah pada
setiap epoch, lalu validation ikut menggeser RNG training. Lonjakan AUPRC sesaat
dapat tercatat sebagai nilai terbaik yang sulit dilampaui.

EXP3 memisahkan sampler training/evaluation, memakai seed evaluasi tetap,
merata-ratakan tiga sampling pass, memakai `min_delta`, memberi scheduler waktu
lebih panjang, memilih threshold F1 hanya dari validation, dan mencatat jumlah
fraud setiap split. Label test tidak dipakai untuk training, checkpoint, atau
pemilihan threshold.

## Menjalankan

```powershell
cd code/exp3_stable_training
python run.py --config config.yaml --strategy uniform --seed 42 --max-rows 10000
```

Hapus `--max-rows 10000` untuk menjalankan 200.000 baris sesuai konfigurasi.
Jalankan `python run.py --config config.yaml` untuk seluruh strategi dan seed.

Artefak memakai awalan `exp3_`, sehingga hasil lama tidak ditimpa. Tiga pass
membuat tahap evaluasi sekitar tiga kali lebih lama, tetapi training batch tetap
satu pass. Early stopping tetap mungkin terjadi dan bukan kegagalan; bandingkan
`best_epoch`, AUPRC test, F1/Recall, jumlah fraud per split, serta rata-rata dan
standar deviasi antar-seed.
