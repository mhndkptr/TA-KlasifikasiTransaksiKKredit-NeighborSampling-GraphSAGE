# Rekap Metrik EXP1-EXP14

File ini dibuat otomatis oleh `generate_recap.py` dari artefak lokal di folder `result/`.
Metrik utama diseragamkan sebagai **AP/AUPRC**: eksperimen lama memakai kolom
`auprc`, EXP14 memakai kolom `ap`, dan eksperimen baru memakai `auprc`/`ap`
sesuai artefak masing-masing.

Penting: tabel lintas eksperimen ini berguna untuk rekap perkembangan, tetapi
tidak semua baris apple-to-apple. Beberapa eksperimen adalah prefix/smoke/pilot
atau ablation, sebagian memakai fitur, split, root sampling, threshold, dan
hardware yang berbeda. Untuk klaim penelitian, bandingkan strategi dalam
comparison/protokol yang sama.

## Best Local Result Per Experiment

| Experiment | Judul | Best strategy/run | AP/AUPRC | Precision | Recall | F1 | Best val AP | Val-test gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP1 | Starter GraphSAGE | uniform | 0.003299 | 0.003745 | 0.050847 | 0.006977 | 0.325354 | 0.322055 |
| EXP2 | GPU-bound GraphSAGE | uniform | 0.002827 | 0.004499 | 0.067797 | 0.008439 | 0.248462 | 0.245635 |
| EXP3 | Stable training/evaluation | uniform | 0.003483 | 0.000000 | 0.000000 | 0.000000 | 0.297554 | 0.294071 |
| EXP4 | GPU-bound stable evaluation | uniform | 0.003089 | 0.000000 | 0.000000 | 0.000000 | 0.276980 | 0.273891 |
| EXP5 | Corrected GPU sampler | uniform | 0.012105 | 0.002798 | 0.720930 | 0.005574 | 0.011416 | -0.000690 |
| EXP6 | Imbalance calibration | topology | 0.023791 | 0.769841 | 0.021778 | 0.042358 | 0.665948 | 0.642157 |
| EXP7 | Memory-bounded importance | importance | 0.011209 | 0.010101 | 0.023256 | 0.014085 | 0.016995 | 0.005786 |
| EXP8 | Adaptive batched importance | importance | 0.023619 | 0.774194 | 0.021554 | 0.041940 | 0.659854 | 0.636234 |
| EXP9 | Reference sampling | topology | 0.025862 | 0.004139 | 0.144365 | 0.008048 | 0.653045 | 0.627183 |
| EXP10 | Temporal robust | uniform | 0.035268 | 0.037763 | 0.065335 | 0.047862 | 0.722792 | 0.687524 |
| EXP11 | Behavioral temporal | topology | 0.187791 | 0.303934 | 0.208128 | 0.247068 | 0.829810 | 0.642019 |
| EXP12 | Recent context | uniform | 0.248571 | 0.428743 | 0.239784 | 0.307559 | 0.845463 | 0.596892 |
| EXP13 | Kaggle T4 deployment |  |  |  |  |  |  |  |
| EXP14 | Local temporal R0 | uniform | 0.187102 | 0.258621 | 0.259317 | 0.258969 |  |  |

## Summary Per Strategy

| Experiment | Strategy | Runs | Mean AP/AUPRC | Max AP/AUPRC | Mean precision | Mean recall | Mean F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EXP1 | uniform | 5 | 0.002955 | 0.003299 | 0.003125 | 0.074576 | 0.005942 |
| EXP2 | uniform | 5 | 0.002411 | 0.002827 | 0.002515 | 0.071186 | 0.004760 |
| EXP3 | uniform | 5 | 0.002865 | 0.003483 | 0.000000 | 0.000000 | 0.000000 |
| EXP4 | uniform | 5 | 0.002640 | 0.003089 | 0.000000 | 0.000000 | 0.000000 |
| EXP5 | uniform | 1 | 0.012105 | 0.012105 | 0.002798 | 0.720930 | 0.005574 |
| EXP6 | topology | 1 | 0.023791 | 0.023791 | 0.769841 | 0.021778 | 0.042358 |
| EXP6 | uniform | 1 | 0.023460 | 0.023460 | 0.695652 | 0.021554 | 0.041812 |
| EXP7 | importance | 1 | 0.011209 | 0.011209 | 0.010101 | 0.023256 | 0.014085 |
| EXP7 | topology | 1 | 0.010802 | 0.010802 | 0.008197 | 0.023256 | 0.012121 |
| EXP7 | uniform | 1 | 0.010800 | 0.010800 | 0.008197 | 0.023256 | 0.012121 |
| EXP8 | importance | 1 | 0.023619 | 0.023619 | 0.774194 | 0.021554 | 0.041940 |
| EXP9 | importance | 1 | 0.025448 | 0.025448 | 0.004298 | 0.152223 | 0.008360 |
| EXP9 | topology | 1 | 0.025862 | 0.025862 | 0.004139 | 0.144365 | 0.008048 |
| EXP9 | uniform | 1 | 0.025288 | 0.025288 | 0.003583 | 0.159183 | 0.007009 |
| EXP10 | importance | 1 | 0.029190 | 0.029190 | 0.021182 | 0.061069 | 0.031454 |
| EXP10 | topology | 1 | 0.028505 | 0.028505 | 0.021190 | 0.051414 | 0.030011 |
| EXP10 | uniform | 3 | 0.014749 | 0.035268 | 0.014817 | 0.156159 | 0.020306 |
| EXP11 | importance | 1 | 0.153851 | 0.153851 | 0.185675 | 0.255501 | 0.215062 |
| EXP11 | topology | 1 | 0.187791 | 0.187791 | 0.303934 | 0.208128 | 0.247068 |
| EXP11 | uniform | 4 | 0.089464 | 0.174721 | 0.206638 | 0.096501 | 0.106272 |
| EXP12 | importance | 1 | 0.224387 | 0.224387 | 0.294223 | 0.258419 | 0.275161 |
| EXP12 | topology | 2 | 0.214479 | 0.220823 | 0.251788 | 0.268747 | 0.255068 |
| EXP12 | uniform | 13 | 0.173205 | 0.248571 | 0.276639 | 0.192664 | 0.222523 |
| EXP14 | importance | 1 | 0.143271 | 0.143271 | 0.198164 | 0.232600 | 0.214005 |
| EXP14 | topology | 1 | 0.157930 | 0.157930 | 0.218816 | 0.232375 | 0.225392 |
| EXP14 | uniform | 1 | 0.187102 | 0.187102 | 0.258621 | 0.259317 | 0.258969 |

## Gambar

- `figures/ap_by_experiment.png`
- `figures/metrics_by_best_experiment.png`
- `figures/strategy_ap_by_experiment.png`
- `figures/precision_recall_scatter.png`
- `figures/val_test_gap_by_experiment.png`
