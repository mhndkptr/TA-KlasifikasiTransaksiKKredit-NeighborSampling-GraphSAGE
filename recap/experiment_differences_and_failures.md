# Perbedaan dan Penyebab Kegagalan EXP1-EXP14

Dokumen ini merangkum perubahan utama, hasil, dan penyebab kegagalan/limitasi
setiap eksperimen berdasarkan artefak lokal `code/` dan `result/`.

## Ringkasan Besar

Perkembangan eksperimen terbagi menjadi empat fase:

1. EXP1-EXP5 membangun baseline GraphSAGE dan memperbaiki masalah implementasi GPU/sampler.
2. EXP6-EXP8 menunjukkan bahwa masalah utama bukan hanya performa GPU, tetapi generalisasi temporal: validation AP tinggi, test AP rendah.
3. EXP9-EXP12 memperbaiki protokol temporal, fitur, root balancing, dan validation policy; performa test naik besar, tetapi banyak angka tidak langsung sebanding karena perubahan fitur/protokol.
4. EXP13-EXP14 mengarah ke deployment/protokol penelitian yang lebih eksplisit; EXP14 memberi perbandingan lokal R0, tetapi baru seed tunggal sehingga masih eksploratif.

## Tabel Perbedaan

| Experiment | Fokus perubahan | Hasil/limitasi utama | Penyebab kegagalan atau risiko |
|---|---|---|---|
| EXP1 | Baseline GraphSAGE, split temporal 70/15/15, tiga sampler, prefix 200k | AP/AUPRC sangat rendah pada test | Prefix kecil dan tidak representatif, imbalance ekstrem, temporal drift, sampler evaluasi masih noisy |
| EXP2 | Jalur GPU-bound untuk fitur, CSR, sampling, forward/backward | Performa komputasi diperbaiki, tetapi protokol/metrik belum menyelesaikan kualitas model | Kebutuhan VRAM naik, candidate Gumbel top-k aproksimatif, tidak langsung sebanding dengan sampler CPU EXP1 |
| EXP3 | Stable evaluation dari EXP1 | Evaluasi lebih reproducible | Masalah generalisasi dan data drift tetap ada; evaluasi lebih stabil tidak otomatis menaikkan ranking test |
| EXP4 | GPU-bound + stable evaluation | Menggabungkan EXP2 dan EXP3 | Masih memakai candidate sampler aproksimatif; risiko OOM dan gap validation-test belum selesai |
| EXP5 | Corrected GPU sampler eksak, threshold fixed 0.5 | Memperbaiki kesetaraan sampler GPU | Threshold fixed menghasilkan banyak false positive; validation kuat tetapi test tetap rendah karena drift |
| EXP6 | Threshold validation, gradient clipping, full-data GPU | Validation AP sekitar 0.66, test AP sekitar 0.023-0.024 | Validation-test gap sangat besar; masalah dominan adalah temporal drift dan threshold overfitting |
| EXP7 | Memory-bounded importance/PPR | Mengurangi risiko OOM untuk importance | Memori importance membaik, tetapi kualitas ranking test tidak membaik banyak; sampler bukan bottleneck utama |
| EXP8 | Adaptive batched importance dan cache bobot | Importance jauh lebih cepat | Optimasi runtime tidak mengubah representasi/protokol, sehingga AP test tetap rendah |
| EXP9 | Implementasi reference sampling mandiri, frozen train graph, anti-leakage lebih jelas | AP test sekitar 0.025 pada artefak lokal | Menghapus leakage/bug tidak cukup; fitur dan validation masih belum cocok dengan distribusi test |
| EXP10 | Robust features, balanced roots, temporal selection metric | AP test naik ke sekitar 0.028-0.035 | Drift fraud online ke chip masih berat; balanced roots mengubah budget komputasi; threshold/selection belum representatif |
| EXP11 | 24 fitur perilaku strictly-past dan validation terbaru | Full-data AP naik ke sekitar 0.154-0.188 | Selection window hanya punya sedikit fraud chip, sehingga checkpoint belum mewakili test chip-heavy |
| EXP12 | 65 fitur recent context dan validation recent-interleaved | Banyak run mencapai AP sekitar 0.20-0.25; fair cohort tiga sampler sekitar 0.207-0.224 | Banyak ablation/pilot tidak sebanding; test set sudah sering dianalisis; feature-rich lokal tidak sama dengan preset proposal compact |
| EXP13 | Deployment Kaggle T4 proposal-aligned out-of-core | Tidak ada metrik lokal `result/exp13` pada workspace ini | Layer deployment sudah siap, tetapi hasil tidak bisa direkap tanpa output Kaggle; 15 fitur compact tidak boleh dibandingkan langsung dengan EXP12 feature-rich |
| EXP14 | Perbandingan lokal R0 uniform/topology/importance dengan protokol eksplisit | Pada seed 42 R0, uniform tertinggi di artefak lokal; topology/importance belum mengalahkan uniform | Baru seed tunggal, hasil eksploratif, test lama sudah diketahui; custom sampler belum terbukti membantu pada protokol R0 |

## Catatan Interpretasi

- Metrik utama di rekap diseragamkan sebagai AP/AUPRC, tetapi nama kolom berbeda antar eksperimen (`auprc`, `ap`, atau agregat `*_mean`).
- EXP10-EXP12 berisi pilot dan ablation dalam folder hasil utama. Karena itu `experiment_metrics_detail.csv` menyimpan semua run, sementara `experiment_best_by_experiment.csv` hanya memilih nilai lokal terbaik per nomor eksperimen.
- Nilai “terbaik” lintas eksperimen bukan bukti bahwa satu perubahan tunggal menyebabkan kenaikan. Fitur, split, threshold, root sampling, loss, hardware, dan validation policy berubah berkali-kali.
- Perbandingan sampler yang paling jujur harus dilakukan pada strategy berbeda dengan `comparison_id`/protokol/seed yang sama.
- EXP13 dicatat sebagai eksperimen karena source dan deployment layer tersedia, tetapi tidak ada hasil metrik lokal yang bisa digrafikkan.

## Penyebab Gagal yang Berulang

1. **Temporal drift:** distribusi fraud berubah antar periode, terutama pergeseran kanal online/chip.
2. **Validation-test mismatch:** checkpoint/threshold tampak bagus di validation tetapi tidak mewakili test.
3. **Class imbalance ekstrem:** tanpa root sampling, pos_weight, atau threshold policy yang hati-hati, model mudah menghasilkan recall rendah atau false positive besar.
4. **Sampler bukan satu-satunya faktor:** optimasi uniform/topology/importance tidak cukup jika fitur/protokol belum menangkap drift.
5. **Ablation confounding:** kenaikan EXP9 ke EXP12 dipengaruhi banyak perubahan sekaligus, bukan hanya neighbor sampling.
6. **Test-set reuse:** test lama sudah dipakai untuk banyak audit, sehingga hasil terbaru harus ditulis sebagai eksploratif sampai ada holdout baru atau rolling evaluation yang dikunci.
