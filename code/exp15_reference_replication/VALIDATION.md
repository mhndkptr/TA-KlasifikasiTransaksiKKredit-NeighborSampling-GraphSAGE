# Validasi EXP15

Status: implementasi dan audit full CSV tersedia; training model full-data belum dijalankan.

Validasi otomatis mencakup parsing IBM, batas rolling window, pemilihan anchor hanya dari dukungan train, graph-level undersampling, namespace graf, FI-GRL sparse formula, metrik edge case, dan pipeline kelima varian dengan backend kompatibilitas.

`doctor` pada environment proyek Python 3.13 menemukan NumPy, pandas, SciPy, scikit-learn, PyTorch, tqdm, dan psutil. XGBoost, TensorFlow, StellarGraph, MATLAB Engine, dan executable MATLAB belum tersedia. Karena itu full replication dengan `config.reference.yaml` belum dapat dijalankan pada environment tersebut. Jalur kompatibilitas tersedia untuk pengujian implementasi, bukan sebagai pengganti hasil utama.

Audit full `credit_card_transactions-ibm_v2.csv` selesai dalam sekitar 124 detik dan mencatat 24.386.900 transaksi, 29.757 fraud, 6.139 client kartu, 100.343 merchant, serta rentang 2 Januari 1991–28 Februari 2020. Anchor pertama yang memenuhi minimum 25 fraud train adalah 13 Juni 2001. Dukungan fold yang dikunci:

| Fold | Train | Fraud train | Holdout | Fraud holdout | Eligible default |
| --- | ---: | ---: | ---: | ---: | --- |
| fold_1 | 8.707 | 27 | 3.501 | 1 | ya |
| fold_2 | 8.600 | 19 | 3.612 | 2 | tidak |
| fold_3 | 8.608 | 11 | 3.665 | 7 | tidak |
| fold_4 | 8.755 | 9 | 3.698 | 17 | tidak |
| fold_5 | 8.766 | 25 | 3.553 | 6 | ya |

Fold rendah dukungan tetap ada di manifest dan bisa dijalankan dengan `--fold` eksplisit. Hasilnya harus diberi keterangan dukungan train di bawah guard awal.

Pilot data nyata `features_only`, fold 1, seed 42 selesai memakai backend `sklearn_random_forest_compat`. Graf train mempunyai 8.707 transaksi, 293 client, 1.543 merchant, dan 2.499 fitur transaksi; holdout mempunyai 3.501 transaksi dengan hanya satu fraud. AP pilot adalah 0,000285633, sama dengan prevalensi, ROC-AUC 0,479143, Lift@1% 0, dan fraud tersebut tidak terdeteksi pada threshold 0,5. Angka ini hanya memvalidasi alur data/artefak; satu fraud holdout dan backend kompatibilitas tidak mendukung kesimpulan performa penelitian.

Preset smoke pada `User0_credit_card_transactions.csv` juga selesai untuk `features_only`, `hinsage_plus_features`, dan `figrl_plus_features`. Window smoke berisi 48 transaksi train dengan satu fraud serta 31 holdout dengan tujuh fraud. Ketiga run menghasilkan AP 0,225806 pada backend kompatibilitas. Kesamaan itu hanya menunjukkan pipeline selesai; ukuran data sangat kecil, embedding 8 dimensi, satu epoch, dan classifier 10 tree sengaja berbeda dari preset referensi.

Isi dokumen ini perlu diperbarui setelah pilot satu fold dan matriks utama benar-benar selesai. Jangan menaruh metrik smoke sebagai hasil penelitian.
