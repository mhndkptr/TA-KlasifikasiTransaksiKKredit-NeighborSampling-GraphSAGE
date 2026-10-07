# Audit parity referensi EXP15

Sumber lokal yang menjadi acuan berada pada commit `b60d1fa93dce25b536aaf90c7860076f8d9377b1`. Perintah `audit-reference` menghitung ulang commit dan SHA-256 file penting ke `result/exp15/reference_manifest.json`.

## Parameter yang terbukti dari notebook/source

- Rolling window 17 hari, langkah 5 hari, dan holdout 5 hari.
- Graf tripartit tidak berarah dengan fitur hanya pada transaksi serta atribut konstan pada client/merchant.
- HinSAGE dua layer berukuran 64, neighbor samples `[2, 32]`, dropout 0, L2 normalization, aktivasi `relu` lalu `linear`.
- Dense sigmoid, binary cross-entropy, Adam 0,001, batch 50, 10 epoch, pembagian posisi 80/20.
- FI-GRL intermediate dimension 400 dan final dimension 64 melalui MATLAB.
- XGBoost 100 estimator, AP, precision–recall curve, dan Lift@1%.
- Undersampling diterapkan pada transaksi train sebelum graf dibangun.

## Adaptasi IBM

- Client adalah kunci komposit `User::Card`; merchant berasal dari `Merchant Name` yang dibaca tanpa konversi float.
- Encoder numerik/kategorikal fit pada seluruh train window sebelum undersampling.
- Lima window literal pertama tidak memiliki fraud. Anchor dipilih dengan aturan dukungan train yang deklaratif di PLAN.
- Seed eksplisit, artefak atomik, status run, metrik tambahan, dan backend kompatibilitas ditambahkan untuk auditabilitas.

## Hal yang belum diketahui

- Recipe `preprocessed_ccf.csv` yang digunakan penulis.
- Lock lengkap dependency historis dan default XGBoost persis pada mesin penulis.
- Grid lengkap undersampling pada paper; notebook yang tersedia hanya menunjukkan parameter mekanismenya.

Backend `stellargraph`, `matlab`, dan `xgboost` adalah jalur referensi. Backend dengan suffix `_compat` adalah implementasi lokal untuk validasi pipeline dan tidak membuktikan parity numerik.

