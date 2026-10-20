# Penjelasan Pipeline Eksperimen 21B (EXP21B)
## *Balanced Hyperparameter Tuning, Tree Regularization, & GNN Representation Contribution*

---

## 1. Latar Belakang & Konteks Investigasi Ilmiah

Pada eksperimen **EXP21**, model Hybrid GraphSAGE + XGBoost menghasilkan metrik evaluasi yang *too good to be true*:
* **Recall**: **100.0%** (seluruh 421 kasus fraud tertangkap sempurna tanpa satu pun missed fraud / False Negative)
* **Precision**: **88.63%**
* **F1-Score**: **0.9397**
* **ROC-AUC**: **0.9999**

Berdasarkan saran pembimbing, dilakukan audit internal langsung terhadap model pohon keputusan tersimpan (`model/exp21/v5/exp21_uniform_xgb.json`). Audit membuktikan bahwa pohon XGBoost sekadar **menghafal ID kuasi sintetis**:
1. Fitur `Zip` (`f4`) memiliki total split gain **30,7 juta** (split count = 100).
2. Fitur `Merchant City` (`f2`) memiliki total split gain **9,25 juta** (split count = 200).
3. Keduanya menyumbang **>99,9% keputusan pohon**, menenggelamkan 128 dimensi representasi struktural GraphSAGE (<25 gain).
4. `scale_pos_weight = 732.18` (rasio penuh) memaksa pohon membuat daun-daun mikro individual untuk menghindari penalti False Negative.

Pada **EXP21A**, kebocoran ini telah diatasi melalui pembersihan atribut (`Zip` dan `Merchant City` dihapus) serta penerapan protokol *Strict Out-of-Sample Preprocessing*.

**EXP21B (Eksperimen Saat Ini)** melangkah lebih jauh sesuai rencana induk pada [07_eksplorasi_data_splitting_atribut_tuning_exp21.md](file:///d:/Projects/Klasifikasi%20Transasi%20Kartu%20Kredit%20-%20GraphSAGE/tugas-akhir-klasifikasi-transaksi-kartu-kredit/planning/07_eksplorasi_data_splitting_atribut_tuning_exp21.md) bagian **Eksperimen B (EXP21-B)**:
> **Tujuan EXP21B**: Menemukan titik temu optimal antara Recall tinggi dan Precision yang sehat melalui pencarian grid hiperparameter berimbang (*balanced hyperparameter tuning*), serta menguji dampak regularisasi pohon dalam membuka ruang bagi 128 fitur embedding GraphSAGE untuk berkontribusi secara nyata pada deteksi fraud.

---

## 2. Inovasi & Desain Metodologis EXP21B

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MATRIKS PENELITIAN EXP21B                         │
├──────────────────────┬──────────────────────────────────────────────────────┤
│ Dimensi Tuning       │ Nilai Eksplorasi Grid (Sesuai Dokumen Perencanaan)   │
├──────────────────────┼──────────────────────────────────────────────────────┤
│ 1. scale_pos_weight  │ [1.0, 10.0, 27.0 (sqrt), 50.0] + [Full 732.0 (Ablasi)│
│    (TUNE-1)          │ Menguji dampak redaman pembobotan kelas terhadap     │
│                      │ lonjakan False Positive dan stabilitas batas putus.  │
├──────────────────────┼──────────────────────────────────────────────────────┤
│ 2. max_depth         │ [3, 4, 5]                                            │
│    (TUNE-2)          │ Membatasi kedalaman pohon untuk mencegah pembuatan   │
│                      │ daun mikro individual (anti-memorization).           │
├──────────────────────┼──────────────────────────────────────────────────────┤
│ 3. min_child_weight  │ [5, 20, 50]                                          │
│    (TUNE-2)          │ Menuntut bobot sampel minimum per daun pohon.        │
├──────────────────────┼──────────────────────────────────────────────────────┤
│ 4. colsample_bytree  │ [0.6, 0.8, 1.0]                                      │
│    (TUNE-2)          │ Subsampling acak fitur per pohon untuk mencegah      │
│                      │ monopoli split oleh satu fitur tabular tunggal.      │
├──────────────────────┼──────────────────────────────────────────────────────┤
│ 5. Threshold Tuning  │ Kalibrasi dinamis tau* pada Validation Set holdout   │
│    (TUNE-4)          │ dengan konstrain presisi operasional perbankan.      │
└──────────────────────┴──────────────────────────────────────────────────────┘
```

---

## 3. Alur Eksekusi Pipeline (6 Phase)

Pipeline EXP21B dieksekusi secara otomatis melalui 6 tahapan yang saling terhubung:

### Phase 0: Strict Chronological Splitting & Out-of-Sample Preprocessing
* Memfilter transaksi era modern (Year $\ge 2016$).
* Memotong data secara strictly kronologis menjadi:
  - **Train**: 300 hari ($T_0$ = 1 Jan 2018 s/d 27 Okt 2018)
  - **Validation**: 30 hari (27 Okt 2018 s/d 26 Nov 2018)
  - **Embargo Gap**: 8 hari (26 Nov 2018 s/d 4 Des 2018 — standar SHINE)
  - **Test**: 60 hari masa depan (4 Des 2018 s/d 2 Feb 2019)
* Penskalaan `StandardScaler` dan kamus kategori **hanya di-fit pada data Train**. Nilai kategori baru pada Validation dan Test dipetakan ke token `<oov>` (index 0).
* Kolom `Zip` dan `Merchant City` dieksklusikan secara permanen. Ditambahkan 4 fitur waktu siklik (`hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`).

### Phase 1: Pelatihan GraphSAGE Encoder
* Model 2-Layer `HeteroGraphSAGE` (`hidden_channels = 128`, `aggr = 'mean'`).
* Entity scalers untuk User dan Merchant di-fit pada graf Train dan ditransformasikan pada graf Val/Test tanpa kebocoran masa depan.
* Pelatihan dengan Adam optimizer dan early stopping berbasis Validation AUCPR.

### Phase 2: Ekstraksi Representasi Induktif (Sekali untuk Seluruh Grid)
* Mengekstrak 128-dimensi embedding GNN dan menggabungkannya dengan fitur tabular bersih untuk Train, Val, dan Test.
* Proses ekstraksi dilakukan **hanya satu kali**, sehingga pencarian grid 108 kombinasi XGBoost dapat berjalan sangat cepat dan efisien tanpa melatih ulang graf.

### Phase 3: Systematic Balanced Hyperparameter Tuning
* Mengeksplorasi seluruh kombinasi grid $(w, d, m, c)$ pada data Train dan mengevaluasinya pada Validation Set.
* Pada setiap iterasi:
  - Model dilatih dengan parameter terkait.
  - Dicari threshold optimal $\tau^*$ yang memaksimalkan F1 pada Validation Set.
  - Dihitung metrik Validation: AUCPR, ROC-AUC, F1, Recall, Precision, G-Mean.
  - Dihitung porsi split gain keputusan untuk Fitur Tabular vs GraphSAGE.
* Seluruh jejak tuning dicatat ke `exp21b_{method}_tuning_results.csv` dan `.json`.
* Konfigurasi terbaik dipilih secara otomatis berbasis metrik validasi (`val_f1` / `val_auprc`).

### Phase 4: Pelatihan Model Terbaik & Evaluasi Test Set Masa Depan (60 Hari)
* Model dengan konfigurasi terbaik dilatih dan dievaluasi pada data Test masa depan yang belum pernah dilihat.
* Evaluasi mencakup dua sudut pandang:
  1. Ambang batas default $\tau = 0.50$ (baseline konvensional).
  2. Ambang batas optimal $\tau^*$ (hasil kalibrasi validasi holdout).
* Dihitung metrik operasional fraud perbankan:
  - **Lift @ 1% Transaksi Teratas**: Berapa kali lipat konsentrasi fraud di 1% alert paling mencurigakan dibandingkan baseline populasi.
  - **TP @ 500 Alert Harian**: Jumlah fraud tertangkap dalam 500 transaksi paling mencurigakan (simulasi kapasitas tim investigasi fraud harian bank).

### Phase 5: Studi Ablasi Komparatif Otomatis (Bahan Bab 4 Skripsi)
* Pipeline secara otomatis mengeksekusi dua studi ablasi kunci:
  1. **Ablasi Dampak Damping Class-Weight**: Menguji `scale_pos_weight` $[1.0, 10.0, 27.0, 50.0, 732.0]$ pada Test Set untuk membuktikan secara empiris kurva trade-off Recall vs Precision.
  2. **Ablasi Nilai Tambah Representasi Graf (Lift)**: Membandingkan performa model hilir pada:
     - Subset A: *Tabular Only* (hanya fitur transaksi bersih).
     - Subset B: *GraphSAGE Only* (hanya 128-d embedding struktural).
     - Subset C: *Hybrid Tuned* (Tabular + GraphSAGE).
* Hasil ablasi disimpan ke `exp21b_{method}_ablation_summary.json`.

### Phase 6: Audit Feature Importance & Generasi Visualisasi
* Mengaudit file booster XGBoost tersimpan dan memetakan kontribusi gain setiap fitur.
* Menghasilkan visualisasi ilmiah 300 DPI di folder `plots/`.

---

## 4. Manfaat Ilmiah untuk Laporan Tugas Akhir / Skripsi (Bab 4)

Penerapan EXP21B memberikan kontribusi substansial pada penulisan Bab 4:

1. **Bukti Solusi Terhadap Kecurigaan Pembimbing**:
   Menjawab tuntas pertanyaan *"Apakah model benar-benar belajar mendeteksi fraud, atau sekadar menghafal ZIP?"* dengan membuktikan performa model pada fitur bersih bebas ID kuasi.
2. **Tabel Sensitivitas Hiperparameter Berbobot**:
   Menyediakan data empiris mengenai pengaruh bobot kelas (`scale_pos_weight`) dan kedalaman pohon (`max_depth`) terhadap false alarm perbankan.
3. **Kuantifikasi Kontribusi Murni GraphSAGE (Lift)**:
   Membuktikan secara kuantitatif persentase peningkatan performa (*gain lift*) yang disumbangkan oleh GraphSAGE di atas baseline tabular bersih.
4. **Perbandingan Komprehensif 3 Metode Sampling**:
   Menyajikan perbandingan adil antara `uniform`, `topology`, dan `importance` sampling setelah classifier hilir di-tuning secara optimal.
