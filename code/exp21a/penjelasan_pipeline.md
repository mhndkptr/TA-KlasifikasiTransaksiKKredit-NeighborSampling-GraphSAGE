# Penjelasan Pipeline Eksperimen 21A (EXP21A)
## *Audit Kebocoran Atribut, Strict Anti-Leakage Preprocessing, & Regularized Classifier*

---

## 1. Latar Belakang & Rasional Ilmiah

Pada eksperimen **EXP21**, pipeline Hybrid GraphSAGE + XGBoost menghasilkan metrik evaluasi pada Test Set masa depan (60 hari) yang tampak "terlalu sempurna":
* Recall: **100.0%** (seluruh 421 fraud tertangkap sempurna tanpa False Negative)
* Precision: **88.63%**
* F1-Score: **0.9397**
* ROC-AUC: **0.9999**

Hasil ini memicu kecurigaan dari pembimbing karena pada dataset riil deteksi kecurangan dengan ketimpangan kelas ekstrem (~0,12% fraud), recall 100% pada data masa depan hampir mustahil dicapai tanpa bias atau kebocoran data (*data leakage*).

### Bukti Audit Internal (*The Smoking Gun*):
Pemeriksaan struktur pohon keputusan tersimpan pada `model/exp21/v5/exp21_uniform_xgb.json` menemukan bahwa:
1. **Fitur Quasi-Identifier Dihafal Pohon**:
   - Fitur `f4` (`Zip`) memiliki total split gain **30.695.715,39**.
   - Fitur `f2` (`Merchant City`) memiliki total split gain **9.251.440,98**.
   - Keduanya menyumbang **>99.9% gain keputusan XGBoost**, sedangkan 128 fitur embedding GraphSAGE hanya mendapatkan gain kecil (<25.0).
   - XGBoost memperlakukan hasil integer `LabelEncoder` dari kode ZIP dan Nama Kota sebagai angka kontinu (`Zip <= 22832`), sehingga pohon memotong dan menghafal klaster ZIP sintetis tempat data fraud IBM TabFormer terkumpul.
2. **Kebocoran Preprocessing**:
   - `LabelEncoder` dan `StandardScaler` di-`fit` pada **seluruh baris data** sebelum pembagian data (*data splitting*).
3. **Pembobotan Ekstrem & Ambang Batas Terlalu Agresif**:
   - `scale_pos_weight = 732` (rasio penuh) memaksa pohon membuat daun-daun mikro yang sangat sensitif demi menghindari False Negative.

---

## 2. Inovasi & Perbaikan Metodologis pada EXP21A

Eksperimen **EXP21A** dirancang secara khusus untuk mengatasi kelemahan di atas melalui 4 pilar remediasi:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             PERBANDINGAN ARSITEKTUR                         │
├──────────────────────────────────────┬──────────────────────────────────────┤
│              EXP21 (Lama)            │            EXP21A (Baru)             │
├──────────────────────────────────────┼──────────────────────────────────────┤
│ 1. Preprocessing fit pada seluruh    │ 1. Preprocessing fit HANYA pada data │
│    dataset (Train + Val + Test).     │    Train Window (Strict Out-of-Sample│
│                                      │    Protocol). Val & Test murni di-   │
│                                      │    transform; kategori baru -> <oov>.│
│                                      │                                      │
│ 2. Memasukkan 'Zip' (~25k kategori)  │ 2. 'Zip' dan 'Merchant City'         │
│    dan 'Merchant City' (~10k) ke     │    DIHAPUS SECARA TOTAL dari input   │
│    pohon keputusan (Hafalan ID).     │    tabular (Eliminasi Memorization). │
│                                      │                                      │
│ 3. Fitur tabular: Amount + 6 kolom   │ 3. Fitur bersih: Amount, Use Chip,   │
│    mentah kategori arbitrer.         │    MCC, Errors?, State + 4 Fitur     │
│                                      │    Waktu Siklik (sin/cos jam & hari).│
│                                      │                                      │
│ 4. XGBoost scale_pos_weight = 732    │ 4. XGBoost scale_pos_weight = ~27    │
│    (rasio penuh), max_depth=5,       │    (sqrt damped), max_depth=4,       │
│    min_child_weight=1 (unregularized)│    min_child_weight=10, colsample=0.8│
│                                      │                                      │
│ 5. Feature importance GNN tertutup   │ 5. Audit otomatis Split Gain: Tabular│
│    oleh dominasi gain Zip (30 juta). │    vs GraphSAGE dihitung transparan. │
└──────────────────────────────────────┴──────────────────────────────────────┘
```

---

## 3. Alur Eksekusi Pipeline (4 Phases)

1. **Phase 0: Strict Chronological Splitting & Out-of-Sample Preprocessing**
   - Data era 2016–2020 dipotong ke 4 jendela: Train (300d), Val (30d), Embargo Gap (8d), Test (60d).
   - `StandardScaler` Amount di-`fit` HANYA di Train.
   - Kamus nilai unik kategorikal di-`fit` HANYA di Train; nilai baru pada Val/Test di-mapping ke index 0 (`<oov>`).
   - Fitur waktu siklik (`hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`) ditambahkan.

2. **Phase 1: Pelatihan GraphSAGE Encoder**
   - 2-Layer HeteroGraphSAGE (`hidden_channels = 128`, `aggr = 'mean'`).
   - Entity scalers untuk User dan Merchant di-fit pada graf Train dan ditransformasikan pada graf Val/Test.
   - Checkpoint bobot terbaik disimpan berbasis Validation AUCPR.

3. **Phase 2: Ekstraksi Representasi Induktif**
   - Mengekstrak embedding GNN 128-dimensi digabung dengan fitur tabular bersih (`train_features`, `val_features`, `test_features`).

4. **Phase 3: Pelatihan XGBoost Ter-regularisasi & Audit Feature Importance**
   - Pelatihan dengan parameter anti-hafalan: `max_depth = 4`, `min_child_weight = 10`, `subsample = 0.8`, `colsample_bytree = 0.8`, `scale_pos_weight = sqrt(neg/pos)`.
   - Menghitung akumulasi split gain untuk fitur Tabular vs fitur GNN dan menyimpannya ke `exp21a_{method}_feature_importance.json`.

5. **Phase 4: Kalibrasi Threshold Dinamis & Evaluasi Final**
   - Menentukan threshold optimal $\tau^*$ dari kurva precision-recall validasi.
   - Mengevaluasi performa Test Set pada $\tau = 0.50$ dan $\tau = \tau^*$.
   - Menghitung metrik independen threshold: AUCPR, ROC-AUC, Lift@1%, dan TP@500.

---

## 4. Ekspektasi Hasil Ilmiah

Dengan dihilangkannya celah hafalan `Zip` dan `Merchant City`:
1. Metrik evaluasi akan bergeser dari angka artifisial 100% Recall ke angka performa **ilmiah yang riil dan wajar** (estimasi Recall: $55\% - 80\%$, F1-score: $0.50 - 0.75$, AUCPR: $0.45 - 0.70$).
2. Bobot keputusan XGBoost tidak lagi didominasi 99.9% oleh satu kolom ZIP, melainkan terdistribusi seimbang antara sinyal nominal/kanal transaksi dan **representasi struktural graf GraphSAGE**.
3. Hasil ini memberikan landasan yang sangat solid, kredibel, dan berintegritas tinggi untuk Bab 4 Tugas Akhir.
