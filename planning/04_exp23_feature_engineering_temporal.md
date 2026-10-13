# Rencana Eksperimen 23 (EXP23): Rekayasa Fitur Perilaku Temporal & Pengkodean Kategorikal yang Tepat

## 1. Identitas Eksperimen
* **ID Eksperimen**: `EXP23`
* **Nama Rencana**: *Behavioral & Temporal Velocity Feature Engineering with Robust Categorical Encoding*
* **Tipe Eksperimen**: Feature Engineering & Representation Optimization
* **Target Utama**: Memperbaiki representasi fitur masukan (*input node features*) pada transaksi dengan menambahkan fitur perilaku temporal (*strictly-past*) dan mengganti pengkodean integer sembarang (*ordinal fallacy*) pada fitur kategorikal.

---

## 2. Latar Belakang & Motivasi
Pemeriksaan pada pipeline `exp18c` (`data_preprocessing.py`) menemukan dua kelemahan mendasar pada fitur node transaksi:
1. **Distorsi Semantik Kategorikal (*Ordinal Fallacy*)**:
   Fitur kategorikal dengan kardinalitas tinggi (`Zip`, `MCC`, `Merchant City`) diproses dengan `LabelEncoder` menjadi integer ($0, 1, 2, \dots$) lalu langsung dimasukkan ke tensor float pada lapisan Linear GNN:
   $$z = W \cdot \text{Zip}_{\text{int}} + b$$
   Hal ini memaksakan asumsi ordinal palsu pada jaringan syaraf (misal menganggap $\text{Zip } 91750$ bernilai lebih besar dari $\text{Zip } 10000$, atau $\text{MCC } 5411$ berada di antara $\text{MCC } 5410$ dan $5412$).
2. **Ketiadaan Fitur Kecepatan & Rasio Transaksi (*Velocity & Behavioral Context*)**:
   Model hanya menerima 7 atribut fisik mentah tanpa memperhitungkan histori transaksi nasabah sebelumnya. Pada deteksi fraud perbankan, prediktor terkuat adalah **fitur deviasi perilaku** (misal: penarikan dana beruntun dalam hitungan menit, atau nominal transaksi yang melompat 10x lipat dari rata-rata histori pengguna).
3. **Temuan Paper SHINE**:
   Pada Tabel VII paper SHINE, model **XGBoost tabular murni** mampu mencapai F1 **0.6612** dan AUCPR **0.6267** bahkan tanpa informasi graf. Ini membuktikan bahwa fitur tabular dasar yang mereka gunakan memiliki daya prediksi (*predictive power*) yang kaya. Jika fitur tabular terlalu lemah, performa GNN maupun XGBoost akan terhambat di batas bawah.

---

## 3. Hipotesis Penelitian
1. Menghilangkan asumsi ordinal pada fitur kategorikal (menggunakan *Frequency Encoding* atau *Entity Embeddings*) akan mencegah bias linear palsu pada representasi GNN.
2. Penambahan fitur kecepatan temporal (*temporal velocity*) dan rasio nominal histori akan meningkatkan AUCPR dan F1-score baik pada GNN maupun XGBoost secara signifikan.
3. Kualitas representasi node awal yang lebih kuat akan menghasilkan *message passing* yang lebih kaya makna bagi agregator GraphSAGE.

---

## 4. Desain & Spesifikasi Fitur Baru (Strictly-Past, No Leakage)

Fitur dibangun secara kumulatif **hanya dari transaksi sebelum waktu transaksi saat ini** ($[0, t_i)$):

### A. Fitur Temporal Velocity & Recency
1. `time_diff_last_card_tx`: Selisih waktu (dalam detik/menit) dari transaksi sebelumnya pada kartu yang sama ($\Delta t_{\text{card}}$).
2. `time_diff_last_user_tx`: Selisih waktu dari transaksi sebelumnya oleh nasabah yang sama ($\Delta t_{\text{user}}$).
3. `tx_count_last_1h`: Jumlah transaksi kartu dalam jendela 1 jam terakhir.
4. `tx_count_last_24h`: Jumlah transaksi kartu dalam jendela 24 jam terakhir.

### B. Fitur Rasio & Deviasi Finansial
5. `amount_to_user_avg_ratio`: Rasio nominal saat ini terhadap rata-rata nominal historis nasabah ($Amount / \mu_{user\_amt}$).
6. `amount_diff_user_avg`: Selisih mutlak nominal terhadap rata-rata historis ($|Amount - \mu_{user\_amt}|$).
7. `amount_to_merchant_avg_ratio`: Rasio nominal terhadap rata-rata nominal pedagang ($Amount / \mu_{merchant\_amt}$).

### C. Pengkodean Kategorikal yang Kuat
8. **Frequency Encoding**: Untuk `MCC`, `Zip`, dan `Merchant City`, ganti integer acak dengan probabilitas frekuensi kemunculannya pada data latih (merefleksikan popularitas lokasi/kategori).
9. **One-Hot Encoding**: Untuk kanal transaksi (`Use Chip`: Swipe, Chip, Online) dan indikator status error (`Errors?`).

---

## 5. Rencana File & Struktur Output
* Folder Skrip: `code/exp23/`
* Modifikasi File:
  * `feature_engineering.py`: Modul khusus pembuat fitur *strictly-past* efisien menggunakan operasi vektor Polars/Pandas.
  * `data_preprocessing.py`: Integrasi fitur baru ke dalam pipeline data.
  * `config.kaggle.yaml`: Penyesuaian `feature_dims['transaction']`.
* Output Hasil:
  * `result/exp23/feature_importance_xgb.json` (Analisis fitur paling berkontribusi pada XGBoost)
  * `result/exp23/exp23_enriched_features_results.json`
  * `result/exp23/README.md`

---

## 6. Kriteria Keberhasilan (Success Criteria)
1. Ekstraksi fitur bersifat *strictly-past* (terverifikasi bebas dari kebocoran data masa depan).
2. XGBoost baseline (tanpa GNN) menunjukkan lonjakan AUCPR $> 0.50$, membuktikan fitur baru membawa sinyal fraud yang nyata.
3. Model hibrida (GNN + XGBoost) mengalami peningkatan F1 dan AUCPR dibandingkan EXP22.

---

## 7. Kontribusi untuk Laporan Tugas Akhir
* Membahas rekayasa fitur domain kartu kredit (*domain-specific feature engineering*) yang memenuhi kaidah anti-*data leakage* pada pemodelan deret waktu finansial.
* Menjelaskan pentingnya representasi numerik yang tepat untuk variabel diskrit berkardinalitas tinggi pada arsitektur GNN.
