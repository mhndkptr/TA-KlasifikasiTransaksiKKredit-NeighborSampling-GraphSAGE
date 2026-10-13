# Penjelasan Pipeline Eksperimen 21 (EXP21)
## *Temporal Window Alignment & Inductive Setup (SHINE FRAUD II Protocol)*

---

## 1. Latar Belakang & Rasional Ilmiah

Pada eksperimen pendahulu (`exp16`–`exp19`), dataset transaksi dibagi menggunakan rasio global 70/15/15 yang mencakup rentang waktu **2 Januari 1991 hingga 28 Februari 2020 (hampir 30 tahun)**. Berdasarkan audit mendalam pada `DATASET_KNOWLEDGE_BASE.md`:
1. **Disparitas Kanal & Pergeseran Konsep (*Concept Drift*)**: Transaksi tahun 1991–2015 didominasi oleh kanal *Swipe* dan *Online*, sedangkan periode uji (2018–2020) didominasi oleh kanal *Chip*. Memaksa model mempelajari histori 24 tahun lalu untuk memprediksi masa depan menyuntikkan noise historis yang masif.
2. **Beban Komputasi Ekstrem (*Timeout* Kaggle)**: Melatih graf pada 17 juta transaksi dan mengekstrak jutaan embedding 2-hop menyebabkan kehabisan alokasi waktu eksekusi (seperti pada exp19 yang terhenti pada detik ke-8.166).

Eksperimen 21 (`EXP21`) menyelaraskan protokol penelitian dengan standar publikasi internasional paper **SHINE (IEEE TKDE 2024, Rafaël Van Belle & Jochen De Weerdt)** pada dataset **FRAUD II**:
- Mengisolasi transaksi ke era modern **2016 s/d 2020**.
- Menerapkan arsitektur **Rolling Window**:
  - **Train Window**: 300 hari transaksi.
  - **Validation Window**: 30 hari transaksi.
  - **Embargo Gap**: 8 hari transaksi (jeda operasional klaim perbankan nyata).
  - **Test Window**: 60 hari transaksi.

---

## 2. Analisis Audit Temporal 2016–2020 & Penentuan Tanggal Referensi ($T_0$)

Berdasarkan pemindaian menyeluruh terhadap baris data era 2016–2020, diperoleh distribusi fraud bulanan:
- **Tahun 2016**: 1.708.924 transaksi, 3.579 fraud (stabil ~174–459 fraud/bulan).
- **Tahun 2017**: Periode Januari–Oktober 2017 memiliki **0 transaksi fraud berlabel** (artefak generator sintetis TabFormer). Fraud baru muncul kembali pada November–Desember 2017 (255 fraud).
- **Tahun 2018**: 1.721.615 transaksi, 2.491 fraud (sangat stabil ~145–300 fraud/bulan).
- **Tahun 2019**: 1.723.938 transaksi, 2.087 fraud (stabil ~142–275 fraud/bulan hingga Oktober).

**Implikasi Penentuan $T_0$:**
- Jika $T_0$ dipatok pada `2017-01-01`, data latih 300 hari (Januari–Oktober 2017) akan memiliki **0 fraud**, yang menyebabkan model biner kolaps karena ketiadaan kelas positif.
- Oleh karena itu, konfigurasi bawaan `EXP21` menetapkan $T_0 = \text{"2018-01-01"}$ (atau alternatif `2016-01-01`):
  - **Train (300 hari)**: 1 Jan 2018 – 28 Okt 2018 ($\approx 1.400.000$ transaksi, $\approx 1.800$ fraud).
  - **Validation (30 hari)**: 28 Okt 2018 – 27 Nov 2018 ($\approx 140.000$ transaksi, $\approx 250$ fraud).
  - **Embargo Gap (8 hari)**: 27 Nov 2018 – 5 Des 2018 (dilewati dari pelatihan & evaluasi).
  - **Test (60 hari)**: 5 Des 2018 – 3 Feb 2019 ($\approx 280.000$ transaksi, $\approx 400$ fraud).

---

## 3. Tahapan Alur Eksekusi (4 Phases)

Pipeline dikoordinasikan dalam skrip `main.py` melalui 4 tahapan berurutan:

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Data Preprocessing (Filter Year >= 2016, Label Encoding) │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Temporal Window Splitting (Train 300d, Val 30d, Test 60d)│
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Phase 1: Train HeteroGraphSAGE (Mean Aggr, 2-Layer, 10 Ep)│
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Phase 2: Feature Extraction (Zero-Memory Overhead Numpy) │
│    - Train Embeddings (Train Graph)                         │
│    - Val Embeddings (Train + Val Graph, Anti-Leakage)       │
│    - Test Embeddings (Train + Val + Test Graph)             │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Phase 3: Train XGBoost Classifier (scale_pos_weight fit) │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. Phase 4: Dynamic Threshold Calibration & Evaluation      │
│    - Cari tau* pada Validation Set                          │
│    - Evaluasi Test @ 0.5 vs @ tau*                          │
│    - Hitung AUCPR, ROC-AUC, Lift@1%, TP@500                 │
└─────────────────────────────────────────────────────────────┘
```

### Phase 1: Pelatihan GraphSAGE Encoder
- **Arsitektur**: 2-layer `HeteroGraphSAGE`, `hidden_channels = 128`, agregator `mean`.
- **Node Features**: Fitur tabular transaksi + fitur skalar User/Merchant (`avg_amount` dan `tx_count` yang distandarisasi).
- **Loss Function**: `manual_weighted_bce` dengan denda penalti kelas minoritas dinamis (`pos_weight`).

### Phase 2: Ekstraksi Fitur & Pencegahan Temporal Leakage
- Embeddings transaksi berdimensi 128 digabungkan (*concatenated*) dengan fitur tabular orisinal.
- **Strict Anti-Leakage Protocol**:
  - Ekstraksi fitur Train hanya melibatkan transaksi Train.
  - Ekstraksi fitur Validation hanya melibatkan graf gabungan Train + Val (tanpa menyentuh Test).
  - Ekstraksi fitur Test melibatkan graf historis hingga Test.

### Phase 3: Klasifikasi XGBoost
- `n_estimators = 100`, `max_depth = 5`, `learning_rate = 0.05`.
- `scale_pos_weight` dihitung otomatis dari rasio kelas pada data latih.

### Phase 4: Kalibrasi Threshold Dinamis ($\tau^*$)
- **Pencarian Ambang Batas Optimal**: Menguji $\tau \in [0.01, 0.50]$ pada subset validasi untuk mencari $\tau^* = \arg\max \text{F1}_{val}(\tau)$.
- **Penerapan Out-of-Sample**: Nilai $\tau^*$ diterapkan langsung pada data uji untuk membuktikan bahwa performa F1 melonjak tanpa kebocoran data.

---

## 4. Metrik Evaluasi yang Dilaporkan

1. **Threshold-Dependent Metrics (@ 0.5 vs @ $\tau^*$)**:
   - F1-Score (Binary & Macro)
   - Recall & Precision
   - Accuracy & G-Mean
2. **Threshold-Independent Metrics**:
   - **AUCPR (Area Under Precision-Recall Curve)**: Metrik emas data tidak seimbang.
   - **ROC-AUC**
3. **Operational Banking Metrics**:
   - **Lift@1%**: Peningkatan deteksi fraud pada 1% transaksi paling berisiko.
   - **TP@500**: Jumlah kasus penipuan riil yang berhasil diamankan dari 500 alert harian bank.
4. **Efisiensi Komputasi**:
   - Waktu pelatihan GNN, ekstraksi fitur, pelatihan XGBoost, dan total eksekusi pipeline.
