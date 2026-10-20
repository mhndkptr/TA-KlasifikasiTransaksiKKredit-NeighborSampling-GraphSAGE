# Rencana Eksplorasi & Remediasi EXP21: Data Splitting, Rekayasa Atribut, dan Hyperparameter Tuning

---

## 1. Ringkasan Eksekutif & Diagnosis Kejanggalan EXP21

### 1.1. Latar Belakang & Tanggapan Pembimbing
Pada eksperimen **EXP21 (v1 s/d v6)**, model Hybrid (GraphSAGE + XGBoost) menghasilkan metrik evaluasi pada Test Set 60 hari yang tampak "terlalu sempurna":
* **Recall**: **100.0%** (seluruh 421 kasus fraud tertangkap tanpa ada satu pun False Negative / missed fraud).
* **F1-Score**: **0.9397**
* **Precision**: **88.63%**
* **ROC-AUC**: **0.9999**

Pembimbing mencurigai adanya **kejanggalan (*too good to be true*)** karena dalam domain deteksi fraud kartu kredit dengan ketimpangan ekstrem (~0,12% fraud), hasil 100% recall pada data masa depan hampir mustahil dicapai secara murni tanpa adanya kebocoran data (*data leakage*), bias artifisial, atau model memorization.

---

### 1.2. Temuan Audit Internal (*The Smoking Gun*)

Audit langsung terhadap file model pohon keputusan XGBoost tersimpan (`model/exp21/v5/exp21_uniform_xgb.json`) membuktikan kebenaran kecurigaan pembimbing:

```text
========================================================================================
AUDIT POHON KEPUTUSAN XGBOOST (EXP21 v5)
========================================================================================
Peringkat Fitur Berdasarkan Split Gain:
  1. f4 (Zip)           : Split Count = 100 | Total Gain = 30.695.715,39 | Avg Gain = 306.957,15
  2. f2 (Merchant City) : Split Count = 200 | Total Gain =  9.251.440,98 | Avg Gain =  46.257,20
  3. f0 (Amount)        : Split Count = 563 | Total Gain =         72,05 | Avg Gain =       0,13
  4. f5 (MCC)           : Split Count =  16 | Total Gain =         38,28 | Avg Gain =       2,39
  5. 128 Fitur GNN      : Split Count <  30 | Total Gain =         24,46 | Avg Gain =       0,82
========================================================================================
Tree 0 Root Node Split:
  Node 0: Feature f4 (Zip) <= 22832.0000 | Gain = 2.170.656,20
========================================================================================
```

**Tiga Akar Masalah Utama yang Ditemukan:**
1. **Atribut Memorization & Pseudo-Ordinal Leakage**:
   - Kolom `Zip` (kardinalitas tinggi ~25.000+) dan `Merchant City` (~10.000+) diubah menggunakan `LabelEncoder` biasa menjadi bilangan bulat (0, 1, 2, ...).
   - XGBoost memperlakukan kode kategori ini sebagai besaran numerik kontinu (`Zip <= 22832`), sehingga pohon memotong dan menghafal rentang ZIP sintetis di mana transaksi fraud terkumpul.
   - Dua fitur ini menyumbang **>99.9% gain keputusan**, menenggelamkan kontribusi representasi graf GraphSAGE.
2. **Preprocessing Data Leakage & Single-Window Bias**:
   - `LabelEncoder` dan `StandardScaler` di-fit pada **seluruh baris data (Train + Val + Gap + Test)** sebelum data di-split.
   - Evaluasi hanya dilakukan pada **satu jendela waktu 60 hari** (5 Des 2018 s/d 3 Feb 2019), sehingga tidak mencerminkan generalisasi di horizon waktu lain.
3. **Hyperparameter Tuning yang Terlalu Agresif**:
   - `scale_pos_weight = 732.18` (rasio penuh perbandingan kelas mayoritas/minoritas) memaksa pohon membuat daun-daun mikro yang sangat sensitif demi menghindari penalti False Negative.
   - Ambang batas optimal ($\tau^*$) dicari hingga ke angka ekstrem `0.01` (`tau = 0.01`).

---

## 2. Matriks Opsi Eksplorasi (Sesuai 3 Arahan Pembimbing)

Berikut opsi-opsi terstruktur yang dapat dipilih dan dipadukan:

### Pilar I: Eksplorasi Pemrosesan Data & Data Splitting

| Kode Opsi | Nama Opsi | Deskripsi Metodologis | Manfaat Ilmiah |
| :--- | :--- | :--- | :--- |
| **SPLIT-1** | **Strict Out-of-Sample Preprocessing (Anti-Leakage Guard)** | Memisahkan tahap *fit* dan *transform*: `StandardScaler`, encoder kategori, dan penghitungan statistik hanya boleh di-`fit` pada data **Train Window**. Data Val dan Test murni di-`transform`. Kategori baru yang belum pernah muncul di-assign ke `<OOV>` (*Out of Vocabulary*). | Mengeliminasi kebocoran informasi masa depan ke masa latih; sesuai standar industri & paper top-tier. |
| **SPLIT-2** | **Multi-Horizon Rolling Window (Walk-Forward Validation)** | Menguji model tidak hanya pada 1 jendela (60 hari), melainkan pada **3 s/d 4 jendela bergulir berurutan** di era 2016–2020 (misal: Window 1: 2017, Window 2: 2018, Window 3: 2019) dengan durasi tetap (Train 300d, Val 30d, Gap 8d, Test 60d). | Membuktikan stabilitas performa terhadap *concept drift* dan membuktikan model tidak hanya "beruntung" di satu periode. |
| **SPLIT-3** | **Ablasi Durasi Embargo Gap** | Menguji pengaruh variasi jeda investigasi fraud (*embargo gap*): 0 hari (tanpa gap), 8 hari (standar SHINE), dan 30 hari (siklus penagihan kartu kredit/chargeback bulanan). | Memberikan wawasan empiris mengenai *reporting delay* transaksi fraud kartu kredit di dunia nyata. |
| **SPLIT-4** | **Strictly Past Graph Edge Masking** | Saat mengekstrak fitur Test pada graf gabungan, edge transaksi hanya boleh menghubungkan transaksi historis sebelum waktu transaksi uji ($t_j < t_i$), mencegah transaksi masa depan mempengaruhi representasi entitas User/Merchant. | Menjamin graf uji bersifat induktif murni secara temporal. |

---

### Pilar II: Eksplorasi Penggunaan Atribut Dataset

| Kode Opsi | Nama Opsi | Deskripsi Metodologis | Manfaat Ilmiah |
| :--- | :--- | :--- | :--- |
| **ATTR-1** | **Eliminasi Fitur Quasi-Identifier (Leakage Prone)** | Menghapus atribut ber-kardinalitas sangat tinggi yang menjadi celah hafalan pohon: hapus `Zip` dan `Merchant City` dari input tabular transaksi. | Menguji apakah model tetap mampu mengenali fraud berdasarkan pola transaksi murni tanpa menghafal kode pos/kota. |
| **ATTR-2** | **Kategorikal Robust: Frequency & Target Encoding dengan Smoothing** | Untuk atribut nominal ber-kardinalitas menengah (seperti `MCC`, `Merchant State`, `Use Chip`): ganti LabelEncoder dengan **Target Encoding ter-regularisasi m-estimate** (dihitung strictly pada Train set) atau **Frequency/Count Encoding**. | Mengubah ID kategorikal menjadi representasi risiko nyata tanpa membuat hubungan ordinal palsu. |
| **ATTR-3** | **Rekayasa Fitur Perilaku Temporal (*Behavioral Velocity*)** | Menambahkan fitur transaksi dinamis strictly-past: <br>1. *Cyclical Time*: $\sin/\cos$ jam (0–23) dan hari dalam seminggu (0–6). <br>2. *Velocity*: Jumlah transaksi user dalam 1 jam terakhir dan 24 jam terakhir. <br>3. *Amount Discrepancy*: Deviasi nominal transaksi terhadap rata-rata historis kartu user ($Z = \frac{x - \mu}{\sigma}$). | Memberikan sinyal fraud yang logis secara bisnis dan berbasis anomali perilaku, bukan ID statis. |
| **ATTR-4** | **Ablasi Komparatif 3 Konfigurasi Fitur** | Membandingkan performa model hilir pada 3 subset input: <br>A. *Only Graph Embeddings* (128-d GNN murni). <br>B. *Only Clean Tabular* (fitur transaksi tanpa ZIP/City). <br>C. *Hybrid* (GNN Embeddings + Clean Tabular). | Menjawab pertanyaan inti skripsi: *"Berapa besar kontribusi nilai tambah (lift) yang disumbangkan oleh GraphSAGE di atas baseline tabular?"* |

---

### Pilar III: Eksplorasi Hyperparameter Tuning

| Kode Opsi | Nama Opsi | Deskripsi Metodologis | Manfaat Ilmiah |
| :--- | :--- | :--- | :--- |
| **TUNE-1** | **Damping Class-Weight XGBoost (`scale_pos_weight`)** | Menguji variasi `scale_pos_weight`: <br>1. `1.0` (tanpa pembobotan). <br>2. `sqrt(neg/pos)` ($\approx 27.0$). <br>3. `10.0` s/d `50.0` (moderat teredam). <br>4. `neg/pos` ($\approx 732.0$ — baseline EXP21). | Mencegah pohon memicu *false alarm* berlebihan dan melatih batas keputusan yang lebih stabil. |
| **TUNE-2** | **Regularisasi Pohon Anti-Memorization** | Membatasi kompleksitas XGBoost: <br>- `max_depth`: 3 s/d 4 (pohon dangkal). <br>- `min_child_weight`: 10 s/d 50 (daun harus memiliki minimal bobot sampel signifikan, mencegah hafalan transaksi individual). <br>- `colsample_bytree`: 0.6 s/d 0.8 (mencegah dominasi satu fitur tunggal). <br>- `subsample`: 0.8. | Memaksa ensemble memanfaatkan berbagai fitur secara seimbang termasuk embedding GNN. |
| **TUNE-3** | **Hyperparameter GraphSAGE & Neighbor Sampling** | Tuning pada sisi representasi graf: <br>- *Sampling Fanout*: $[10, 5]$ vs $[15, 10]$ vs $[25, 10]$. <br>- *Hidden Dimension*: 64 vs 128 vs 256. <br>- *GNN Loss Strategy*: Sqrt Pos-Weight vs Focal Loss ($\gamma = 2.0$). | Mengoptimalkan kualitas embedding sebelum diekstraksi ke classifier hilir. |
| **TUNE-4** | **Protokol Kalibrasi Threshold Dinamis Terisolasi** | Pencarian threshold optimal $\tau^*$ dilakukan pada data validasi dengan metrik terarah (misal: memaksimumkan F1 dengan konstrain Precision minimal $\ge 50\%$, bukan mencari $\tau$ serendah mungkin hingga 0.01). | Menghasilkan ambang batas operasional yang realistis untuk sistem perbankan. |

---

## 3. Desain Rencana Eksperimen Terstruktur (Planning Roadmap)

Untuk menjawab saran pembimbing secara komprehensif tanpa membebani komputasi berlebihan, kami merancang **3 Fase Kerja**:

```mermaid
flowchart TD
    subgraph Fase 1: Remediasi & Audit Kebocoran Data
        F1A["Audit Fitur & Eliminasi ZIP/City Memorization (ATTR-1)"]
        F1B["Strict Train-Only Preprocessing Fit (SPLIT-1)"]
        F1C["Baseline Sanity Check: Tabular vs GNN vs Hybrid"]
    end

    subgraph Fase 2: Eksplorasi Hyperparameter Tuning
        F2A["Tuning Regularisasi XGBoost (TUNE-1 & TUNE-2)"]
        F2B["Tuning Kapasitas GraphSAGE & Sampling Size (TUNE-3)"]
        F2C["Kalibrasi Threshold Berimbang (TUNE-4)"]
    end

    subgraph Fase 3: Eksplorasi Data Splitting & Evaluasi Robustness
        F3A["Multi-Window Rolling Walk-Forward 3-Folds (SPLIT-2)"]
        F3B["Analisis Variasi Embargo Gap (SPLIT-3)"]
        F3C["Penyusunan Tabel Komparasi Bab 4 Skripsi"]
    end

    Fase 1 --> Fase 2 --> Fase 3
```

---

### Detail Matriks Eksperimen yang Direncanakan

#### **Eksperimen A: Audit & Eliminasi Kebocoran Atribut (EXP21-A)**
* **Tujuan**: Menghilangkan efek hafalan `Zip` dan `Merchant City` serta memverifikasi performa riil.
* **Konfigurasi Variabel**:
  - `Set-1 (Asli EXP21)`: Amount + 6 Kategori (termasuk Zip & City) + GNN.
  - `Set-2 (No High-Cardinality)`: Amount + Use Chip + MCC + Errors (Hapus Zip & City) + GNN.
  - `Set-3 (Feature Engineered)`: Set-2 + Sin/Cos Time + Amount Discrepancy + GNN.
* **Output yang Diharapkan**: Metrik F1 dan Recall turun ke rentang yang wajar dan realistis ($0.55 - 0.78$), membuktikan model mengevaluasi sinyal generalisasi.

#### **Eksperimen B: Hyperparameter Tuning Berimbang (EXP21-B)**
* **Tujuan**: Menemukan titik temu optimal antara Recall tinggi dan Precision yang sehat.
* **Grid Parameter**:
  - `scale_pos_weight`: $[1.0, 10.0, 27.0 (\text{sqrt}), 50.0]$
  - `max_depth`: $[3, 4, 5]$
  - `min_child_weight`: $[5, 20, 50]$
  - `colsample_bytree`: $[0.6, 0.8, 1.0]$
* **Output yang Diharapkan**: Identifikasi konfigurasi hiperparameter yang tidak overfitting dan memberikan ruang bagi 128 fitur embedding GraphSAGE untuk berkontribusi.

#### **Eksperimen C: Multi-Window Temporal Splitting (EXP21-C)**
* **Tujuan**: Membuktikan ketahanan model pada beberapa jendela waktu di era modern 2016–2020.
* **Protokol**:
  - **Window 1 (2017–2018)**: Train Jan 2017 – Okt 2017, Test Nov 2017 – Jan 2018.
  - **Window 2 (2018–2019)**: Train Jan 2018 – Okt 2018, Test Des 2018 – Feb 2019 (Horizon EXP21 saat ini).
  - **Window 3 (2019–2020)**: Train Jan 2019 – Okt 2019, Test Des 2019 – Feb 2020.
* **Output yang Diharapkan**: Nilai rata-rata dan standar deviasi ($\mu \pm \sigma$) dari metrik F1, AUCPR, dan Recall lintas waktu.

---

## 4. Keuntungan Ilmiah untuk Laporan Tugas Akhir / Skripsi

1. **Integritas Akademik yang Sangat Kuat**:
   - Menunjukkan kejujuran ilmiah (*scientific rigor*) bahwa peneliti menyadari hasil awal tidak wajar, mampu menemukan akar masalah teknisnya (memorization pada `Zip` & `City`), dan memperbaikinya.
2. **Narasi Bab 4 yang Jauh Lebih Berbobot**:
   - Bukan sekadar menampilkan skor tinggi, tetapi menampilkan **Studi Investigasi & Ablasi Fitur**:
     - *Dampak Penggunaan Atribut Quasi-Identifier vs Atribut Terkurasi*.
     - *Perbandingan Single-Window vs Multi-Window Rolling Split*.
     - *Kontribusi Murni GraphSAGE terhadap Fitur Tabular*.
3. **Komparasi Sejajar dengan Literatur Terkemuka**:
   - Hasil evaluasi setelah pembersihan akan dapat dibandingkan secara adil dan langsung dengan benchmark paper internasional seperti **SHINE (IEEE TKDE)** dan **TabFormer (IBM Research)**.
