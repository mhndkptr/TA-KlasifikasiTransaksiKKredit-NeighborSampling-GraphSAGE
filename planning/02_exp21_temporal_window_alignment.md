# Rencana Eksperimen 21 (EXP21): Penyelarasan Protokol Waktu & Window Induktif Modern (2016–2020)

## 1. Identitas Eksperimen
* **ID Eksperimen**: `EXP21`
* **Nama Rencana**: *Temporal Window Alignment & Concept Drift Elimination (SHINE FRAUD II Protocol)*
* **Tipe Eksperimen**: Perombakan Protokol Partisi Data & Lingkungan Induktif
* **Target Utama**: Menyelaraskan protokol split waktu dengan metodologi paper SHINE untuk dataset FRAUD II (era modern 2016–2020), mengeliminasi *30-year concept drift*, dan mempercepat waktu komputasi agar bebas dari risiko *timeout* Kaggle.

---

## 2. Latar Belakang & Motivasi
Pada eksperimen exp16–exp19, data dibagi dengan rasio 70/15/15 mencakup seluruh transaksi dari tahun **1991 hingga 2020 (hampir 30 tahun)**. Berdasarkan audit pada `DATASET_KNOWLEDGE_BASE.md`:
1. Pola fraud tahun 1991–2015 didominasi transaksi *Swipe* dan *Online*, sedangkan periode uji 2018–2020 didominasi fraud berbasis teknologi *Chip*.
2. Mempelajari histori 24 tahun lalu untuk memprediksi transaksi masa depan menghasilkan *concept drift* yang masif dan menyuntikkan noise dari jutaan transaksi usang.
3. Memproses 17 juta transaksi latih dan 3,65 juta transaksi uji membebani memori graf secara ekstrem dan memicu *timeout* eksekusi (seperti pada exp19).

Sebaliknya, paper SHINE (Bagian IV-C1 & Catatan Kaki 6) secara eksplisit mendefinisikan dataset **FRAUD II** hanya pada transaksi era modern **2016 s/d 2020** (1.959 pengguna kartu dan 56.545 pedagang) dengan protokol **Rolling Window**:
* **Training Window**: 300 hari transaksi
* **Validation Window**: 30 hari transaksi
* **Embargo Gap**: 8 hari transaksi (jeda verifikasi investigasi klaim)
* **Test Window**: 60 hari transaksi

Dengan protokol ini, volume transaksi per jendela waktu terkontrol, bebas dari pergeseran teknologi puluhan tahun, dan mendekati realitas operasional fraud detection modern.

---

## 3. Hipotesis Penelitian
1. Membatasi horizon data ke era 2016–2020 dan menggunakan jendela waktu 300 hari (train) vs 60 hari (test) akan mengeliminasi disparitas kanal (*concept drift* chip vs swipe).
2. Performa baseline GraphSAGE dan XGBoost akan melonjak mendekati hasil Tabel VII paper SHINE (F1 $\approx 0.50 - 0.68$, AUCPR $\approx 0.50 - 0.67$).
3. Waktu komputasi fase pelatihan GNN dan ekstraksi fitur akan berkurang drastis (di bawah 30 menit), mengeliminasi risiko *out-of-memory* (OOM) dan *timeout* di Kaggle.

---

## 4. Desain & Spesifikasi Teknis

### A. Filter Dataset & Pembentukan Window Waktu
* **Sumber Data**: `credit_card_transactions-ibm_v2.csv`
* **Filter Waktu**: $\text{Year} \ge 2016$ (mengambil data mulai 1 Januari 2016 sampai 28 Februari 2020).
* **Konfigurasi Window Waktu Induktif**:
  * Tentukan tanggal referensi (misal $T_0$ = 1 Januari 2017).
  * **Train**: $T_0$ s/d $T_0 + 300\text{ hari}$
  * **Validation**: $T_0 + 300\text{ hari}$ s/d $T_0 + 330\text{ hari}$ (30 hari)
  * **Embargo Gap**: $T_0 + 330\text{ hari}$ s/d $T_0 + 338\text{ hari}$ (8 hari dilewati / tidak dipakai latih/evaluasi)
  * **Test**: $T_0 + 338\text{ hari}$ s/d $T_0 + 398\text{ hari}$ (60 hari)

### B. Arsitektur Model (Baseline Terkontrol)
* Tetap menggunakan arsitektur baseline yang stabil untuk memastikan perbandingan yang adil:
  * **GraphSAGE**: 2-layer HeteroGraphSAGE, `hidden_channels = 128`, `aggr = 'mean'`.
  * **Feature Scaling**: `StandardScaler` pada User/Merchant aggregates (dari temuan exp18c v2).
  * **Classifier**: XGBoost (`n_estimators = 100`, `max_depth = 5`, `learning_rate = 0.05`).
  * **Neighbor Sampling**: Uniform Sampling (`num_neighbors = [15, 10]`, `batch_size = 1024`).

### C. Protokol Evaluasi
* Binerisasi dengan $\tau^*$ optimal dari data validasi (temuan EXP20) serta pelaporan threshold default 0.5 sebagai komparasi.
* Pencatatan metrik: F1-Score, Recall, Precision, AUCPR, Lift@1%, TP@500, dan waktu komputasi.

---

## 5. Rencana File & Struktur Output
* Folder Skrip: `code/exp21/`
* Modifikasi File:
  * `data_preprocessing.py`: Menambahkan fungsi filter temporal 2016–2020 dan generator rolling window.
  * `config.kaggle.yaml`: Menambahkan parameter `time_window` (300d train, 30d val, 8d gap, 60d test).
  * `main.py`: Adaptasi pipeline pemotongan data berbasis tanggal.
* Output Hasil:
  * `result/exp21/exp21_uniform_results.json`
  * `result/exp21/window_statistics.json` (jumlah node, transaksi, fraud rate per split)
  * `result/exp21/README.md`

---

## 6. Kriteria Keberhasilan (Success Criteria)
1. Eksekusi end-to-end (GNN training + feature extraction + XGBoost) selesai dalam waktu $< 45$ menit di GPU Kaggle tanpa kendala memori.
2. Metrik AUCPR dan F1-Score mengalami lompatan signifikan dibanding exp18c v2, membuktikan bahwa eliminasi *30-year concept drift* adalah faktor penentu utama.
3. Terbentuknya basis data uji bersih (*clean testbed*) untuk eksperimen perbaikan arsitektur dan sampling berikutnya.

---

## 7. Kontribusi untuk Laporan Tugas Akhir
* Memberikan bab justifikasi metodologis mengenai pentingnya *temporal locality* dan mitigasi *concept drift* pada pemodelan transaksi finansial jangka panjang.
* Menjadikan baseline penelitian setara secara metodologis dengan standar publikasi internasional (IEEE TKDE / SHINE).
