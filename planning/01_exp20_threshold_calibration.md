# Rencana Eksperimen 20 (EXP20): Kalibrasi Threshold Dinamis & Audit Metrik Evaluasi

## 1. Identitas Eksperimen
* **ID Eksperimen**: `EXP20`
* **Nama Rencana**: *Dynamic Threshold Tuning & Evaluation Metric Recalibration*
* **Tipe Eksperimen**: Post-Processing & Evaluation Protocol Optimization (Nol Training Ulang GNN)
* **Target Utama**: Memvalidasi pengaruh *fixed threshold 0.5* terhadap kolapsnya F1-score pada exp18c v2 dan merekalibrasi performa menggunakan *best threshold* pada Validation Set.

---

## 2. Latar Belakang & Motivasi
Pada hasil pengujian `exp18c v2`, model menghasilkan metrik:
* F1-Score: **0.1399** (Uniform), **0.0772** (Topology), **0.0205** (Importance)
* Recall: **8.5%** (Uniform), **4.4%** (Topology), **4.0%** (Importance)

Pemeriksaan pada `main.py` (baris 203) menunjukkan bahwa binerisasi prediksi menggunakan ambang batas tetap:
$$\hat{y} = \mathbb{I}(P(\text{fraud}) > 0.5)$$

Pada dataset transaksi kartu kredit dengan rasio kelas fraud sangat langka (~0,12%), probabilitas fraud yang dihasilkan oleh model terlatih secara alami berkisar pada nilai rendah ($0.01 - 0.30$). Memotong probabilitas pada angka $0.5$ membuang lebih dari 90% kasus fraud yang terdeteksi, sehingga Recall anjlok dan F1-score kolaps. 

Paper SHINE (Tabel VII) melaporkan F1 sebesar **0.6895** pada GraphSAGE dan **0.7499** pada SHINE dengan mengevaluasi model pada titik kerja optimal atau menggunakan metrik berbasis kurva seperti **AUCPR** ($0.6725 - 0.8329$).

---

## 3. Hipotesis Penelitian
1. Probabilitas fraud keluaran model XGBoost pada exp18c terkonsentrasi di bawah $0.5$.
2. Dengan mencari threshold optimal $\tau^* \in [0.01, 0.50]$ yang memaksimalkan F1-score pada data **Validation**, nilai F1-Score pada data **Test** akan meningkat signifikan (diproyeksikan naik dari $\approx 0.14$ ke $\approx 0.40 - 0.60$) tanpa mengubah bobot model GNN atau XGBoost yang sudah ada.
3. Metrik AUCPR dan kurva Precision-Recall mencerminkan kemampuan pemisahan kelas yang sebenarnya tanpa distorsi oleh pemilihan threshold biner tunggal.

---

## 4. Desain & Spesifikasi Teknis

### A. Data & Model Checkpoint yang Digunakan
* Menggunakan artefak model yang sudah tersimpan dari **exp18c v2**:
  * GNN Weight: `model/exp18c/exp18c_{sampling}_gnn.pt`
  * XGBoost Model: `model/exp18c/exp18c_{sampling}_xgb.json`
* Subset data evaluasi:
  * **Validation Set** (15% data transaksi kronologis) $\rightarrow$ Digunakan **hanya** untuk mencari threshold optimal $\tau^*$.
  * **Test Set** (15% data transaksi kronologis) $\rightarrow$ Digunakan untuk menguji performa final menggunakan $\tau^*$.

### B. Algoritma Pencarian Threshold Optimal ($\tau^*$)
1. Ekstraksi probabilitas prediksi pada subset validasi: $P_{val} = f(X_{val})$.
2. Iterasi calon threshold $\tau \in \{0.001, 0.002, \dots, 0.500\}$ dengan resolusi $0.005$:
   $$\text{F1}(\tau) = \frac{2 \cdot \text{Precision}(\tau) \cdot \text{Recall}(\tau)}{\text{Precision}(\tau) + \text{Recall}(\tau)}$$
3. Pilih $\tau^* = \arg\max_\tau \text{F1}_{val}(\tau)$.
4. Terapkan $\tau^*$ secara strictly out-of-sample ke subset uji:
   $$\hat{y}_{test} = \mathbb{I}(P_{test} \ge \tau^*)$$

---

## 5. Metrik yang Dicatat & Dianalisis
1. **F1-Score @ 0.5** vs **F1-Score @ $\tau^*$** (Pembuktian dampak kalibrasi threshold).
2. **Recall & Precision @ $\tau^*$**.
3. **AUCPR (Area Under Precision-Recall Curve)**: Metrik utama terlepas dari threshold.
4. **Kurva Precision-Recall (PR Curve)**: Visualisasi perbandingan metode sampling (`uniform`, `topology`, `importance`).
5. **Lift@1% & TP@500**: Mengukur volume fraud tertangkap pada kapasitas operasional harian perbankan.

---

## 6. Rencana File & Struktur Output
* Folder Skrip: `code/exp20/` (atau notebook evaluasi `recap/recalibrate_exp18c.ipynb`)
* File Evaluasi: `evaluate_threshold.py`
* Output Hasil:
  * `result/exp20/threshold_tuning_results.json`
  * `result/exp20/pr_curves_comparison.png`
  * `result/exp20/threshold_vs_f1_plot.png`
  * `result/exp20/README.md`

---

## 7. Kriteria Keberhasilan (Success Criteria)
* Teridentifikasinya nilai $\tau^*$ optimal yang realistis pada validasi (diprediksi berada di rentang $0.05 - 0.25$).
* Terjadi peningkatan F1-Score pada Test Set sebesar minimal **2x lipat** dibanding baseline threshold 0.5.
* Tidak ada kebocoran data (*data leakage*): threshold mutlak ditentukan dari data validasi, bukan data uji.

---

## 8. Kontribusi untuk Laporan Tugas Akhir
* Memberikan bab pembahasan kritis (*Critical Discussion*) mengenai mengapa metrik default (threshold 0.5) gagal pada dataset yang sangat timpang (*extreme imbalance*).
* Memisahkan antara kelemahan arsitektur model vs kelemahan protokol evaluasi binerisasi.
