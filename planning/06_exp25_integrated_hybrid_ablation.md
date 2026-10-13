# Rencana Eksperimen 25 (EXP25): Integrasi Utuh Pipeline Hibrida & Studi Ablasi Komparatif Final

## 1. Identitas Eksperimen
* **ID Eksperimen**: `EXP25`
* **Nama Rencana**: *Integrated Optimal Hybrid Pipeline & Comprehensive Ablation Study*
* **Tipe Eksperimen**: Integrasi Sistem & Evaluasi Akhir (*Final Benchmark*)
* **Target Utama**: Mengintegrasikan seluruh komponen terbaik yang telah divalidasi dari EXP20 hingga EXP24 ke dalam satu arsitektur hibrida terpadu, melakukan studi ablasi (*ablation study*) komprehensif, dan menyusun tabel komparasi final untuk Bab 4 Tugas Akhir.

---

## 2. Latar Belakang & Motivasi
Setelah melalui tahapan pembuktian inkremental:
* **EXP20**: Menuntaskan kalibrasi threshold dinamis ($\tau^*$) pada data validasi.
* **EXP21**: Menyelaraskan protokol jendela waktu induktif dengan era 2016–2020 (protokol SHINE FRAUD II).
* **EXP22**: Membuktikan keunggulan agregator `max` dalam menangkap sinyal anomali minoritas.
* **EXP23**: Memperkaya fitur dengan konteks kecepatan temporal (*velocity*) dan rasio historis nasabah.
* **EXP24**: Memilih strategi *neighbor sampling* terbaik berbasis kemiripan atribut.

`EXP25` bertindak sebagai puncak dari seluruh rangkaian penelitian. Eksperimen ini mengonsolidasikan arsitektur final dan menguji secara ketat kontribusi setiap komponen melalui **Studi Ablasi (*Ablation Study*)** untuk menjawab hipotesis besar Tugas Akhir:
> *"Apakah representasi graf dari GraphSAGE dengan strategi neighbor sampling berbasis atribut mampu memberikan peningkatan performa deteksi fraud yang signifikan jika digabungkan dengan pengklasifikasi XGBoost dibandingkan model tabular murni maupun model graf murni?"*

---

## 3. Matriks Studi Ablasi (*Ablation Design*)

Untuk memberikan analisis akademik yang mendalam di Bab 4, EXP25 mengeksekusi 5 variasi model pada kondisi data uji yang identik:

| Model ID | Komponen Arsitektur | Deskripsi Model | Tujuan Pembuktian |
| :--- | :--- | :--- | :--- |
| **M1** | Tabular Only | XGBoost murni (Hanya fitur tabular EXP23 tanpa GNN) | Mengukur performa baseline pohon keputusan tanpa graf |
| **M2** | GNN Only | Standalone HeteroGraphSAGE (Klasifikasi langsung dari layer sigmoid GNN) | Mengukur performa baseline graf murni tanpa ensemble pohon |
| **M3** | Hybrid Baseline | GraphSAGE (Mean + Uniform) $\rightarrow$ XGBoost (Menyerupai exp18c) | Mengukur performa hibrida standar awal |
| **M4** | Hybrid Arsitektur | GraphSAGE (Max + Uniform) $\rightarrow$ XGBoost | Mengukur kontribusi agregator `max` |
| **M5** | **Proposed Optimal** | **GraphSAGE (Max + Enhanced Sampling) $\rightarrow$ XGBoost + Threshold $\tau^*$** | **Model Final Usulan Tugas Akhir** |

---

## 4. Spesifikasi Konfigurasi Akhir (Best Setup)
1. **Data Split**: Window 2016–2020 (300 hari Train, 30 hari Val, 8 hari Gap, 60 hari Test).
2. **Feature Set**: 15–20 fitur terkayakan (termasuk $\Delta t$, rolling tx count, rasio nominal).
3. **Graph Topology**: Tripartit Heterogen (`User`, `Merchant`, `Transaction`).
4. **Sampling Strategy**: Strategi terbaik dari EXP24 (`topology_temporal` / `nn_sampler`).
5. **GNN Architecture**:
   * Lapisan: 2-Layer `HeteroGraphSAGE`
   * Hidden Dim: 128 channels
   * Agregator: `max` pooling
   * Batch size: 1024 (Train) / 4096 (Inference)
6. **Classifier**: XGBoost (`n_estimators = 100`, `max_depth = 5`, `learning_rate = 0.05`).
7. **Binarization**: Threshold optimal $\tau^*$ dari kurva PR Validation Set.

---

## 5. Metrik & Analisis Komparatif yang Disajikan
1. **Metrik Klasifikasi Biner**:
   * F1-Score (Macro & Fraud-class)
   * Precision & Recall
   * Area Under Precision-Recall Curve (AUCPR)
   * Area Under ROC Curve (AUCROC)
2. **Metrik Operasional Industri Finansial**:
   * **Lift@1%**: Peningkatan konsentrasi fraud pada 1% transaksi paling mencurigakan.
   * **TP@500**: Jumlah fraud yang berhasil diinvestigasi dalam kuota harian 500 alert (setara tolok ukur paper SHINE).
3. **Efisiensi Komputasi**:
   * Waktu Pelatihan GNN (detik)
   * Waktu Ekstraksi Fitur / Inferensi GNN (detik)
   * Waktu Pelatihan & Prediksi XGBoost (detik)

---

## 6. Rencana File & Struktur Output
* Folder Skrip: `code/exp25/`
* Skrip Utama:
  * `main_ablation.py`: Skrip terintegrasi yang mengeksekusi ke-5 variasi model M1 s/d M5 secara otomatis.
* Output Hasil:
  * `result/exp25/ablation_study_summary.json`
  * `result/exp25/final_comparison_table.csv`
  * `result/exp25/pr_curves_ablation.png`
  * `result/exp25/roc_curves_ablation.png`
  * `result/exp25/recap_exp25_final.ipynb`
  * `result/exp25/README.md`

---

## 7. Kriteria Keberhasilan (Success Criteria)
1. Model Usulan (**M5**) membuktikan keunggulan statistik atas Model Tabular (**M1**) dan Model Graf Murni (**M2**) pada metrik AUCPR dan F1-Score.
2. Hasil AUCPR dan F1-Score model akhir berada pada skala performa yang sebanding dengan temuan paper SHINE untuk dataset FRAUD II (AUCPR $\ge 0.70$, F1 $\ge 0.65$).
3. Seluruh eksekusi berjalan stabil tanpa kendala memori GPU atau batas waktu komputasi.

---

## 8. Kontribusi untuk Laporan Tugas Akhir
* Menyajikan **Tabel Hasil Utama (Master Results Table)** dan grafik perbandingan untuk Bab 4.
* Memberikan kesimpulan ilmiah yang kuat untuk Bab 5 mengenai efektivitas integrasi GraphSAGE, strategi neighbor sampling berbasis atribut, dan classifier XGBoost pada deteksi fraud transaksi kartu kredit.
