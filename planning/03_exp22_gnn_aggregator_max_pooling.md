# Rencana Eksperimen 22 (EXP22): Inovasi Agregator GNN: Elementwise Maximization (Max) vs Mean

## 1. Identitas Eksperimen
* **ID Eksperimen**: `EXP22`
* **Nama Rencana**: *GNN Aggregator Innovation for Extreme Imbalance: Elementwise Maximization vs Mean*
* **Tipe Eksperimen**: Eksplorasi Arsitektur GNN (*Architectural Ablation*)
* **Target Utama**: Menguji pengaruh fungsi agregasi pesan `aggr='max'` dibandingkan baseline `aggr='mean'` pada lapisan `SAGEConv` untuk mengatasi permasalahan *minority feature dilution* pada transaksi fraud.

---

## 2. Latar Belakang & Motivasi
Dalam Graph Neural Network standar (termasuk GraphSAGE asli dan implementasi exp16–exp18c), fungsi agregasi ketetanggaan yang umum digunakan adalah **`mean`**:
$$m_{\mathcal{N}_i} = \frac{1}{|\mathcal{N}_i|} \sum_{j \in \mathcal{N}_i} h_j$$

Pada paper SHINE (Bagian III-D1, Persamaan 10 & 11), penulis menjelaskan kelemahan fundamental dari agregator `mean` pada graf transaksi perbankan dengan ketimpangan ekstrem:
> *"Dengan ketidakseimbangan kelas, tetangga kelas minoritas (fraud) berisiko tinggi kalah jumlah secara masif oleh tetangga kelas mayoritas. Penggunaan agregator 'mean' menyebabkan pengaruh fitur tetangga fraud lenyap (terlarut/diluted) ke dalam rata-rata ribuan transaksi normal pada merchant atau user yang sama."*

Untuk mengatasi hal tersebut, SHINE mengganti operasi rata-rata dengan **`elementwise maximization`**:
$$m_k = \max_{1 \le j \le |\mathcal{N}_{i,r}|} (H_{jk})$$

Pada `exp19`, ide agregator `max` sempat dicoba, namun gagal mendapatkan hasil karena dikombinasikan dengan arsitektur 3-layer dan 256 dimensi pada 5,1 juta transaksi, sehingga sesi Kaggle kehabisan waktu (*timeout* di tahap ekstraksi fitur).

Pada `EXP22`, pengujian agregator `max` dilakukan pada basis protokol data bersih (EXP21) dengan arsitektur **2-layer ringan (128 dimensi)**, sehingga isolasi efek agregator dapat dibuktikan secara murni tanpa kendala komputasi.

---

## 3. Hipotesis Penelitian
1. Agregator `max` (elementwise maximization) mempertahankan aktivasi anomali ekstrem yang dibawa oleh tetangga fraud, sehingga embedding yang dihasilkan memiliki daya pemisah (*separability*) yang lebih tajam bagi XGBoost dibanding agregator `mean`.
2. Model dengan `aggr='max'` akan menghasilkan AUCPR dan Lift@1% yang lebih tinggi dibanding model dengan `aggr='mean'`.
3. Dengan mempertahankan kedalaman 2 lapisan dan 128 dimensi pada protokol EXP21, pelatihan dan ekstraksi fitur akan berjalan cepat dan stabil tanpa ancaman *timeout*.

---

## 4. Desain & Spesifikasi Teknis

### A. Varian Model yang Dibandingkan (*Controlled Experiment*)
Semua konfigurasi data, seed (42), sampling (Uniform), optimizer, dan classifier XGBoost dijaga **identik**, dengan satu-satunya variabel pembeda adalah fungsi agregasi pada `HeteroConv`:

1. **Varian A (Baseline)**: `SAGEConv(..., aggr='mean')`
2. **Varian B (Proposed)**: `SAGEConv(..., aggr='max')`
3. *(Opsional Varian C)*: `SAGEConv(..., aggr='sum')` (Elementwise summation sesuai Persamaan 11 paper SHINE)

### B. Spesifikasi Arsitektur Model
* Kedalaman: **2 Layer** GraphSAGE.
* Hidden Dimension: **128 channels**.
* Input Projections: Linear projection untuk transaksi (15 dimensi/sesuai fitur), dan `StandardScaler` untuk User/Merchant aggregates (2 dimensi).
* Normalisasi & Aktivasi: `BatchNorm1d` + `ReLU`.
* Classifier Akhir: XGBoost (`n_estimators = 100`, `max_depth = 5`, `learning_rate = 0.05`).

---

## 5. Rencana File & Struktur Output
* Folder Skrip: `code/exp22/`
* Modifikasi File:
  * `model.py`: Menambahkan parameter fleksibel `aggr: str = 'max'` pada inisialisasi `HeteroConv(SAGEConv(...))`.
  * `config.kaggle.yaml`: Menambahkan parameter `model.aggr: 'max'`.
* Output Hasil:
  * `result/exp22/exp22_mean_results.json`
  * `result/exp22/exp22_max_results.json`
  * `result/exp22/embeddings_tsne_comparison.png` (Visualisasi t-SNE separabilitas kelas: Mean vs Max)
  * `result/exp22/README.md`

---

## 6. Kriteria Keberhasilan (Success Criteria)
1. Pelatihan 10 epoch dan ekstraksi fitur selesai dalam waktu $< 30$ menit.
2. Varian `max` mengungguli varian `mean` pada metrik **AUCPR** dan **Lift@1%**.
3. Hasil visualisasi embedding menunjukkan klaster transaksi fraud lebih terpisah dan tidak bertumpuk di tengah klaster transaksi normal.

---

## 7. Kontribusi untuk Laporan Tugas Akhir
* Membuktikan secara empiris teori SHINE bahwa fungsi agregasi non-linier (`max`) lebih superior daripada agregasi rata-rata (`mean`) pada graf biner yang timpang ekstrem.
* Menjawab kegagalan teknis exp19 dengan solusi arsitektur yang terukur efisiensinya.
