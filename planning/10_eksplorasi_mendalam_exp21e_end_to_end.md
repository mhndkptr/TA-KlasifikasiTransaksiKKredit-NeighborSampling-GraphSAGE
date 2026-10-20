# Rencana Eksplorasi Lanjutan EXP21E: Pure End-to-End HeteroGraphSAGE (`num_neighbors = [15, 10]`)

**Tanggal Penyusunan**: 17 Oktober 2026  
**Basis Eksperimen**: `EXP21E` ([`code/exp21e/`](file:///d:/Projects/Klasifikasi%20Transasi%20Kartu%20Kredit%20-%20GraphSAGE/tugas-akhir-klasifikasi-transaksi-kartu-kredit/code/exp21e/))  
**Konfigurasi Terkunci (Locked Parameters)**:
- **Arsitektur**: Pure End-to-End `HeteroGraphSAGE` (2-Layer, `hidden_channels = 128`, `BatchNorm1d`, `Dropout = 0.2`, `Linear(128, 1)` + `Sigmoid` — **Tanpa XGBoost Hilir**)
- **Kedalaman & Ukuran Sampling (Hop Fan-out)**: **`num_neighbors = [15, 10]`** ($k_1 = 15$ pada Hop-1, $k_2 = 10$ pada Hop-2)
- **Jendela Temporal (SHINE Protocol)**: **300 Hari Train / 30 Hari Val / 8 Hari Embargo Gap / 60 Hari Test** (`283.529` transaksi uji, `421` fraud `[0.148%]`)
- **Strategi Sampling Utama**:
  1. `Uniform Random Sampling` (Baseline)
  2. `Topology-Aware Sampling` (Inverse Sublinear Log-Degree + Min-Max Stabilization)
  3. `Importance-Based Sampling` (Sublinear Log-Degree Hub Proxy + Min-Max Stabilization)

---

## 1. Latar Belakang & Rasionalitas Eksplorasi pada `EXP21E`

Berdasarkan hasil evaluasi dan arahan dosen pembimbing:
1. **Isolasi Murni Efek Neighbor Sampling**: Penggunaan pengklasifikasi pohon keputusan hilir (XGBoost) dapat menyamarkan (*masking*) efek murni dari *neighbor sampling* karena *tree boosting* melakukan pemisahan non-linier berlapis di atas ruang embedding. Dengan menggunakan **Pure End-to-End GraphSAGE (`EXP21E`)**, kepala klasifikasi hanyalah proyeksi linier tunggal ($\hat{y}_t = \sigma(\mathbf{w}^\top \mathbf{h}_t^{(2)} + b)$), sehingga kualitas klasterisasi dan keterpisahan kelas (*linear separability*) sepenuhnya ditentukan oleh mekanisme *neighbor sampling* pada ukuran hop `[15, 10]`.
2. **Menghindari Perbandingan Tidak Setara (Embedding vs Atribut Mentah)**: Representasi laten $\mathbf{h}_t^{(2)} \in \mathbb{R}^{128}$ yang telah dilatih secara *supervised* tidak setara jika dibandingkan langsung (misalnya lewat *split gain share*) dengan atribut mentah $\mathbf{x}_t \in \mathbb{R}^{10}$. Oleh karena itu, eksplorasi harus membandingkan **ruang laten 128-d antar-sampler secara setara** atau membandingkan **model neural end-to-end dengan arsitektur identik** (dengan graf vs tanpa graf).

Saat ini, baseline `EXP21E` (`[15, 10]`, `seed=42`) telah menghasilkan temuan awal yang sangat kuat:

| Strategi Sampling (`[15, 10]`) | Optimal $\tau^*$ | Test F1 ($\tau^*$) | Test F1 ($\tau=0.50$) | Test Recall | Test Precision | Test AUPRC | Test ROC-AUC | TP @ 500 | FP / FN ($\tau^*$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Uniform (Baseline)** | 0.745 | 0.8854 | 0.6364 | 99.05% (417/421) | 80.04% | 0.8972 | 0.99990 | 414 / 500 | 104 FP / 4 FN |
| **Topology-Aware** | 0.720 | 0.9093 | 0.8744 | **100.00% (421/421)** | 83.37% | 0.9381 | 0.99994 | 420 / 500 | 84 FP / **0 FN** |
| **Importance-Based** | 0.730 | **0.9152** | **0.8957** | **100.00% (421/421)** | **84.37%** | **0.9620** | **0.99996** | **421 / 500** | **78 FP** / **0 FN** |

Agar analisis komparatif ketiga strategi *neighbor sampling* pada hop `[15, 10]` menjadi sangat mendalam, tajam, dan layak publikasi IEEE, berikut adalah **5 Pilar Eksplorasi Lanjutan** yang dapat diimplementasikan pada `EXP21E`.

---

## 2. Peta 5 Pilar Eksplorasi pada `EXP21E` (`num_neighbors = [15, 10]`)

### Pilar 1: Eksplorasi Ruang Laten 128-d & Kuantifikasi *Representation Dilution* (Antar-Sampler)
* **Tujuan Ilmiah**: Membuktikan secara matematis dan visual mengapa `Uniform Sampling` mengalami *representation dilution* (tercermin dari jatuhnya F1 di $\tau=0.50$ ke `0.6364` dan AUPRC `0.8972`), sedangkan `Topology-Aware` dan `Importance-Based` menjaga kemurnian representasi fraud (`AUPRC 0.9381` dan `0.9620`).
* **Metode Ekstraksi**: Menggunakan fungsi `model(x_dict, edge_index_dict, return_embeddings=True)` pada [`code/exp21e/model.py`](file:///d:/Projects/Klasifikasi%20Transasi%20Kartu%20Kredit%20-%20GraphSAGE/tugas-akhir-klasifikasi-transaksi-kartu-kredit/code/exp21e/model.py#L61-L85) menggunakan bobot model terbaik yang sudah tersimpan di [`model/exp21e/`](file:///d:/Projects/Klasifikasi%20Transasi%20Kartu%20Kredit%20-%20GraphSAGE/tugas-akhir-klasifikasi-transaksi-kartu-kredit/model/exp21e/).
* **Metrik & Eksplorasi yang Dihitung**:
  1. **Inter-Class Centroid Distance ($\Delta_{\text{cos}}$ & $\Delta_{\text{L2}}$)**: Jarak kosinus dan Euclidean antara pusat massa (*centroid*) embedding transaksi Normal ($\boldsymbol{\mu}_{\text{norm}} \in \mathbb{R}^{128}$) dan transaksi Fraud ($\boldsymbol{\mu}_{\text{fraud}} \in \mathbb{R}^{128}$).
  2. **Fisher's Discriminant Ratio (FDR) & Intra-Class Compactness**: Mengukur rasio jarak antar-kelas terhadap variansi di dalam kelas ($\frac{\|\boldsymbol{\mu}_{\text{fraud}} - \boldsymbol{\mu}_{\text{norm}}\|_2^2}{\sigma_{\text{fraud}}^2 + \sigma_{\text{norm}}^2}$). Jika terjadi *representation dilution* pada `Uniform`, maka $\sigma_{\text{fraud}}^2$ akan membengkak dan mendekati distribusi normal.
  3. **Logit Margin & Probability Calibration Analysis**: Menganalisis distribusi skor logit mentah $z_t = \mathbf{w}^\top \mathbf{h}_t + b$ dan *Expected Calibration Error* (ECE) / *Brier Score*. Menjelaskan mengapa `Uniform` menghasilkan banyak transaksi normal dengan skor di zona abu-abu $[0.50, 0.74]$ (sehingga F1 @ $\tau=0.50$ anjlok ke `0.6364`), sementara `Importance` menekan skor normal jauh di bawah `0.50`.
  4. **Visualisasi Proyeksi 2D t-SNE / UMAP (3 Panel Sejajar)**: Menampilkan pemisahan klaster transaksi Normal vs Fraud (termasuk penanda khusus untuk titik *False Positive* dan *False Negative*) pada `Uniform`, `Topology-Aware`, dan `Importance-Based`.
* **Kebutuhan Komputasi**: **Zero-Retraining (100% Post-Hoc)** — hanya butuh *forward pass* inferensi (~15 detik per sampler menggunakan file `.pt` yang sudah ada).

---

### Pilar 2: Analisis Terstratifikasi Derajat Simpul (*Degree-Stratified*) & Ketahanan Kamuflase (*Heterophily / Camouflage Analysis*)
* **Tujuan Ilmiah**: Menjawab *bagaimana* mekanisme pembobotan derajat pada `Topology-Aware` ($w \propto 1/\log(1+\deg)$) dan `Importance-Based` ($w \propto \log(1+\deg)$) bekerja secara spesifik pada berbagai zona kepadatan graf dengan fan-out `[15, 10]`. Ketika hop-1 dibatasi hanya mengambil $k_1 = 15$ tetangga dan hop-2 mengambil $k_2 = 10$ tetangga, pemilihan tetangga pada *super-hub merchant* (yang memiliki ribuan transaksi) menjadi sangat krusial!
* **Metode Eksplorasi**:
  1. **Stratifikasi Berdasarkan Derajat Merchant & User ($\text{deg}(m), \text{deg}(u)$)**:
     Membagi `283.529` transaksi uji ke dalam 3–4 kelompok kuartil derajat Merchant/User tempat transaksi terjadi:
     - **Low-Degree Bucket ($Q_1$, $\text{deg} \le 15$)**: Seluruh tetangga terambil ($|\mathcal{N}| \le k_1$).
     - **Medium-Degree Bucket ($Q_2 - Q_3$, $15 < \text{deg} \le 200$)**: Sampling mulai menyeleksi subset tetangga.
     - **Super-Hub Bucket ($Q_4$ / Top 10%, $\text{deg} > 200$)**: Efek *representation dilution* paling ekstrem karena hanya 15 dari ratusan/ribuan transaksi yang diambil.
     - *Output*: Tabel & Grafik F1, Recall, AUPRC, dan jumlah *False Positive* (`104` vs `84` vs `78`) serta *False Negative* (`4` vs `0` vs `0`) pada masing-masing bucket derajat untuk ketiga sampler.
  2. **Analisis *Camouflaged Fraud* (Rasio Homofili Lokal Merchant)**:
     Mengelompokkan 421 transaksi fraud di Test Set berdasarkan tingkat kamuflasenya di merchant:
     - *High-Camouflage Fraud*: Transaksi fraud yang terjadi di merchant raksasa dengan rasio historis fraud $< 0.2\%$ (dikelilingi $>99.8\%$ transaksi normal).
     - *Low-Camouflage / Concentrated Fraud*: Transaksi fraud pada merchant yang sering menjadi target sindikat.
     - *Output*: Membuktikan di kelompok mana 4 kasus *False Negative* milik `Uniform` gagal terdeteksi namun berhasil diselamatkan 100% oleh `Topology-Aware` dan `Importance-Based`.
* **Kebutuhan Komputasi**: **Zero-Retraining (100% Post-Hoc)** — menggunakan file prediksi [`result/exp21e/*.npz`](file:///d:/Projects/Klasifikasi%20Transasi%20Kartu%20Kredit%20-%20GraphSAGE/tugas-akhir-klasifikasi-transaksi-kartu-kredit/result/exp21e/) dan struktur derajat graf uji (~30 detik eksekusi lokal).

---

### Pilar 3: Komparasi Apple-to-Apple Tanpa Bias Arsitektur (*End-to-End Tabular MLP* vs *GraphSAGE `[15, 10]`*)
* **Tujuan Ilmiah**: Menggantikan tabel "GNN vs XGBoost" yang dikritik dosen pembimbing dengan eksperimen kontrol yang **100% setara secara arsitektur dan fungsi objektif**.
* **Desain Eksperimen Kontrol**:
  1. **Baseline Non-Graf Setara (`End-to-End Tabular MLP`, $\mathcal{N}(v) = \emptyset$)**:
     Membangun jaringan saraf tiruan murni di atas fitur transaksi $\mathbf{x}_t \in \mathbb{R}^{10}$ dengan arsitektur **identik** dengan cabang diri (*self-projection*) pada `HeteroGraphSAGE`:
     - 2 Lapisan `Linear(in, 128) -> BatchNorm1d(128) -> ReLU -> Dropout(0.2)` diikuti kepala klasifikasi `Linear(128, 1) -> Sigmoid`.
     - Dilatih dengan konfigurasi identik: `pos_weight = sqrt(neg/pos) = 27.06`, `Adam(lr=0.003)`, `epochs=20`, dan kalibrasi $\tau^*$ pada Validation Set.
     - Perbedaan **satu-satunya** antara `Tabular MLP` dan `EXP21E GraphSAGE` adalah ada atau tidaknya pesan dari tetangga graf ($\mathcal{N}(v)$ dengan hop `[15, 10]`).
  2. **Ablasi Komponen Fitur Node Entitas (`No User/Merchant Summary Features`)**:
     Menguji `EXP21E` (`Uniform`, `Topology`, `Importance`) ketika fitur inisialisasi $\mathbf{x}_u, \mathbf{x}_m$ diatur menjadi vektor konstan/nol, sehingga informasi yang mengalir dari hop-1 (`15`) dan hop-2 (`10`) murni berasal dari fitur transaksi tetangga dan struktur konektivitas.
* **Kebutuhan Komputasi**: **Sangat Ringan** — Pelatihan `Tabular MLP` tidak melibatkan *message passing* graf sehingga berjalan dalam **< 1 menit** di GPU/CPU lokal.

---

### Pilar 4: Eksplorasi Sensitivitas Mekanisme Internal `EXP21E` pada Hop `[15, 10]`
* **Tujuan Ilmiah**: Menguji seberapa kokoh keunggulan ketiga strategi *neighbor sampling* pada hop `[15, 10]` terhadap variasi komponen internal GNN:
  1. **Eksplorasi Fungsi Agregasi (`Mean Aggregator` vs `Max-Pooling Aggregator` vs `Sum/Attention`) pada `EXP21E`**:
     - Pada `Mean` aggregator, semua 15 tetangga hop-1 dan 10 tetangga hop-2 dirata-ratakan sehingga sangat sensitif terhadap *dilution*.
     - Pada `Max` aggregator ($\max_{u \in \mathcal{N}_S(v)}$), model mengambil sinyal aktivasi tertinggi. Menguji ketiga sampler (`Uniform`, `Topology`, `Importance`) dengan `aggr='max'` pada `EXP21E` akan membuktikan apakah *Importance* dan *Topology* tetap unggul di berbagai jenis agregator.
  2. **Sensitivitas Bobot Fungsi Loss (`pos_weight` Ablation pada `EXP21E`)**:
     - Membandingkan efek *Damped Weighting* ($\sqrt{N_- / N_+} \approx 27.06$) vs *Unweighted* ($1.0$) vs *Full Inverse Frequency* ($N_- / N_+ \approx 732.18$) pada arsitektur `EXP21E` end-to-end.
  3. **Sensitivitas Smoothing Floor ($\alpha$) pada `Importance` & `Topology` Sampler**:
     - Saat ini [`code/exp21e/samplers.py`](file:///d:/Projects/Klasifikasi%20Transasi%20Kartu%20Kredit%20-%20GraphSAGE/tugas-akhir-klasifikasi-transaksi-kartu-kredit/code/exp21e/samplers.py) menggunakan batas bawah (*smoothing floor*) $\alpha = 0.1$ pasca *min-max normalization* ($w_{\text{norm}} + 0.1$). Mengeksplorasi dampak variasi $\alpha \in \{0.01, 0.1, 0.5\}$ terhadap keseimbangan eksplorasi-eksploitasi tetangga pada hop `[15, 10]`.
* **Kebutuhan Komputasi**: Membutuhkan *training run* `EXP21E` (~7–9 menit per run di GPU).

---

### Pilar 5: Uji Ketahanan Statistik, *Multi-Seed Stability*, & Kurva Operasional Perbankan
* **Tujuan Ilmiah**: Memastikan bahwa selisih performa antara `Uniform` (`F1 0.8854`, `AUPRC 0.8972`), `Topology-Aware` (`F1 0.9093`, `AUPRC 0.9381`), dan `Importance-Based` (`F1 0.9152`, `AUPRC 0.9620`) pada hop `[15, 10]` terbukti **signifikan secara statistik** dan berdampak nyata secara operasional.
* **Metode Eksplorasi**:
  1. **Uji McNemar ($\chi^2$) Berpasangan**: Menguji signifikansi perbedaan prediksi biner pada `283.529` transaksi uji antara (`Uniform` vs `Topology`), (`Uniform` vs `Importance`), dan (`Topology` vs `Importance`).
  2. **Non-Parametric Bootstrap (1.000 Iterasi, 95% Confidence Interval)**: Menghitung $\text{Mean} \pm \text{SD}$ dan rentang 95% CI untuk F1-Score, AUPRC, Precision, dan Recall dari ketiga sampler.
  3. **Kurva *Precision@K* & *Alert Triage Capacity* ($K \in [100, 200, 300, 421, 500, 1000]$)**:
     Mengevaluasi berapa banyak fraud yang tertangkap jika tim analis bank hanya mampu memeriksa $K$ transaksi berisiko tertinggi per periode. Pada $K=500$, `Uniform` menangkap `414/421` (`98.34%`), `Topology` menangkap `420/421` (`99.76%`), dan `Importance` menangkap **`421/421` (`100.00%`)**!
  4. **Kurva Sensitivitas Threshold ($\tau \in [0.10, 0.95]$)**:
     Memvisualisasikan stabilitas F1-Score terhadap pergeseran threshold. Membuktikan secara visual bahwa kurva F1 `Uniform` sangat sempit/curam (runtuh ke `0.6364` di $\tau=0.50$), sedangkan `Topology` (`0.8744`) dan `Importance` (`0.8957`) memiliki dataran tinggi (*plateau*) F1 yang jauh lebih lebar dan tahan terhadap *threshold drift*.
  5. **Evaluasi Multi-Seed (`seed = 42, 100, 2026`)** *(Opsional bila waktu mengizinkan)*: Menjalankan `EXP21E` (`[15, 10]`) pada 3 *random seed* berbeda untuk melaporkan rata-rata dan simpangan baku antar-inisialisasi bobot.
* **Kebutuhan Komputasi**: Poin 1–4 **Zero-Retraining (100% Post-Hoc, < 1 menit)** dari file `.npz` yang sudah ada; Poin 5 membutuhkan 2 seed tambahan (~25 menit per seed).

---

## 3. Matriks Prioritas Implementasi & Estimasi Waktu

Untuk memudahkan eksekusi secara bertahap, seluruh eksplorasi di atas dibagi menjadi **3 Tahap (Tier)** berdasarkan waktu komputasi dan dampaknya terhadap paper:

| Tahap | Kode Eksplorasi | Deskripsi Eksplorasi (`EXP21E`, Hop `[15, 10]`) | Butuh Retraining GNN? | Estimasi Waktu | Output untuk Paper IEEE |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **Tahap 1 (Prioritas Utama & Instan)** | **`EXP21E-PostHoc`** | **Pilar 1** (Jarak Centroid 128-d, FDR, Logit Margin, t-SNE) + **Pilar 2** (Stratifikasi Derajat Merchant/User & Analisis Kamuflase) + **Pilar 5** (Uji McNemar, Bootstrap 95% CI, Kurva Threshold & Triage $K$) | **Tidak** (Pakai `.pt` & `.npz` `EXP21E` yang sudah ada) | **~3 - 5 Menit** | Tabel Separabilitas Laten, Tabel & Grafik *Degree-Stratified*, Plot t-SNE, Kurva PR & $\tau$, Nilai $p$-value McNemar |
| **Tahap 2 (Sangat Cepat)** | **`EXP21E-MLP`** | **Pilar 3**: Pelatihan & Evaluasi `End-to-End Tabular MLP` (arsitektur 2-layer 128-d & loss identik tanpa graf) sebagai pembanding *Apple-to-Apple* terhadap `EXP21E` | Hanya melatih MLP (Tanpa Graf) | **~1 - 2 Menit** | Tabel *Pure Architectural Graph Lift* (`Tabular MLP` vs `Uniform` vs `Topology` vs `Importance`) |
| **Tahap 3 (Eksplorasi GNN Tambahan)** | **`EXP21E-Ablation`** | **Pilar 4**: Ablasi Agregator (`Mean` vs `Max`) atau Sensitivitas Embargo Gap (`0d`, `8d`, `30d`) / Multi-Seed langsung pada arsitektur `EXP21E` (`[15, 10]`) | **Ya** (Melatih ulang `EXP21E`) | **~24 Menit** per paket (3 sampler $\times$ 8 mnt) | Tabel Ketahanan Lintas-Agregator (`Mean` vs `Max`) & Validasi *Embargo Gap* Murni pada `EXP21E` |

---

## 4. Rencana Struktur Kode & Artefak Output (`code/exp21e/`)

Untuk menjaga kerapian repositori, implementasi eksplorasi lanjutan akan ditempatkan langsung secara modular di dalam ekosistem `EXP21E`:

1. **`code/exp21e/explore_posthoc_analysis.py`** *(Tahap 1)*:
   - Memuat graf uji (`300d/30d/8d/60d`, hop `[15, 10]`), bobot [`model/exp21e/exp21e_{uniform,topology,importance}_best_gnn.pt`](file:///d:/Projects/Klasifikasi%20Transasi%20Kartu%20Kredit%20-%20GraphSAGE/tugas-akhir-klasifikasi-transaksi-kartu-kredit/model/exp21e/), dan prediksi [`result/exp21e/exp21e_{uniform,topology,importance}_predictions.npz`](file:///d:/Projects/Klasifikasi%20Transasi%20Kartu%20Kredit%20-%20GraphSAGE/tugas-akhir-klasifikasi-transaksi-kartu-kredit/result/exp21e/).
   - Mengekstrak embedding 128-d pada Test Set dan menghitung metrik separabilitas laten (Cosine/L2 Centroid Distance, Fisher Ratio, Silhouette Score, ECE).
   - Melakukan stratifikasi performa berdasarkan derajat Merchant/User (*Low*, *Medium*, *Super-Hub*) dan tingkat heterofili/kamuflase.
   - Menjalankan uji signifikansi McNemar ($\chi^2$) dan Bootstrap 1.000 iterasi.
   - Menghasilkan seluruh plot visualisasi beresolusi 300 DPI di `result/exp21e/figures/`.
2. **`code/exp21e/run_tabular_mlp_baseline.py`** *(Tahap 2)*:
   - Mengimplementasikan kelas `TabularMLP(in_channels=10, hidden_channels=128, dropout=0.2)` yang identik dengan cabang linier `HeteroGraphSAGE`.
   - Melatih model pada jendela `300d Train / 30d Val / 8d Gap / 60d Test` dengan `pos_weight = 27.06` dan mengevaluasi semua metrik (`F1`, `Recall`, `Precision`, `AUPRC`, `ROC-AUC`, `TP@500`, `FP/FN`).
   - Menyimpan hasil ke `result/exp21e/exp21e_tabular_mlp_results.json` dan `.npz`.
3. **`code/exp21e/run_ablation_max_aggr.py`** *(Tahap 3 - Opsional/Lanjutan)*:
   - Menjalankan `EXP21E` dengan `aggr = 'max'` pada hop `[15, 10]` untuk ketiga sampler guna menguji konsistensi urutan performa di luar `Mean Aggregator`.

