# Rencana Eksplorasi Lanjutan EXP21D: Hybrid HeteroGraphSAGE + XGBoost (`num_neighbors = [15, 10]`)

**Tanggal Penyusunan**: 18 Oktober 2026  
**Basis Eksperimen**: `EXP21D v1` ([`code/exp21d/`](file:///d:/Projects/Klasifikasi%20Transasi%20Kartu%20Kredit%20-%20GraphSAGE/tugas-akhir-klasifikasi-transaksi-kartu-kredit/code/exp21d/) & [`result/exp21d/v1/`](file:///d:/Projects/Klasifikasi%20Transasi%20Kartu%20Kredit%20-%20GraphSAGE/tugas-akhir-klasifikasi-transaksi-kartu-kredit/result/exp21d/v1/))  
**Konfigurasi Terkunci (Locked Parameters)**:
- **Arsitektur Utama**: Hybrid `HeteroGraphSAGE` (2-Layer, `hidden_channels = 128`, `aggr = 'mean'`, `dropout = 0.2`) $\rightarrow$ Ekstraksi Fitur Hibrida $\rightarrow$ `Regularized XGBoost Classifier` (`n_estimators = 100`, `max_depth = 4`, `min_child_weight = 10`, `subsample = 0.8`, `colsample_bytree = 0.8`, `scale_pos_weight = sqrt(neg/pos) = 27.06`)
- **Kedalaman & Ukuran Sampling (Hop Fan-out)**: **`num_neighbors = [15, 10]`** ($k_1 = 15$ pada Hop-1, $k_2 = 10$ pada Hop-2)
- **Jendela Temporal Terbaik (SHINE FRAUD II Protocol)**: **300 Hari Train / 30 Hari Val / 8 Hari Embargo Gap / 60 Hari Test** (`283.529` transaksi uji, `421` fraud `[0.148%]`)
- **3 Strategi Neighbor Sampling yang Dianalisis**:
  1. `Uniform Random Sampling` (Baseline)
  2. `Topology-Aware Sampling` (Inverse Sublinear Log-Degree + Min-Max Stabilization)
  3. `Importance-Based Sampling` (Sublinear Log-Degree Hub Proxy + Min-Max Stabilization)

---

## 1. Evaluasi Baseline `EXP21D v1` (`[15, 10]`) & Tantangan Metodologis yang Harus Dijawab

Pada eksperimen [`EXP21D v1`](file:///d:/Projects/Klasifikasi%20Transasi%20Kartu%20Kredit%20-%20GraphSAGE/tugas-akhir-klasifikasi-transaksi-kartu-kredit/result/exp21d/v1/exp21d_summary_report.md) (`num_neighbors = [15, 10]`, jendela `300d/30d/8d/60d`), hasil komparasi yang tercatat saat ini adalah:

| Strategi Sampling (`[15, 10]`) | Subset Fitur | Optimal $\tau^*$ | Test F1 ($\tau^*$) | Test F1 ($\tau=0.50$) | Test Recall | Test Precision | Test AUPRC | Test ROC-AUC | FP / FN ($\tau^*$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Uniform** | Tabular Only (10-d) | 0.745 | 0.9207 | — | 90.97% (383/421) | 93.19% | 0.9479 | 0.99990 | 28 FP / 38 FN |
| | GraphSAGE Only (128-d) | 0.225 | 0.9083 | — | 100.00% (421/421) | 83.20% | 0.8952 | 0.99992 | 85 FP / 0 FN |
| | **Hybrid (138-d)** | **0.600** | **0.9481** | **0.9459** | **99.76% (420/421)** | **90.32%** | **0.9550** | **0.99995** | **45 FP / 1 FN** |
| **Topology-Aware** | Tabular Only (10-d) | 0.735 | 0.9207 | — | 90.97% (383/421) | 93.19% | 0.9424 | 0.99988 | 28 FP / 38 FN |
| | GraphSAGE Only (128-d) | 0.600 | 0.9015 | — | 100.00% (421/421) | 82.07% | 0.9301 | 0.99994 | 92 FP / 0 FN |
| | **Hybrid (138-d)** | **0.380** | **0.9450** | **0.9461** | **100.00% (421/421)** | **89.57%** | **0.9746** | **0.99997** | **49 FP / 0 FN** |
| **Importance-Based** | Tabular Only (10-d) | 0.735 | 0.9185 | — | 90.97% (383/421) | 92.74% | 0.9438 | 0.99989 | 30 FP / 38 FN |
| | GraphSAGE Only (128-d) | 0.440 | 0.8901 | — | 100.00% (421/421) | 80.19% | 0.8808 | 0.99991 | 104 FP / 0 FN |
| | **Hybrid (138-d)** | **0.415** | **0.9461** | **0.9461** | **100.00% (421/421)** | **89.77%** | **0.9634** | **0.99996** | **48 FP / 0 FN** |

### Mengapa `EXP21D` (`[15, 10]`) Membutuhkan Eksplorasi Lanjutan?
Ada **3 alasan ilmiah utama** mengapa kita perlu melakukan eksplorasi mendalam pada `EXP21D`:
1. **Menjawab Kritik Dosen Pembimbing (Masalah Kesetaraan Embedding vs Atribut Mentah)**:
   - Dosen pembimbing menilai bahwa membandingkan *Tree Split Gain Share* antara **128 fitur embedding GNN** (yang sudah dilatih dengan label fraud) melawan **10 atribut tabular mentah** (`>99.5%` vs `<0.5%`) adalah **tidak setara (unfair comparison)** karena ketimpangan jumlah dimensi ($128 \gg 10$) dan perbedaan sifat fitur (*supervised latent representation* vs *raw feature*).
   - Oleh karena itu, kita harus merancang eksperimen kontrol di mana **embedding dibandingkan dengan embedding yang setara**, dan **dimensi disetarakan ($10\text{-d}$ vs $10\text{-d}$)** di dalam XGBoost!
2. **Koreksi Aktifasi Bobot Sampler pada `code/exp21d/samplers.py` (*Sampler Correctness Gate*)**:
   - Audit kode menunjukkan bahwa pada [`code/exp21d/samplers.py`](file:///d:/Projects/Klasifikasi%20Transasi%20Kartu%20Kredit%20-%20GraphSAGE/tugas-akhir-klasifikasi-transaksi-kartu-kredit/code/exp21d/samplers.py#L99-L106), pemanggilan `NeighborLoader` belum menyertakan argumen `weight_attr="edge_weight"`, dan perhitungan derajat sumber pada sisi `(transaction, rev_buys, user)` masih bernilai konstan 1 karena setiap transaksi hanya terhubung ke 1 user dan 1 merchant.
   - Dengan mengadopsi perbaikan *counterpart degree* dan `weight_attr="sampling_weight"` (sebagaimana telah diterapkan di [`code/exp21e_v5/samplers.py`](file:///d:/Projects/Klasifikasi%20Transasi%20Kartu%20Kredit%20-%20GraphSAGE/tugas-akhir-klasifikasi-transaksi-kartu-kredit/code/exp21e_v5/samplers.py)), perbedaan antara `Uniform`, `Topology-Aware`, dan `Importance-Based` pada `EXP21D` (`[15, 10]`) akan menjadi **100% murni, kausal, dan terverifikasi secara matematis**.
3. **Membedah Peran Sinergis Stage-1 (GraphSAGE `[15, 10]`) dan Stage-2 (XGBoost)**:
   - Pada `EXP21D v1`, `Topology-Aware` meraih **AUPRC tertinggi (`0.9746`)** dan `Importance-Based` meraih **AUPRC `0.9634`** dengan **Recall `100.00%` (0 FN)**, mengalahkan `Uniform` (`AUPRC 0.9550`, `1 FN`), sementara `Tabular Only` kehilangan **38 kasus fraud (`Recall 90.97%`)**. Kita perlu mengeksplorasi *bagaimana* XGBoost memanfaatkan embedding dari ketiga sampler tersebut untuk menyelamatkan ke-38 transaksi fraud yang lolos.

---

## 2. Peta 6 Pilar Eksplorasi Lanjutan untuk `EXP21D` (`num_neighbors = [15, 10]`)

Berikut adalah **6 Pilar Eksplorasi** yang dirancang khusus untuk arsitektur **Hybrid GraphSAGE + XGBoost (`EXP21D`, hop `[15, 10]`)**:

---

### Pilar 1: Eksplorasi Kesetaraan Representasi & Dimensi pada XGBoost (*Fair Feature & Embedding Comparison* — Menjawab Kritik Pembimbing)
Untuk menggantikan argumen *Split Gain Share 99.6%* yang dinilai tidak setara oleh dosen pembimbing, kita melakukan **3 eksperimen kontrol yang 100% setara (Apple-to-Apple)** di atas pengklasifikasi XGBoost yang sama:

1. **Eksperimen 1A — `MLP-Embedding + XGBoost` vs `GraphSAGE-Embedding + XGBoost` (128-d vs 128-d Setara)**:
   - Melatih jaringan saraf tabular (`Tabular MLP` 2-layer 128-d tanpa graf) dengan fungsi loss *Damped Weighted BCE* ($w_+ = 27.06$) yang identik dengan tahap pertama `HeteroGraphSAGE`.
   - Mengekstrak **128-d embedding laten dari `Tabular MLP`** lalu melatih **XGBoost** di atasnya (`XGBoost on 128-d MLP Embedding` & `Hybrid 10-d Tabular + 128-d MLP Embedding`).
   - Membandingkannya melawan **`XGBoost on 128-d GraphSAGE Embedding` (`Uniform`, `Topology`, `Importance`)**.
   - *Nilai Ilmiah*: Sekarang kedua pihak sama-sama memiliki **128 dimensi representasi neural yang dilatih secara supervised**! Selisih performanya di XGBoost adalah **100% kontribusi murni topologi graf & strategi sampling `[15, 10]`**, bebas dari bias dimensi maupun bias supervisi.
2. **Eksperimen 1B — Penyetaraan Dimensi via Kompresi Orthonormal (`10-d PCA GNN Embedding` vs `10-d Clean Tabular` di XGBoost)**:
   - Menerapkan *Principal Component Analysis* (PCA yang di-*fit* hanya pada data Train) untuk memampatkan 128-d embedding GraphSAGE (`Uniform`, `Topology`, `Importance`) menjadi **tepat 10 dimensi (`10-d GNN-PCA`)** — setara persis dengan jumlah **10 atribut tabular bersih (`10-d Tabular`)**.
   - Melatih XGBoost pada gabungan `10-d Tabular + 10-d GNN-PCA` (total 20 fitur berimbang 10:10) dan mengevaluasi performa serta kontribusi masing-masing blok menggunakan **Permutation Importance pada Test Set (Drop in AUPRC / F1 saat blok fitur diacak)**, bukan sekadar *in-sample tree split gain*.
3. **Eksperimen 1C — `Unsupervised / Random-Init GraphSAGE [15, 10] + XGBoost` (Isolasi Murni Struktur Topologi Tanpa Label Bocor di Embedding)**:
   - Mengekstrak 128-d fitur tetangga dari `HeteroGraphSAGE` (`[15, 10]`) dengan bobot acak tetap (*frozen untrained GNN* / *topological neighborhood smoothing*) untuk `Uniform`, `Topology`, dan `Importance`, lalu melatih XGBoost di atasnya.
   - *Nilai Ilmiah*: Karena GNN belum pernah melihat label `Is Fraud?` sama sekali saat ekstraksi fitur, embedding 128-d ini murni merupakan ringkasan struktural tetangga 2-hop `[15, 10]`. Ini menjawab tuntas kekhawatiran dosen pembimbing bahwa embedding GNN "sudah memuat label".

---

### Pilar 2: Koreksi *Active Weighted Sampler* (`EXP21D-v4`) & Audit Empiris Subgraf `[15, 10]`
Memindahkan perbaikan kausal dan matematis dari `EXP21E-v3/v5` ke dalam pipeline `EXP21D` (`[15, 10]`):
1. **Perbaikan `code/exp21d/samplers.py` (Active `weight_attr` & Counterpart Degree)**:
   - Saat mencuplik transaksi kandidat pada hop ke-2 melalui simpul `user` (`transaction -> rev_buys -> user`), bobot kandidat dihitung dari **derajat historis merchant** tempat transaksi kandidat tersebut berlangsung ($\text{deg}(m_{\text{cand}})$).
   - Saat mencuplik transaksi kandidat melalui simpul `merchant` (`transaction -> rev_sells -> merchant`), bobot kandidat dihitung dari **derajat historis user** ($\text{deg}(u_{\text{cand}})$).
   - Memastikan `NeighborLoader(..., weight_attr="sampling_weight")` aktif.
2. **Audit Distribusi Tetangga Tercuplik (*Sampled Neighborhood Audit* pada Hop `[15, 10]`)**:
   - Mengukur secara empiris perbedaan subgraf yang terambil oleh `Uniform`, `Topology-Aware`, dan `Importance-Based` pada fan-out `[15, 10]`:
     - Rata-rata & kuartil derajat tetangga yang terpilih di Hop-1 ($k_1=15$) dan Hop-2 ($k_2=10$).
     - Persentase tumpang tindih sisi (*Jaccard Edge Overlap*) antara `Uniform` vs `Topology` vs `Importance`.
     - Proporsi *homofili efektif* (kepadatan sinyal fraud di dalam subgraf `[15, 10]` yang tercuplik) — dianalisis secara *post-hoc*.

---

### Pilar 3: Bedah Sinergi Dua Tahap (*Stage-1 GNN Head* vs *Stage-2 XGBoost Classifier*) & Ruang Laten 128-d
Mengapa menggunakan arsitektur hibrida `GraphSAGE + XGBoost`? Pilar ini membuktikan secara kuantitatif bagaimana Stage-1 dan Stage-2 saling melengkapi pada hop `[15, 10]`:
1. **Evaluasi Berdampingan *Stage-1 (Direct GNN Head)* vs *Stage-2 (Downstream XGBoost)* pada Checkpoint yang Sama**:
   - Pada setiap run `EXP21D` (`Uniform`, `Topology`, `Importance`), sebelum membuang kepala klasifikasi linier GNN, kita menyimpan hasil prediksi langsung dari `HeteroGraphSAGE` (*Stage-1*), lalu membandingkannya dengan hasil prediksi `XGBoost` (*Stage-2* pada `GNN-Only 128-d` dan `Hybrid 138-d`).
   - *Output*: Tabel langsung yang menunjukkan **Boost Lift dari XGBoost** ($\Delta \text{AUPRC}$ dan pengurangan *False Positive* dari ~80–104 FP pada GNN murni menjadi ~45–49 FP pada Hybrid XGBoost!).
2. **Analisis Separabilitas Ruang Laten 128-d (Input ke XGBoost)**:
   - Menghitung **Inter-Class Centroid Distance** (Cosine & Euclidean), **Fisher's Discriminant Ratio (FDR)**, dan memvisualisasikan **t-SNE 2D** dari embedding 128-d yang diumpankan ke XGBoost untuk ketiga sampler.

---

### Pilar 4: Analisis Terstratifikasi Derajat Simpul (*Degree-Stratified*) & *Fraud Rescue Analysis* pada Hop `[15, 10]`
Mengeksplorasi **di mana** ketiga strategi *neighbor sampling* dan model hibrida bekerja paling efektif pada `283.529` transaksi uji (`421` fraud):
1. **Stratifikasi Berdasarkan Kuartil Derajat Merchant & User (*Low-Degree* $\le 15$, *Medium-Degree* $16\text{–}200$, *Super-Hub* $> 200$)**:
   - Karena Hop-1 mengambil maksimal $k_1 = 15$ tetangga, seluruh perbedaan pencuplikan terjadi pada entitas dengan $\text{deg} > 15$, terutama pada **Super-Hub ($\text{deg} > 200$)**.
   - Memetakan F1-Score, AUPRC, *False Positive*, dan *False Negative* dari `Uniform`, `Topology-Aware`, dan `Importance-Based` pada masing-masing strata derajat.
2. **Analisis Penyelamatan Kasus Fraud (*The 38 Rescued Frauds Anatomy*)**:
   - Pada [`EXP21D v1`](file:///d:/Projects/Klasifikasi%20Transasi%20Kartu%20Kredit%20-%20GraphSAGE/tugas-akhir-klasifikasi-transaksi-kartu-kredit/result/exp21d/v1/exp21d_summary_report.md), model `Tabular-Only XGBoost` gagal mendeteksi **38 transaksi fraud** (`Recall 90.97%`, `383/421`), sedangkan `Hybrid GraphSAGE + XGBoost` dengan `Topology-Aware` dan `Importance-Based` berhasil menangkap **ke-38 transaksi fraud tersebut 100% (`421/421`, 0 FN)**, sementara `Uniform` masih meloloskan 1 kasus (`420/421`, 1 FN).
   - Kita membedah profil ke-38 transaksi fraud yang diselamatkan ini (nominal transaksi normal/kecil yang menyamar, rasio kamuflase merchant, dan konektivitas 2-hop ke transaksi fraud historis) serta 1 transaksi fraud yang terlewat oleh `Uniform` namun tertangkap oleh `Topology` & `Importance`!

---

### Pilar 5: Eksplorasi Interaksi Hiperparameter XGBoost & Agregator GNN pada Hop `[15, 10]`
Mengeksplorasi bagaimana komponen internal GNN dan pengklasifikasi XGBoost berinteraksi dengan ketiga strategi sampling pada fan-out `[15, 10]`:
1. **Sensitivitas Kompleksitas Pohon XGBoost terhadap Sampler (`max_depth` $\in \{3, 4, 6, 8\}$ & `colsample_bytree` $\in \{0.5, 0.8, 1.0\}$)**:
   - Karena pelatihan ulang XGBoost di atas embedding 128-d yang sudah diekstrak hanya membutuhkan waktu **~3–5 detik per konfigurasi**, kita dapat mengeksplorasi matriks lengkap bagaimana kedalaman pohon (`max_depth`) dan proporsi pencuplikan kolom (`colsample_bytree`) memengaruhi selisih performa antara `Uniform`, `Topology-Aware`, dan `Importance-Based`.
2. **Komparasi Pengklasifikasi Hilir Alternatif (*Downstream Classifier Benchmark* di atas Embedding `[15, 10]`)**:
   - Membandingkan `XGBoost` vs `LightGBM` vs `Random Forest` vs `Logistic Regression (Linear Probe)` di atas matriks fitur hibrida 138-d yang sama untuk membuktikan mengapa `Regularized XGBoost` adalah pasangan hilir terbaik bagi `HeteroGraphSAGE`.
3. **Ablasi Fungsi Agregator GNN (`Mean` vs `Max-Pooling` pada `EXP21D` `[15, 10]`)**:
   - Menguji apakah keunggulan `Topology-Aware` dan `Importance-Based` di dalam pipeline Hybrid GraphSAGE + XGBoost tetap bertahan ketika agregator GNN diubah dari `mean` ke `max`.

---

### Pilar 6: Validasi Statistik *Multi-Seed Paired* (`5 Seeds`) & Kurva Operasional Perbankan
Mengadopsi standar statistik ketat untuk publikasi IEEE:
1. **Matriks *Paired Multi-Seed* (`Seeds = [42, 100, 2026, 3407, 8713]`)**:
   - Menjalankan `EXP21D` (`[15, 10]`, `300d/30d/8d/60d`) untuk 3 strategi sampling $\times$ 5 *paired seeds* = 15 run agar dapat melaporkan **$\text{Mean} \pm \text{SD}$** serta **95% Confidence Interval**.
2. **Uji Signifikansi Berpasangan (McNemar's $\chi^2$ Test & Paired Bootstrap pada AUPRC/F1)**:
   - Menguji signifikansi statistik antara `Tabular-Only XGBoost` vs `Hybrid Uniform` vs `Hybrid Topology` vs `Hybrid Importance`.
3. **Kurva Kapasitas Triase Perbankan (*Recall@K* & *Precision@K* untuk $K \in \{100, 200, 300, 421, 500, 1000\}$) & Sensitivitas Threshold ($\tau \in [0.05, 0.95]$)**:
   - Memvisualisasikan efisiensi triase peringatan fraud bagi tim analis bank dan ketahanan F1-Score terhadap pergeseran threshold ($\tau^*$ vs $\tau=0.50$).

---

## 3. Matriks Tahapan Eksekusi & Prioritas Implementasi (`EXP21D`, Hop `[15, 10]`)

Berikut adalah pembagian tahapan implementasi dari yang paling cepat (bisa langsung dijalankan menggunakan artefak `EXP21D v1` yang sudah ada) hingga eksperimen pelatihan penuh:

| Tahap | Kode Modul | Cakupan Eksplorasi pada `EXP21D` (`[15, 10]`) | Butuh Training Ulang GNN? | Estimasi Waktu | Output Utama untuk Skripsi / Paper IEEE |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **Tahap 1 (Instan / Post-Hoc pada `EXP21D v1`)** | **`EXP21D-PostHoc`** | **Pilar 1B** (`10-d PCA GNN` vs `10-d Tabular` + *Permutation Importance*), **Pilar 3** (*Stage-1 GNN Head* vs *Stage-2 XGBoost* + Separabilitas Laten 128-d & t-SNE), **Pilar 4** (*Degree-Stratified Analysis* & Bedah 38 Fraud yang Diselamatkan), **Pilar 5A/5B** (Grid XGBoost & Benchmark Classifier Hilir), **Pilar 6** (Bootstrap 95% CI, McNemar, Kurva TP@K) | **Tidak** (Menggunakan checkpoint `.pt`, `.json`, & `.npz` `EXP21D v1` `[15, 10]`) | **~3 - 5 Menit** | Tabel *Equal-Dimension 10-d vs 10-d*, *Permutation Gain Drop*, Anatomi 38 Fraud Terselamatkan, Tabel *Degree-Stratified*, Grafik 300 DPI |
| **Tahap 2 (Kontrol Setara Pembimbing)** | **`EXP21D-FairCtrl`** | **Pilar 1A** (`128-d Tabular MLP Embedding + XGBoost`) & **Pilar 1C** (`128-d Untrained/Frozen GraphSAGE [15, 10] + XGBoost`) | Hanya melatih MLP & inferensi GNN tanpa training (~2 mnt) | **~2 - 3 Menit** | Tabel **Apple-to-Apple 128-d Embedding Comparison** di atas XGBoost (Menjawab telak kritik dosen pembimbing!) |
| **Tahap 3 (Koreksi Sampler & Multi-Seed)** | **`EXP21D-v4/v5`** | **Pilar 2** (Koreksi `samplers.py` dengan `weight_attr` aktif & *counterpart degree* + Audit Subgraf `[15, 10]`) & **Pilar 6A** (3 Sampler $\times$ 5 *Paired Seeds* pada `EXP21D` `[15, 10]`) | **Ya** (3 run untuk single-seed `v4`, atau 15 run untuk 5-seed `v5`) | **~30 Menit** (1 seed) / **~2.2 Jam** (5 seeds di Kaggle GPU) | Tabel Master Final `EXP21D` dengan *Active Weighted Sampler* ($\text{Mean} \pm \text{SD}$ & 95% CI) + Grafik Audit Mekanisme Sampling |
| **Tahap 4 (Ablasi Lanjutan - Opsional)** | **`EXP21D-Aggr`** | **Pilar 5C**: Komparasi `Mean` vs `Max` Aggregator pada `EXP21D` (`[15, 10]`) | **Ya** (3 run `aggr='max'`) | **~30 Menit** | Tabel Konsistensi Hierarki Sampling Lintas-Agregator |

---

## 4. Rencana Struktur File Implementasi (`code/exp21d/`)

Untuk mengimplementasikan eksplorasi di atas secara bersih tanpa merusak hasil `result/exp21d/v1`, `v2`, dan `v3` yang sudah ada:

1. **`code/exp21d/explore_exp21d_deep_analysis.py`** *(Tahap 1 & Tahap 2)*:
   - Memuat graf `300d/30d/8d/60d` (`num_neighbors = [15, 10]`) dan mengekstrak matriks fitur dari model `EXP21D v1`.
   - Menjalankan **Eksperimen 1A** (`Tabular MLP 128-d + XGBoost`), **Eksperimen 1B** (`PCA 10-d GNN + 10-d Tabular` + *Test Permutation Importance*), dan **Eksperimen 1C** (`Frozen Untrained GraphSAGE [15, 10] + XGBoost`).
   - Menjalankan **Pilar 3** (Perbandingan *Stage-1 GNN Direct* vs *Stage-2 XGBoost* serta metrik jarak *Inter-Class Centroid* & *Fisher Ratio* 128-d).
   - Menjalankan **Pilar 4** (Stratifikasi derajat Merchant/User *Low* vs *Medium* vs *Super-Hub* dan analisis 38 kasus fraud yang diselamatkan oleh GraphSAGE + XGBoost).
   - Menjalankan **Pilar 5A & 5B** (Sensitivitas hiperparameter XGBoost & perbandingan classifier hilir) serta **Pilar 6** (Uji McNemar, Bootstrap 95% CI, dan kurva operasional).
   - Menyimpan seluruh laporan JSON, tabel CSV, dan plot visualisasi 300 DPI ke `result/exp21d/exploration_v1_hops15_10/`.
2. **`code/exp21d/samplers.py` & `code/exp21d/main.py` (Upgrade ke `EXP21D-v4`)** *(Tahap 3)*:
   - Memperbarui `num_neighbors: [15, 10]` dan `train_days: 300, val_days: 30, gap_days: 8, test_days: 60` sebagai konfigurasi terkunci di [`code/exp21d/config.reference.yaml`](file:///d:/Projects/Klasifikasi%20Transasi%20Kartu%20Kredit%20-%20GraphSAGE/tugas-akhir-klasifikasi-transaksi-kartu-kredit/code/exp21d/config.reference.yaml).
   - Mengintegrasikan *counterpart degree scoring* dan `weight_attr="sampling_weight"` ke dalam [`code/exp21d/samplers.py`](file:///d:/Projects/Klasifikasi%20Transasi%20Kartu%20Kredit%20-%20GraphSAGE/tugas-akhir-klasifikasi-transaksi-kartu-kredit/code/exp21d/samplers.py) beserta dukungan `--seed` untuk eksekusi *multi-seed* di `result/exp21d/v4/`.

