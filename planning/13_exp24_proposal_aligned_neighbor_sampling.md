# Rencana Eksperimen 24 (EXP24): Implementasi Neighbor Sampling Sesuai Proposal dengan ASA-GNN Attribute Homophily & Scalable Personalized PageRank

**Status Dokumen**: Rencana Eksekusi Eksperimen  
**Referensi Utama**: [`planning/12_eksplorasi_mendalam_exp21d_hybrid_xgboost_REVISI_SAMPLER.md`](file:///d:/Projects/Klasifikasi%20Transasi%20Kartu%20Kredit%20-%20GraphSAGE/tugas-akhir-klasifikasi-transaksi-kartu-kredit/planning/12_eksplorasi_mendalam_exp21d_hybrid_xgboost_REVISI_SAMPLER.md) dan Hasil Multi-Seed EXP21D ([`NEIGHBOR_SAMPLING_KNOWLEDGE_BASE.md`](file:///d:/Projects/Klasifikasi%20Transasi%20Kartu%20Kredit%20-%20GraphSAGE/tugas-akhir-klasifikasi-transaksi-kartu-kredit/NEIGHBOR_SAMPLING_KNOWLEDGE_BASE.md))  
**Target Utama**: Menguji secara empiris formulasi *neighbor sampling* yang selaras verbatim dengan **Proposal Tugas Akhir (Bab III Subbab 3.5.2 & 3.5.3, Algoritma 3 & 4)**, dengan mengintegrasikan **Solusi 3: Attribute Homophily berbasis ASA-GNN (Tian et al., 2024)** untuk menjamin **Zero Label Leakage**, komputasi **Local Personalized PageRank (PPR)** yang skalabel, serta **Protokol Telemetri Granular Per-Epoch** (Waktu, Metrik, dan Sumber Daya).

---

## 1. Motivasi & Justifikasi Ilmiah

### 1.1 Menjembatani Proposal Tugas Akhir dan Realisasi Produksi
Pada Proposal Tugas Akhir (Muhammad Hendika Putra, Telkom University, 2026), dirancang dua metode *advanced sampling*:
1. **Topology-Aware Sampling** (Algoritma 3 Proposal): Mengombinasikan **Jaccard Similarity** $J(v, u)$ dan **Local Homophily Ratio** $H(v, u)$.
2. **Importance-Based Sampling** (Algoritma 4 Proposal): Mengombinasikan **Degree Centrality** $C_d(u)$ dan **Personalized PageRank (PPR)** $C_p(u)$.

Pada EXP21D, kedua metode diaproksimasi menggunakan proksi derajat tertutup (*closed-form log-degree proxies*: $1/\ln(1+d)$ dan $\ln(1+d)$). Meskipun proksi EXP21D berhasil membuktikan pergeseran subgraf dan memenangkan metrik AUPRC/F1, penguji atau pembimbing (Dr. Fitriyani) berhak menanyakan:
> *"Apakah model berkinerja lebih baik atau lebih buruk jika formulasi Jaccard, Homophily, dan Personalized PageRank diimplementasikan secara langsung sesuai algoritma proposal?"*

EXP24 dirancang untuk menjawab pertanyaan tersebut secara definitif melalui eksperimen komparatif yang ketat.

### 1.2 Resolusi Label Leakage via ASA-GNN (Tian et al., 2024)
Formula homofili asli pada proposal:
$$H(v, u) = \frac{|\{w \in \mathcal{N}(u) : L[w] = L[v]\}|}{|\mathcal{N}(u)| + 1}$$
mensyaratkan label transaksi target $L[v]$. Pada data uji (*unlabeled streaming transactions*), $L[v]$ belum diketahui. Menggunakan $L[v]$ pada data uji merupakan **pelanggaran kaidah ilmiah (*label leakage*)**.

Untuk menyelesaikan masalah ini tanpa mengubah esensi proposal, EXP24 mengadopsi **Solusi 3: Attribute Homophily berbasis ASA-GNN** (*Adaptive Sampling and Aggregation-Based GNN for Transaction Fraud Detection*, IEEE TCSS 2024, Ref [19] Proposal):
- **Premis ASA-GNN**: Penipu (*fraudsters*) berkamuflase dengan berinteraksi di merchant normal, namun pola atribut transaksinya (waktu transaksi, disparitas nominal amount, kanal kartu) menyimpang dari pola tetangga sekitarnya.
- **Formulasi Anti-Leakage**: Kesamaan atribut antara transaksi target $v$ dan kandidat $u$ dihitung menggunakan *Scaled Cosine Similarity* pada ruang fitur transaksi ternormalisasi:
  $$H_{\text{ASA-GNN}}(v, u) = \frac{1 + \cos(x_v, x_u)}{2} \in [0, 1]$$
- **Keunggulan Metodologis**: Tidak membutuhkan label $L[v]$ sama sekali (**100% Bebas Label Leakage**), dapat dievaluasi secara induktif pada data uji, dan secara presisi menangkap fenomena kamuflase penipu.

---

## 2. Protokol Kontrol Ketat (*Fair Comparison Against EXP21D*)

Agar atribusi performa murni mencerminkan perbedaan formulasi sampling (bukan faktor pengganggu lain), seluruh konfigurasi sistem **wajib dikunci identik dengan EXP21D-v10**:

| Komponen Eksperimen | Status Kontrol | Spesifikasi Terkunci (Identik EXP21D) |
|---|---|---|
| **Downstream Pipeline** | Dikunci | Hybrid HeteroGraphSAGE + XGBoost |
| **Dimensi Representasi** | Dikunci | 128-d GNN Latent Embedding + 10-d Clean Tabular (Total 138-d Hybrid Input) |
| **Arsitektur GNN** | Dikunci | 2-Layer HeteroGraphSAGE, Mean Aggregator, Hidden Size 256, ReLU, Batch Normalization |
| **Ukuran Fan-out** | Dikunci | Kedalaman 2-Hop: $[S_1, S_2] = [15, 10]$ |
| **Protokol Temporal Split** | Dikunci | `300d train / 30d validation / 8d embargo / 60d test` |
| **Dataset & Row Identity** | Dikunci | IBM Synthetic Credit Card (283.529 test rows, 421 fraud cases identik via SHA-256) |
| **Konfigurasi XGBoost** | Dikunci | `scale_pos_weight` otomatis, max_depth=6, n_estimators=100, learning_rate=0.1 |
| **Threshold Selection** | Dikunci | Dipilih murni dari Precision-Recall curve set validasi (tidak menyentuh test) |
| **Multi-Seed Paired Design** | Dikunci | 5 Seed Berpasangan: `[42, 100, 2026, 3407, 8713]` (Total 15 Paired Runs) |
| **Variabel Bebas Tunggal** | **DIUJI** | **Formulasi Pembobotan Sampling** (`uniform`, `topology_proposal`, `importance_proposal`) |

---

## 3. Rincian Formulasi Matematis Strategi Sampling EXP24

Pengambilan sampel dilakukan pada **Hop Kedua** (kompetisi kandidat transaksi di bawah node User dan Merchant):

```mermaid
flowchart TD
    Root["Root Transaksi v (Fitur x_v)"] -->|Hop 1: d=1, w=1.0| Hop1["User & Merchant Node"]
    Hop1 -->|Hop 2: Kandidat Transaksi u (Fitur x_u)| Sampler{Metode Sampling EXP24}
    
    Sampler -->|1. Uniform| W_Uni["w(v, u) = 1.0"]
    
    Sampler -->|2. Topology-Aware Proposal| W_Topo["Jaccard: J(v, u)<br/>Homophily ASA-GNN: H(v, u)<br/>w(v, u) = 0.5 * J(v, u) + 0.5 * H(v, u)"]
    
    Sampler -->|3. Importance-Based Proposal| W_Imp["Degree Centrality: Cd(u)<br/>Local PPR: Cp(u)<br/>w(v, u) = 0.3 * Cd(u) + 0.7 * Cp(u)"]
    
    W_Topo --> Floor["Smoothing Floor & Stabilisasi:<br/>w = (raw - min)/(max - min) + 0.1"]
    W_Imp --> Floor
    
    Floor --> Loader["PyG NeighborLoader(weight_attr='sampling_weight')"]
    W_Uni --> Loader
```

### 3.1 Metode 1: Uniform Random Sampling (`uniform`)
Baseline standar GraphSAGE (Hamilton et al., 2017) dan EXP21D:
$$w(v, u) = 1.0, \quad P(u \mid \text{perantara}) = \frac{1}{|\mathcal{C}|}$$

---

### 3.2 Metode 2: Topology-Aware Sampling Proposal-ASA (`topology_proposal`)

Mengimplementasikan **Algoritma 3 Proposal** dengan resolusi ASA-GNN:
$$w(v, u) = \alpha \cdot J(v, u) + (1 - \alpha) \cdot H_{\text{ASA-GNN}}(v, u), \quad \alpha = 0.5$$

#### A. Jaccard Similarity Bipartit 2-Hop ($J(v, u)$)
Mengukur tingkat tumpang tindih lingkungan struktural antara transaksi target $v$ dan transaksi kandidat $u$.
- Pada kanal **via User** (kandidat $u$ dan target $v$ memiliki user yang sama):
  $$J_{\text{user\_chan}}(v, u) = \frac{|\mathcal{N}_2(v) \cap \mathcal{N}_2(u)|}{|\mathcal{N}_2(v) \cup \mathcal{N}_2(u)|}$$
  di mana $\mathcal{N}_2(\cdot)$ adalah himpunan merchant yang pernah dikunjungi oleh entitas terkait. Jika user melakukan transaksi di merchant yang sering dikunjungi bersama dalam riwayat historisnya, nilai Jaccard tinggi.
- Nilai $J(v, u) \in [0, 1]$.

#### B. Local Homophily Ratio ASA-GNN ($H_{\text{ASA-GNN}}(v, u)$)
Mengukur kedekatan pola perilaku transaksi antara target $v$ dan kandidat $u$ pada ruang atribut bersih 10-dimensi ($x_v$ dan $x_u$):
$$\cos(x_v, x_u) = \frac{x_v^T x_u}{\|x_v\|_2 \|x_u\|_2 + \epsilon}$$
$$H_{\text{ASA-GNN}}(v, u) = \frac{1 + \cos(x_v, x_u)}{2} \in [0, 1]$$
- **Rasional Anti-Kamuflase**: Jika transaksi target $v$ adalah penipuan yang mencoba menyamar pada merchant normal, transaksi normal di sekitarnya akan memiliki pola atribut berbeda ($H$ rendah). Sebaliknya, transaksi dengan karakteristik yang mirip dan relevan akan diprioritaskan untuk diagregasi.
- **Jaminan Anti-Leakage**: 100% bebas dari label target $L[v]$ maupun label kandidat $L[u]$.

#### C. Stabilisasi dan Normalisasi Lokal
$$\text{raw}(v, u) = 0.5 \cdot J(v, u) + 0.5 \cdot H_{\text{ASA-GNN}}(v, u)$$
$$w(v, u) = \frac{\text{raw}(v, u) - \min(\mathbf{raw})}{\max(\mathbf{raw}) - \min(\mathbf{raw}) + 10^{-8}} + 0.1$$

---

### 3.3 Metode 3: Importance-Based Sampling Proposal-PPR (`importance_proposal`)

Mengimplementasikan **Algoritma 4 Proposal** dengan optimasi skalabilitas:
$$w(v, u) = \gamma \cdot C_d(u) + (1 - \gamma) \cdot C_p(u), \quad \gamma = 0.3$$

#### A. Degree Centrality ($C_d(u)$)
Mengukur sentralitas konektivitas entitas pasangan (*counterpart entity*):
- Pada kanal via User (kandidat $u \to \text{User}$): $C_d(u) = \frac{d_{\text{merchant}}^{(u)}}{|V_{\text{merchant}}| - 1}$
- Pada kanal via Merchant (kandidat $u \to \text{Merchant}$): $C_d(u) = \frac{d_{\text{user}}^{(u)}}{|V_{\text{user}}| - 1}$
- Derajat dihitung strictly secara kumulatif historis ($t \le t_{\text{split}}$) untuk mencegah *temporal leakage*.

#### B. Local Personalized PageRank ($C_p(u)$)
Menghitung pengaruh topologi kandidat terhadap transaksi target $v$ menggunakan propagasi $r$-langkah ($r = 3$) dengan faktor teleportasi $\beta = 0.15$:
$$\pi_v^{(0)} = e_v$$
$$\pi_v^{(t)} = (1 - \beta) \tilde{A}_{\text{local}} \pi_v^{(t-1)} + \beta e_v, \quad t = 1, \dots, r$$
$$C_p(u) = \pi_v^{(r)}[u]$$

##### Arsitektur Komputasi Cepat & Skalabel:
Untuk mencegah ledakan komputasi pada 24 juta node:
1. PPR tidak dihitung pada graf global, melainkan dihitung secara lokal pada **Subgraf Ego 2-Hop** dari target batch $v$.
2. Matriks ketetanggaan lokal $\tilde{A}_{\text{local}}$ dinormalisasi secara simetris ($D^{-1/2} A D^{-1/2}$) dan dieksekusi via *Sparse Matrix Multiplication* di PyTorch GPU.
3. Karena ukuran subgraf 2-hop lokal per batch hanya berkisar ratusan hingga ribuan node, 3 langkah propagasi selesai dalam **< 10 milidetik per batch**.

#### C. Stabilisasi dan Normalisasi Lokal
$$\text{raw}(v, u) = 0.3 \cdot C_d(u) + 0.7 \cdot C_p(u)$$
$$w(v, u) = \frac{\text{raw}(v, u) - \min(\mathbf{raw})}{\max(\mathbf{raw}) - \min(\mathbf{raw}) + 10^{-8}} + 0.1$$

---

## 4. Protokol Telemetri & Pencatatan Rinci Per-Epoch (*Comprehensive Logging Protocol*)

Untuk memastikan akuntabilitas ilmiah penuh, analisis konvergensi, serta perbandingan latensi komputasi yang presisi (sesuai Bab III Subbab 3.6 & 3.7 Proposal), EXP24 menerapkan **pencatatan data tingkat tinggi (*high-resolution telemetry*) pada setiap epoch**.

### 4.1 Metrik yang Wajib Dicatat Per Epoch

Setiap selesai satu epoch (pelatihan dan validasi), skrip akan mengalkulasi dan mencatat metrik berikut:

1. **Metrik Pelatihan (Training Progress)**:
   - `epoch`: Nomor iterasi epoch saat ini ($1, \dots, E_{\max}$).
   - `train_loss`: Weighted Binary Cross-Entropy Loss rata-rata per sampel.
   - `train_acc`: Akurasi prediksi biner pada kepala transaksi training batch (threshold 0.5).
   - `train_tx_count`: Total transaksi yang diproses selama training epoch tersebut.

2. **Metrik Validasi (Validation Evaluation)**:
   - `val_loss`: Unweighted Binary Cross-Entropy Loss pada seluruh set validasi (sebagai ukuran *generalization loss* sejati).
   - `val_auprc`: Area Under the Precision-Recall Curve (Average Precision score) — **Metrik Utama Early Stopping**.
   - `val_roc_auc`: Area Under the ROC Curve.
   - `val_f1_default`: F1-Score menggunakan ambang batas default 0.5.
   - `val_f1_optimal`: F1-Score tertinggi yang diperoleh dari pencarian ambang batas adaptif (*optimal threshold tuning* pada kurva PR validasi).
   - `val_precision_optimal`: Precision yang berkorespondensi dengan `val_f1_optimal`.
   - `val_recall_optimal`: Recall yang berkorespondensi dengan `val_f1_optimal`.
   - `val_optimal_threshold`: Nilai threshold optimal yang terpilih (misal 0.42 atau 0.58).
   - `val_fp_optimal`: Jumlah *False Positive* pada threshold optimal di set validasi.
   - `val_fn_optimal`: Jumlah *False Negative* pada threshold optimal di set validasi.

3. **Status Optimizer & Regularisasi**:
   - `learning_rate`: Nilai learning rate aktual dari optimizer Adam pada epoch tersebut (memantau penurunan LR akibat scheduler `ReduceLROnPlateau`).
   - `patience_counter`: Status counter early stopping saat ini terhadap batas maksimal `early_stopping_patience = 7`.
   - `is_best_epoch`: Penanda boolean (`true`/`false`) jika epoch tersebut menghasilkan rekor `val_auprc` tertinggi baru.

---

### 4.2 Pencatatan Waktu & Profiling Latensi Per Epoch (*Execution Timing*)

Setiap epoch diprofiling secara modular menggunakan modul `time.perf_counter()` presisi tinggi untuk mengukur waktu secara terisolasi:

1. **`time_train_epoch_sec`**:
   - Total durasi waktu dinding (*wall-clock time*) yang dibutuhkan untuk menyelesaikan seluruh mini-batch training pada epoch tersebut (termasuk transfer data ke GPU, forward pass, loss computation, backward pass, gradient clipping, dan optimizer step).

2. **`time_sampling_overhead_sec`**:
   - Waktu yang dihabiskan secara khusus oleh *loader/sampler* untuk menghasilkan subgraf tetangga, menghitung bobot ASA-GNN cosine similarity, dan komputasi Local PPR.
   - **Tujuan Khusus**: Mengukur secara empiris overhead latensi algoritma proposal dibandingkan Uniform baseline.

3. **`time_val_epoch_sec`**:
   - Durasi waktu inferensi evaluasi validasi tanpa gradien (`torch.no_grad()`), pengumpulan probabilitas, dan kalkulasi seluruh metrik evaluasi.

4. **`time_epoch_total_sec`**:
   - Penjumlahan: `time_train_epoch_sec + time_val_epoch_sec`.

5. **`cumulative_runtime_sec`**:
   - Akumulasi total waktu berjalan sejak epoch 1 hingga epoch saat ini.

6. **Latensi Inferensi Standar Proposal (Halaman 35 Proposal)**:
   - `val_latency_ms_per_1k_tx`: Waktu yang dibutuhkan untuk melakukan inferensi per 1.000 transaksi validasi (dalam milidetik):
     $$\text{Latency}_{1k} = \frac{\text{time\_val\_epoch\_sec}}{\text{total\_val\_tx}} \times 1.000 \times 1.000$$

---

### 4.3 Monitoring Sumber Daya Komputasi & Hardware Per Epoch

Untuk memverifikasi efisiensi memori dan mendeteksi potensi *memory leak*:
- `peak_gpu_vram_mb`: Alokasi memori VRAM GPU tertinggi selama epoch berlangsung, diambil via `torch.cuda.max_memory_allocated() / (1024**2)`.
- `peak_ram_mb`: Penggunaan RAM sistem (Resident Set Size) dari proses utama dan child-process, diambil via `psutil.Process().memory_info().rss / (1024**2)`.

---

### 4.4 Format Penyimpanan Artefak Telemetri

Data telemetri disimpan dalam **tiga format redundan** untuk menjamin keamanan data dan kemudahan visualisasi:

1. **`training_history.csv` (Tabular Format)**:
   - Berkas CSV per baris epoch. Sangat memudahkan pembuatan kurva loss, AUPRC, dan waktu menggunakan `pandas` atau `seaborn` saat menyusun Bab 4 Skripsi.
   - Kolom:
     ```text
     epoch,train_loss,train_acc,val_loss,val_auprc,val_roc_auc,val_f1_optimal,val_precision_optimal,val_recall_optimal,val_optimal_threshold,val_fp,val_fn,learning_rate,time_train_sec,time_sampling_sec,time_val_sec,time_total_sec,cumulative_sec,val_latency_ms_1k,peak_vram_mb,peak_ram_mb
     ```

2. **`training_history.json` (Structured Hierarchical Dictionary)**:
   - Berkas JSON yang memuat seluruh list deret metrik, konfigurasi hyperparameter awal, metadata seed, serta rekor checkpoint terbaik (`best_epoch`, `best_val_auprc`).

3. **`events.jsonl` (Append-Only Event Stream)**:
   - Log baris-per-baris dengan timestamp ISO 8601 mikrodetik untuk setiap kejadian penting:
     ```json
     {"timestamp": "2026-10-20T07:15:32.124Z", "event": "epoch_completed", "method": "topology_proposal", "seed": 42, "epoch": 1, "train_loss": 0.4215, "val_auprc": 0.9412, "val_f1": 0.9210, "time_epoch_sec": 42.15, "peak_vram_mb": 3120.4}
     {"timestamp": "2026-10-20T07:15:32.126Z", "event": "checkpoint_saved", "epoch": 1, "val_auprc": 0.9412}
     ```

4. **Telemetri Khusus Downstream XGBoost (`xgboost_training_history.json`)**:
   - Selain model GNN GraphSAGE, fitting model downstream XGBoost juga mencatat logloss dan AUPRC untuk setiap boosting round (`1` s.d. `100`), beserta total waktu pelatihan XGBoost dan waktu inferensi akhir pada 283.529 transaksi test.

---

## 5. Rencana Kerja dan Struktur File Eksperimen

Pipeline EXP24 akan ditempatkan pada folder terisolasi: `code/exp24_proposal_sampler/` dengan output di `result/exp24_proposal_sampler/`.

```text
code/exp24_proposal_sampler/
├── config.reference.yaml          # Konfigurasi pipeline terkunci identik EXP21D
├── config.kaggle.yaml             # Konfigurasi eksekusi Kaggle GPU
├── data_preprocessing.py          # Preprocessing & temporal splitting identik
├── graph_builder.py               # Konstruksi HeteroData & anti-leakage scalers
├── samplers_proposal.py           # CORE: Implementasi Uniform, Topology-ASA, Importance-PPR
├── validate_sampler_exp24.py      # Gate kelayakan 8-titik sebelum training
├── model.py                       # Arsitektur Hybrid HeteroGraphSAGE + XGBoost identik
├── run_seed.py                    # Eksekusi paired multi-seed (5 seed) dengan resource & telemetry monitor
├── aggregate_exp24.py             # Agregasi metrik, bootstrap, & perbandingan thd EXP21D
├── create_notebook.py             # Builder notebook otomatis untuk Kaggle
└── README.md                      # Dokumentasi teknis & petunjuk eksekusi
```

---

## 6. Protokol Verifikasi Integritas (*Correctness Gates*)

Sebelum menjalankan 15 training run multi-seed yang membutuhkan kuota GPU, skrip [`validate_sampler_exp24.py`](file:///d:/Projects/Klasifikasi%20Transasi%20Kartu%20Kredit%20-%20GraphSAGE/tugas-akhir-klasifikasi-transaksi-kartu-kredit/) wajib meloloskan 8 pengujian otomatis:
1. **Zero Label Leakage Check**: Memverifikasi secara asersi bahwa modul sampling tidak menerima tensor label $Y$, dan permutasi label target tidak mengubah bobot sampling sama sekali.
2. **ASA-GNN Range & Gradient**: Memverifikasi nilai $H_{\text{ASA-GNN}} \in [0, 1]$ dan bernilai $1.0$ jika $x_v = x_u$.
3. **Jaccard Range**: Memverifikasi nilai Jaccard $J(v, u) \in [0, 1]$ pada struktur subgraf bipartit.
4. **PPR Convergence**: Memverifikasi nilai $\sum \pi_v \approx 1.0$ dan simpul yang lebih dekat/terkoneksi kuat ke target memiliki skor PPR lebih tinggi.
5. **Hop 1 Deterministic Uniformity**: Memverifikasi bobot pada Hop 1 (`user -> tx` dan `merchant -> tx`) tetap bernilai tepat $1.0$.
6. **Finite & Non-Zero Weights**: Memverifikasi seluruh bobot $w \in [0.1, 1.1]$ dan tidak terdapat nilai NaN/Inf.
7. **Loader Weight Attribute Contract**: Memverifikasi `NeighborLoader` PyG menerima dan mengaktifkan argumen `weight_attr="sampling_weight"`.
8. **Row Identity Consistency**: Memverifikasi array identitas transaksi uji identik secara SHA-256 dengan `test_row_identity.npz` milik EXP21D.

---

## 7. Hipotesis Riset & Pertanyaan Ilmiah (EXP24 vs EXP21D)

Eksperimen ini akan menguji secara formal:

### RQ1: Apakah Integrasi Atribut (ASA-GNN) Mengungguli Proksi Topologi Murni?
- **Hipotesis (H1)**: Pada Topology-Aware Sampling, menggabungkan kesamaan atribut transaksi $H_{\text{ASA-GNN}}$ dengan Jaccard akan menghasilkan F1-Score dan Precision yang lebih tinggi daripada proksi derajat murni $1/\ln(1+d)$ EXP21D, karena model mampu membedakan transaksi anomali di antara riwayat belanja normal pada entitas yang sama.

### RQ2: Apakah Personalized PageRank Lokal Meningkatkan Stabilitas AUPRC?
- **Hipotesis (H2)**: Pada Importance-Based Sampling, penggunaan skor Personalized PageRank lokal $C_p(u)$ akan menurunkan standar deviasi AUPRC dibandingkan EXP21D, karena bobot transaksi kandidat mencerminkan kedekatan transmisi acak (*random walk reachability*) aktual ke transaksi target.

### RQ3: Komparasi Efisiensi Komputasi (*Trade-Off Latensi Inferensi*)
- **Hipotesis (H3)**: Penambahan komputasi cosine similarity (ASA-GNN) dan propagasi PPR 3-langkah akan menambah runtime sampling sekitar 5–15%, namun tetap berada dalam batas toleransi kelayakan inferensi batch.

---

## 8. Rencana Analisis Hasil & Komparasi Matriks Skripsi

Setelah 15 run selesai, output EXP24 akan dibandingkan secara *head-to-head* dengan EXP21D dalam tabel utama Bab 4 Skripsi:

| Metode Sampling | Versi Formulasi | AUPRC (Primary) | F1-Score | Precision | Recall | False Positive | Latensi (ms / 1k tx) | Waktu / Epoch |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Uniform** | Standard Baseline | *(dari EXP21D)* | *(dari EXP21D)* | *(dari EXP21D)* | *(dari EXP21D)* | 51,0 | *[Evaluated]* | ~Baseline |
| **Topology-Aware** | Proksi Derajat (EXP21D) | 0,9614 ± 0,0153 | 0,9433 ± 0,0054 | 0,8928 ± 0,0097 | 1,0000 ± 0,0000 | 50,6 | *[Evaluated]* | ~1.02x |
| **Topology-Aware** | **Proposal + ASA-GNN (EXP24)** | *[Evaluated]* | *[Evaluated]* | *[Evaluated]* | *[Evaluated]* | *[Evaluated]* | *[Evaluated]* | *[Evaluated]* |
| **Importance-Based** | Proksi Derajat (EXP21D) | 0,9673 ± 0,0059 | 0,9471 ± 0,0036 | 0,9006 ± 0,0064 | 0,9986 ± 0,0013 | 46,4 | *[Evaluated]* | ~1.02x |
| **Importance-Based** | **Proposal + Local PPR (EXP24)** | *[Evaluated]* | *[Evaluated]* | *[Evaluated]* | *[Evaluated]* | *[Evaluated]* | *[Evaluated]* | *[Evaluated]* |

### Stop / Go Criteria:
1. Jika EXP24 berhasil meningkatkan AUPRC atau menekan FP lebih lanjut dengan perbedaan signifikan ($p < 0.05$ via paired bootstrap), maka formulasi proposal dengan ASA-GNN menjadi **rekomendasi final model Tugas Akhir**.
2. Jika EXP24 memberikan performa setara dengan EXP21D, temuan ini membuktikan bahwa **proksi derajat tertutup EXP21D adalah aproksimasi komputasi yang sangat efisien dan cukup representatif** untuk menggantikan komputasi kompleks PPR dan Jaccard pada graf transaksi berskala besar. Kedua skenario memberikan nilai akademik yang sangat kuat dan defensible untuk sidang skripsi.
