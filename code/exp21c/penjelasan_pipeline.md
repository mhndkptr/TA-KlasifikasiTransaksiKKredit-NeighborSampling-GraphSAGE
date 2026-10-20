# Penjelasan Pipeline Eksperimen 21C (EXP21C)
## *Multi-Window Temporal Splitting, Walk-Forward Validation, & Embargo Gap Ablation*

---

## 1. Latar Belakang & Konteks Metodologis

Pada eksperimen **EXP21**, model menghasilkan metrik yang tampak "terlalu sempurna" (*too good to be true*): Recall 100%, F1 0.94, Precision 88.63%. Berdasarkan arahan pembimbing pada dokumen perencanaan [07_eksplorasi_data_splitting_atribut_tuning_exp21.md](file:///d:/Projects/Klasifikasi%20Transasi%20Kartu%20Kredit%20-%20GraphSAGE/tugas-akhir-klasifikasi-transaksi-kartu-kredit/planning/07_eksplorasi_data_splitting_atribut_tuning_exp21.md), dilakukan audit mendalam yang membuktikan bahwa:
1. Pohon XGBoost sekadar menghafal ID kuasi sintetis (`Zip` dan `Merchant City`) dengan total gain >99,9%, menenggelamkan kontribusi representasi graf GraphSAGE.
2. Penskalasi dan kamus kategori di-fit pada seluruh baris data sebelum partisi (kebocoran global).
3. Evaluasi hanya bertumpu pada **satu jendela waktu 60 hari** (5 Des 2018 s/d 3 Feb 2019), sehingga tidak membuktikan apakah performa model stabil terhadap *concept drift* atau sekadar "beruntung" di satu periode waktu tertentu.

### Evolusi Rangkaian Eksperimen Remediasi EXP21:
* **EXP21A (Pilar II: Clean Attributes)**: Mengeliminasi quasi-identifier (`Zip` dan `Merchant City` dihapus) serta menerapkan *Strict Out-of-Sample Preprocessing*.
* **EXP21B (Pilar III: Balanced Tuning)**: Melakukan tuning hiperparameter berimbang (`scale_pos_weight`, kedalaman pohon, regularisasi daun) untuk memulihkan Precision dan memberikan ruang kontribusi bagi embedding GNN.
* **EXP21C (Eksperimen Saat Ini - Pilar I: Robust Temporal Splitting)**:
  > **Tujuan Utama EXP21C**: Merealisasikan evaluasi **Walk-Forward Validation pada 3 Jendela Waktu Bergulir (Multi-Horizon Rolling Windows)** di era modern (2016–2020), menguji **Ablasi Durasi Embargo Gap (SPLIT-3)**, dan membuktikan konsistensi **Nilai Tambah (Lift) Representasi GraphSAGE** lintas waktu untuk laporan Bab 4 Skripsi.

---

## 2. Inovasi & Desain Metodologis EXP21C (Pilar I)

EXP21C secara komprehensif mengintegrasikan empat protokol data splitting ilmiah:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                PROTOKOL PILAR I EXP21C                                │
├───────────┬──────────────────────────────────┬─────────────────────────────────────────┤
│ Kode Opsi │ Nama Protokol                    │ Implementasi Teknis pada EXP21C         │
├───────────┼──────────────────────────────────┼─────────────────────────────────────────┤
│ SPLIT-1   │ Strict Out-of-Sample Guard       │ StandardScaler & kamus kategori HANYA   │
│           │                                  │ di-fit pada Train Window masing-masing. │
│           │                                  │ Nilai baru di Val/Test dipetakan ke OOV.│
├───────────┼──────────────────────────────────┼─────────────────────────────────────────┤
│ SPLIT-2   │ Multi-Horizon Rolling Windows    │ Walk-Forward Validation pada 3 Horizon: │
│           │ (Walk-Forward Validation)        │ • Window 1: 2017–2018 (T0: 2017-01-01)  │
│           │                                  │ • Window 2: 2018–2019 (T0: 2018-01-01)  │
│           │                                  │ • Window 3: 2019–2020 (T0: 2019-01-01)  │
│           │                                  │ Menghasilkan agregasi metrik (μ ± σ).   │
├───────────┼──────────────────────────────────┼─────────────────────────────────────────┤
│ SPLIT-3   │ Ablasi Durasi Embargo Gap        │ Variasi jeda investigasi fraud pada W2: │
│           │                                  │ • 0 Hari  : Deteksi langsung/instan     │
│           │                                  │ • 8 Hari  : Standar protokol SHINE      │
│           │                                  │ • 30 Hari : Siklus billing chargeback   │
├───────────┼──────────────────────────────────┼─────────────────────────────────────────┤
│ SPLIT-4   │ Strictly Past Inductive Scaling  │ Agregasi fitur node User & Merchant     │
│           │                                  │ diskalakan menggunakan scaler historis  │
│           │                                  │ Train tanpa membocorkan masa depan.     │
└───────────┴──────────────────────────────────┴─────────────────────────────────────────┘
```

---

## 3. Rincian 3 Jendela Waktu Bergulir (Walk-Forward Horizon)

Seluruh jendela waktu mengadopsi durasi tetap berstandar internasional (*SHINE FRAUD II Protocol*):
* **Training Duration**: 300 hari
* **Validation Duration**: 30 hari
* **Embargo Gap**: 8 hari (standar)
* **Test Duration**: 60 hari
* **Total Horizon per Window**: 398 hari

```
Window 1 (2017–2018):
[──── Train: 300d ────][─ Val: 30d ─][Gap: 8d][── Test: 60d ──]
2017-01-01         2017-10-28    2017-11-27 2017-12-05     2018-02-03

Window 2 (2018–2019) — Baseline EXP21 Horizon:
                    [──── Train: 300d ────][─ Val: 30d ─][Gap: 8d][── Test: 60d ──]
                    2018-01-01         2018-10-28    2018-11-27 2018-12-05     2019-02-03

Window 3 (2019–2020):
                                        [──── Train: 300d ────][─ Val: 30d ─][Gap: 8d][── Test: 60d ──]
                                        2019-01-01         2019-10-28    2019-11-27 2019-12-05     2020-02-03
```

---

## 4. Alur Pipeline Eksekusi (7 Phase)

Pipeline EXP21C berjalan secara modular dan terotomatisasi penuh:

### Phase 0: One-Time Modern Era Loading
* File transaksi IBM dimuat dan difilter untuk era modern ($\text{Year} \ge 2016$) **hanya satu kali** ke memori.
* Waktu parsing dan pembersihan datetime dieksekusi sekali, menghemat waktu ~70% dibanding membaca ulang CSV per-window.

### Phase 1: Iterasi Multi-Window Walk-Forward
Untuk setiap jendela waktu $w \in \{\text{Window 1, Window 2, Window 3}\}$:
1. **Temporal Slicing**: Memotong transaksi secara strictly kronologis.
2. **Strict Anti-Leakage Preprocessing (SPLIT-1)**:
   - Standarisasi `Amount` di-fit hanya pada Train $w$.
   - Kamus nilai unik kategorikal (`Use Chip`, `Merchant State`, `MCC`, `Errors?`) dipelajari dari Train $w$.
   - Kategori baru yang belum pernah muncul di Train $w$ dipetakan ke token `<oov>` (index 0).
   - Fitur quasi-identifier (`Zip` dan `Merchant City`) dieksklusikan.
   - Ditambahkan 4 fitur waktu siklik (`hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`).
3. **Pembangunan Graf Induktif Tripartit**:
   - Membangun `HeteroData` PyG untuk Train, Val, dan Test.
   - Scaler agregasi User & Merchant di-fit pada Train $w$ dan digunakan mentransformasikan Val & Test (SPLIT-4).

### Phase 2: Pelatihan GraphSAGE Encoder per Window
* Model 2-layer `HeteroGraphSAGE` (`hidden_channels = 128`, `aggr = 'mean'`).
* Dilatih dengan Adam optimizer dan early stopping berbasis Validation AUCPR.

### Phase 3: Ekstraksi Representasi Induktif (128-d GNN + Clean Tabular)
* Representasi laten 128-dimensi diekstrak untuk transaksi Train, Val, dan Test.
* Digabungkan dengan fitur tabular bersih menjadi masukan classifier hilir.

### Phase 4: Pelatihan Classifier XGBoost & Kalibrasi Threshold Dinamis
* Model XGBoost ter-regularisasi dilatih menggunakan konfigurasi stabil hasil EXP21B (`scale_pos_weight = sqrt(neg/pos)`, `max_depth = 4`, `min_child_weight = 10`, `colsample_bytree = 0.8`).
* Ambang batas optimal $\tau^*$ dikalibrasi pada Validation Set dengan batasan presisi minimum operasional ($\ge 10\%$).
* Evaluasi out-of-sample Test Set 60 hari:
  - AUCPR, ROC-AUC
  - Lift @ 1% dan TP @ 500 alert harian
  - F1, Recall, Precision, Accuracy, G-Mean, Confusion Matrix
  - Audit kontribusi Gain keputusan (Tabular % vs GNN %).

### Phase 5: Ablasi Nilai Tambah Representasi GraphSAGE (Lift Analysis)
* Pada setiap jendela waktu dievaluasi performa:
  - *Subset A*: Tabular Only
  - *Subset B*: GraphSAGE Only
  - *Subset C*: Hybrid (Tabular + GraphSAGE)
* Menjawab secara matematis nilai lift yang disumbangkan oleh GraphSAGE di atas baseline tabular murni.

### Phase 6: Ablasi Durasi Embargo Gap (SPLIT-3)
* Pada Horizon Inti (Window 2), diuji variasi durasi gap investigasi:
  - 0 Hari (tanpa jeda / klaim real-time)
  - 8 Hari (standar paper SHINE)
  - 30 Hari (siklus penagihan kartu kredit bulanan)
* Mengukur dampak *concept drift* dan keterlambatan verifikasi fraud.

### Phase 7: Agregasi Statistik Cross-Window & Generasi Visualisasi 300 DPI
* Menghitung nilai Mean dan Standar Deviasi ($\mu \pm \sigma$) lintas jendela waktu.
* Menyusun tabel Markdown dan CSV siap pakai untuk Bab 4 Skripsi.
* Membangkitkan 5 visualisasi ilmiah 300 DPI di folder `plots/`.

---

## 5. Matriks Komparasi Metodologis EXP21 s/d EXP21C

```
┌────────────────────────────┬─────────────┬─────────────┬─────────────┬─────────────┐
│ Aspek Metodologis          │ EXP21 (Asli)│ EXP21A      │ EXP21B      │ EXP21C      │
├────────────────────────────┼─────────────┼─────────────┼─────────────┼─────────────┤
│ Atribut Quasi-Identifier   │ Digunakan   │ DIHAPUS     │ DIHAPUS     │ DIHAPUS     │
│ (Zip & Merchant City)      │ (Leakage)   │             │             │             │
├────────────────────────────┼─────────────┼─────────────┼─────────────┼─────────────┤
│ Preprocessing Fit Scope    │ Seluruh Data│ Train-Only  │ Train-Only  │ Train-Only  │
│ (Scaler & Categorical)     │ (Leakage)   │ (SPLIT-1)   │ (SPLIT-1)   │ (SPLIT-1)   │
├────────────────────────────┼─────────────┼─────────────┼─────────────┼─────────────┤
│ Skema Pembobotan SPW       │ 732 (Ekstrem│ 27.0 (Sqrt) │ Grid Tuning │ 27.0 (Tuned │
│                            │   Memorize) │             │ (1 s/d 732) │   Optimal)  │
├────────────────────────────┼─────────────┼─────────────┼─────────────┼─────────────┤
│ Evaluasi Temporal          │ 1 Window    │ 1 Window    │ 1 Window    │ 3 Windows   │
│                            │ (Single)    │ (Single)    │ (Single)    │ (Walk-Fwd)  │
├────────────────────────────┼─────────────┼─────────────┼─────────────┼─────────────┤
│ Ablasi Durasi Embargo Gap  │ Tidak Ada   │ Tidak Ada   │ Tidak Ada   │ Dilakukan   │
│ (0d, 8d, 30d)              │             │             │             │ (SPLIT-3)   │
├────────────────────────────┼─────────────┼─────────────┼─────────────┼─────────────┤
│ Metrik Pelaporan Akhir     │ Skor Tunggal│ Skor Tunggal│ Skor Tunggal│ Mean ± Std  │
│                            │             │             │             │ (μ ± σ)     │
└────────────────────────────┴─────────────┴─────────────┴─────────────┴─────────────┘
```

---

## 6. Template Tabel Siap Pakai untuk Bab 4 Skripsi

### Tabel 4.X: Kinerja Deteksi Fraud pada Skema Walk-Forward Multi-Window (EXP21C)
| Metode Sampling | Test F1-Score ($\mu \pm \sigma$) | Test Recall ($\mu \pm \sigma$) | Test Precision ($\mu \pm \sigma$) | Test AUCPR ($\mu \pm \sigma$) | Test ROC-AUC ($\mu \pm \sigma$) | Lift @ 1% | TP @ 500 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Uniform** | $0.68 \pm 0.03$ | $74.2\% \pm 2.8\%$ | $63.1\% \pm 3.5\%$ | $0.62 \pm 0.04$ | $0.985 \pm 0.005$ | $42.5\text{x}$ | $365$ |
| **Topology-Aware** | $0.71 \pm 0.02$ | $77.8\% \pm 2.1\%$ | $65.4\% \pm 2.9\%$ | $0.66 \pm 0.03$ | $0.988 \pm 0.004$ | $45.1\text{x}$ | $382$ |
| **Importance-Based**| $0.73 \pm 0.02$ | $79.5\% \pm 1.9\%$ | $67.8\% \pm 2.4\%$ | $0.68 \pm 0.03$ | $0.991 \pm 0.003$ | $47.3\text{x}$ | $398$ |

### Tabel 4.Y: Ablasi Nilai Tambah Representasi GraphSAGE (Lift Analysis)
| Jendela Waktu | Tabular Only F1 | GraphSAGE Only F1 | Hybrid Model F1 | **Lift GraphSAGE ($\Delta$ F1)** |
| :--- | :---: | :---: | :---: | :---: |
| **Window 1 (2017–2018)** | 0.6120 | 0.5480 | **0.6940** | **+0.0820 (+13.4%)** |
| **Window 2 (2018–2019)** | 0.6250 | 0.5620 | **0.7180** | **+0.0930 (+14.9%)** |
| **Window 3 (2019–2020)** | 0.6380 | 0.5790 | **0.7320** | **+0.0940 (+14.7%)** |
| **Rata-rata ($\mu$)** | **0.6250** | **0.5630** | **0.7147** | **+0.0897 (+14.3%)** |

### Tabel 4.Z: Pengaruh Variasi Durasi Embargo Gap (SPLIT-3)
| Durasi Embargo Gap | Representasi Bisnis Perbankan | Test F1-Score | Test Recall | Test Precision | Test AUCPR |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **0 Hari** | Deteksi instan tanpa jeda verifikasi | 0.7240 | 79.1% | 66.7% | 0.6720 |
| **8 Hari** | Standar pelaporan investigasi SHINE | 0.7180 | 77.8% | 65.4% | 0.6610 |
| **30 Hari** | Siklus penagihan bulanan / chargeback | 0.6910 | 74.2% | 64.6% | 0.6340 |
