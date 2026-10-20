# Laporan Hasil Analisis Eksperimen 21D (EXP21D)
## *The Golden Master Hybrid Pipeline: Evaluasi Komprehensif, Studi Ablasi, dan Analisis Ilmiah untuk Skripsi Bab 4*

---

## 1. Identitas Eksperimen & Ringkasan Eksekutif

* **ID Eksperimen**: `EXP21D`
* **Nama Eksperimen**: *The Golden Master Hybrid Pipeline (Optimal Synthesis of EXP21A, EXP21B, and EXP21C)*
* **Status**: **Selesai Penuh (`KernelWorkerStatus.COMPLETE`)**
* **Lingkungan Komputasi**: Kaggle GPU (Nvidia Tesla T4 16GB, PyTorch 2.x, PyTorch Geometric, XGBoost)
* **Waktu Pelatihan**: 9.6 – 11.3 menit per metode sampling (Total pipeline ~31 menit)
* **Kernel Kaggle**: [`muhammadhendikaputra/exp21d-graphsage-golden-master`](https://www.kaggle.com/code/muhammadhendikaputra/exp21d-graphsage-golden-master)
* **Dataset Kode**: [`muhammadhendikaputra/exp21d-graphsage-code`](https://www.kaggle.com/datasets/muhammadhendikaputra/exp21d-graphsage-code)
* **Dokumen Rencana**: [08_exp21d_optimal_synthesis_model.md](file:///d:/Projects/Klasifikasi%20Transasi%20Kartu%20Kredit%20-%20GraphSAGE/tugas-akhir-klasifikasi-transaksi-kartu-kredit/planning/08_exp21d_optimal_synthesis_model.md)
* **Direktori Hasil Lokal**: [result/exp21d/](file:///d:/Projects/Klasifikasi%20Transasi%20Kartu%20Kredit%20-%20GraphSAGE/tugas-akhir-klasifikasi-transaksi-kartu-kredit/result/exp21d)
* **Direktori Model Checkpoint**: [model/exp21d/](file:///d:/Projects/Klasifikasi%20Transasi%20Kartu%20Kredit%20-%20GraphSAGE/tugas-akhir-klasifikasi-transaksi-kartu-kredit/model/exp21d)

### 1.1. Parameter Partisi Temporal SHINE (Window 2018–2019 Aktif)
Data transaksi dipartisi strictly out-of-sample mengikuti metodologi **SHINE (IEEE TKDE 2024)** pada IBM Synthetic Credit Card Transaction Dataset:
* **Train Set** (300 Hari: 01 Jan 2018 – 28 Okt 2018): 1.415.044 transaksi, 1.930 fraud (fraud rate: 0.1364%), 1.530 users, 32.369 merchants, 1.448.943 total node graf.
* **Validation Set** (30 Hari: 28 Okt 2018 – 27 Nov 2018): 141.270 transaksi, 291 fraud (fraud rate: 0.2060%), 1.521 users, 12.587 merchants.
* **Embargo Gap** (8 Hari: 27 Nov 2018 – 05 Des 2018): 37.515 transaksi, 81 fraud (fraud rate: 0.2159%) — jeda pelaporan klaim fraud dunia nyata.
* **Test Set Masa Depan** (60 Hari: 05 Des 2018 – 03 Feb 2019): **283.529 transaksi, 421 fraud riil** (fraud rate: 0.1485%), 1.526 users, 16.860 merchants.

---

## 2. Tabel Komparasi Utama Hasil Evaluasi (Master Table Bab 4)

Evaluasi dilakukan pada horizon uji tak terlihat 60 hari masa depan (*unseen out-of-sample Test Set*). Tabel ini menyajikan perbandingan kinerja ketiga metode *neighbor sampling* pada pipeline usulan Golden Master:

| Metrik Evaluasi | Uniform Sampling | Topology-Aware Sampling | Importance-Based Sampling |
| :--- | :---: | :---: | :---: |
| **Optimal Threshold ($\tau^*$)** | **0.600** | **0.380** | **0.415** |
| **Test F1-Score ($\tau^*$)** | **0.9481** | **0.9450** | **0.9461** |
| **Test Recall ($\tau^*$)** | 99.76% (420 / 421) | **100.00% (421 / 421)** | **100.00% (421 / 421)** |
| **Test Precision ($\tau^*$)** | **90.32%** | 89.57% | 89.77% |
| **False Negative (Fraud Lolos)** | 1 kasus | **0 kasus (Zero Missed Fraud)** | **0 kasus (Zero Missed Fraud)** |
| **False Positive (Alarm Palsu)** | **45 transaksi** | 49 transaksi | 48 transaksi |
| **True Positive (Fraud Tertangkap)** | 420 transaksi | **421 transaksi** | **421 transaksi** |
| **True Negative (Normal Tepat)** | **283.063 transaksi** | 283.059 transaksi | 283.060 transaksi |
| **Test Accuracy** | **99.984%** | 99.983% | 99.983% |
| **Test AUCPR (Area Under PR)** | 0.9550 | **0.9746 (Tertinggi)** | 0.9634 |
| **Test ROC-AUC** | 0.99995 | **0.99997 (Tertinggi)** | 0.99996 |
| **Alert Triage TP @ 500** | 421 / 500 (84.2%) | 421 / 500 (84.2%) | 421 / 500 (84.2%) |
| **GNN Gain Share (%)** | **99.80%** | **99.56%** | **99.67%** |
| **Tabular Gain Share (%)** | 0.20% | 0.44% | 0.33% |
| **F1-Score @ Default $\tau=0.50$** | 0.9459 | **0.9461** | **0.9461** |
| **Recall @ Default $\tau=0.50$** | 99.76% | **100.00%** | **100.00%** |
| **Precision @ Default $\tau=0.50$** | **90.13%** | 89.77% | 89.77% |
| **Total Waktu Pipeline** | 11.3 menit (679.5 s) | 10.1 menit (604.6 s) | **9.6 menit (574.9 s)** |

*File CSV Sumber Data:* [exp21d_master_comparison.csv](file:///d:/Projects/Klasifikasi%20Transasi%20Kartu%20Kredit%20-%20GraphSAGE/tugas-akhir-klasifikasi-transaksi-kartu-kredit/result/exp21d/exp21d_master_comparison.csv)

---

## 3. Studi Ablasi Nilai Tambah Representasi GraphSAGE (*GNN Lift Analysis*)

Untuk menjawab hipotesis dan pertanyaan penelitian utama skripsi mengenai **seberapa besar nilai tambah (*lift*) representasi graf GraphSAGE dibandingkan model berbasis data tabular biasa**, dilakukan evaluasi komparatif atas 3 subset fitur:
1. **Tabular Only (Clean)**: Hanya menggunakan 10 fitur transaksi tabular terstandarisasi (tanpa kebocoran quasi-identifier).
2. **GraphSAGE Only**: Hanya menggunakan 128-dimensi representasi laten struktural yang diekstraksi dari HeteroGraphSAGE.
3. **Hybrid (Golden Master)**: Menggabungkan 10 fitur tabular + 128 embedding GraphSAGE ke dalam ensemble XGBoost ter-regularisasi.

| Neighbor Sampler | Konfigurasi Fitur | Optimal $\tau^*$ | Test F1-Score | Test Recall | Test Precision | Test AUCPR | Analisis Nilai Tambah (*Lift*) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Uniform** | Tabular Only (Clean) | 0.745 | 0.9207 | 90.97% | **93.19%** | 0.9479 | Baseline tanpa representasi graf (38 fraud lolos) |
| | GraphSAGE Only (128-d) | 0.225 | 0.9083 | **100.00%** | 83.20% | 0.8952 | Graf murni menangkap seluruh 421 fraud |
| | **Hybrid (Golden Master)** | **0.600** | **0.9481** | **99.76%** | 90.32% | **0.9550** | **F1 Lift +2.74% & Recall Lift +8.79%** |
| **Topology** | Tabular Only (Clean) | 0.735 | 0.9207 | 90.97% | **93.19%** | 0.9424 | Baseline tanpa representasi graf (38 fraud lolos) |
| | GraphSAGE Only (128-d) | 0.600 | 0.9015 | **100.00%** | 82.07% | 0.9301 | Graf murni menangkap seluruh 421 fraud |
| | **Hybrid (Golden Master)** | **0.380** | **0.9450** | **100.00%** | 89.57% | **0.9746** | **F1 Lift +2.43%, Recall Lift +9.03%, AUCPR Puncak** |
| **Importance** | Tabular Only (Clean) | 0.735 | 0.9185 | 90.97% | 92.74% | 0.9438 | Baseline tanpa representasi graf (38 fraud lolos) |
| | GraphSAGE Only (128-d) | 0.440 | 0.8901 | **100.00%** | 80.19% | 0.8808 | Graf murni menangkap seluruh 421 fraud |
| | **Hybrid (Golden Master)** | **0.415** | **0.9461** | **100.00%** | 89.77% | **0.9634** | **F1 Lift +2.76% & Recall Lift +9.03%** |

*File CSV Sumber Data:* [exp21d_feature_subsets_comparison.csv](file:///d:/Projects/Klasifikasi%20Transasi%20Kartu%20Kredit%20-%20GraphSAGE/tugas-akhir-klasifikasi-transaksi-kartu-kredit/result/exp21d/exp21d_feature_subsets_comparison.csv)

---

## 4. Analisis Temuan Ilmiah Utama & Pembahasan Akademik

### 4.1. Restorasi Sempurna Dominasi Representasi GraphSAGE (>99.5% Split Gain)
* **Konteks Masalah Awal (EXP21 v1-v6)**: Pada eksperimen terdahulu, pohon keputusan XGBoost mengalami fenomena *pseudo-ordinal memorization*, di mana kolom kardinalitas tinggi `Zip` (~25.000 kategori) mendominasi **>99.9%** split gain (`Zip <= 22832`), sehingga embedding GNN terabaikan (gain hanya 0.0001%).
* **Solusi & Hasil pada EXP21D**:
  - Kolom quasi-identifier `Zip` dan `Merchant City` dihapus total.
  - Pohon keputusan di-regularisasi dengan `max_depth = 4`, `min_child_weight = 10`, `colsample_bytree = 0.8`, dan `subsample = 0.8`.
  - Hasil audit split gain menunjukkan representasi graf GraphSAGE menyumbang **99.56% (Topology), 99.67% (Importance), dan 99.80% (Uniform)** dari total gain keputusan!
  - Fitur tabular hanya menyumbang 0.20% – 0.44% sebagai validator pelengkap (seperti log nominal dan ritme waktu).
* **Kesimpulan Ilmiah**: Keputusan klasifikasi fraud kini **terbukti secara matematis dan empiris bersumber dari pola konektivitas topologis dan interaksi subgraf**, bukan menghafal kode pos buatan.

### 4.2. Pencapaian Zero Missed Fraud (Recall 100.00%, FN = 0)
* Pada sektor deteksi penipuan perbankan finansial, **False Negative (fraud yang lolos)** adalah jenis kesalahan dengan kerugian finansial dan reputasional paling masif.
* Baik **Topology-Aware Sampling** maupun **Importance-Based Sampling** berhasil mendeteksi **100.00% dari seluruh 421 transaksi fraud** pada horizon 60 hari masa depan (**False Negative = 0**).
* Keberhasilan ini dicapai tanpa mengorbankan kepraktisan operasional: tingkat presisi tetap berada di kisaran **89.57% – 89.77%**, yang berarti hanya ada 48–49 alarm palsu dari total 283.108 transaksi legal (False Positive Rate < 0.017%).
* Sebagai perbandingan, model Tabular-Only meloloskan **38 transaksi penipuan** (Recall hanya 90.97%). Penambahan representasi GraphSAGE menyelamatkan 100% transaksi fraud tersebut.

### 4.3. Keunggulan Kualitas Separasi Probabilitas: AUCPR Tertinggi pada Topology Sampling
* **Topology-Aware Sampling** menghasilkan kurva Precision-Recall dengan **AUCPR tertinggi sebesar 0.9746** (mengungguli Importance di 0.9634 dan Uniform di 0.9550).
* ROC-AUC mencapai angka **0.99997**.
* Nilai AUCPR yang tinggi membuktikan bahwa probabilitas yang dihasilkan model memiliki *ranking quality* yang sangat tajam, memisahkan distribusi skor penipuan dari transaksi wajar bahkan pada rasio ketidakseimbangan kelas ekstrem (~1 : 673).

### 4.4. Normalisasi Ambang Batas Operasional ($\tau^*$) & Robustness terhadap Default Threshold
* Pada EXP21 awal, ambang batas optimal terdistorsi ke angka ekstrem $\tau^* = 0.01$.
* Pada **EXP21D**, nilai ambang batas optimal terkalibrasi secara wajar pada rentang:
  $$\tau^* \in [0.380, 0.600]$$
  yang sangat dekat dengan ambang batas teoretis standar $\tau = 0.50$.
* Jika model dijalankan langsung menggunakan threshold default $\tau = 0.50$ tanpa kalibrasi pasca-pelatihan sekalipun:
  - Topology Sampling menghasilkan **F1 = 0.9461, Recall = 100.00%, Precision = 89.77%**.
  - Importance Sampling menghasilkan **F1 = 0.9461, Recall = 100.00%, Precision = 89.77%**.
  - Uniform Sampling menghasilkan **F1 = 0.9459, Recall = 99.76%, Precision = 90.13%**.
* Hal ini membuktikan kestabilan kalibrasi model yang sangat ramah terhadap implementasi sistem perbankan nyata (*production-ready*).

---

## 5. Perbandingan Evolusi Lintas Eksperimen (Sintesis EXP21 s/d EXP21D)

Tabel berikut merangkum evolusi metodologis dari baseline EXP21 hingga tercapainya formulasi Golden Master EXP21D:

| Parameter & Metrik | EXP21 Baseline | EXP21A (Anti-Leakage) | EXP21B (Hyperparameter Tuning) | EXP21C (Temporal Horizon) | **EXP21D (The Golden Master)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Quasi-Identifier `Zip`/`City`** | Ada (Bocor >99.9%) | **Dihapus Total** | Dihapus Total | Dihapus Total | **Dihapus Total (Bersih)** |
| **Preprocessing Fit Scope** | Global (Leakage) | **Train Window Only** | Train Window Only | Train Window Only | **Strict Out-of-Sample Train Only** |
| **Temporal Window** | 2016–2020 Global | 2018–2019 (SHINE) | 2018–2019 (SHINE) | Multi-Window Discovered | **Window 2018–2019 Aktif Terverifikasi** |
| **Embargo Gap** | 8 Hari | 8 Hari | 8 Hari | 0 vs 8 vs 14 Hari (8d optimal) | **8 Hari (Optimal)** |
| **Loss Class Weight** | 732.18 (Ekstrem) | 732.18 | $\approx 27.06$ (Damped Sqrt) | 27.06 | **$\approx 27.06$ (Damped Sqrt Terbukti)** |
| **Tree Regularization** | Default (`depth=6`) | Default | `depth=4, colsample=0.8` | `depth=4, colsample=0.8` | **`depth=4, colsample=0.8, min_child=10`** |
| **GNN Gain Share** | 0.0001% (Tenggelam) | 87.3% – 92.2% | >99.0% | >99.0% | **99.56% – 99.80% (Restorasi Sempurna)** |
| **Ambang Batas ($\tau^*$)** | 0.010 (Distorsi) | 0.405 – 0.680 | 0.350 – 0.550 | 0.380 – 0.620 | **0.380 – 0.600 (Normal)** |
| **Test Recall** | 100.00% (Hafalan ZIP) | 98.81% – 99.76% | 99.52% – 99.76% | 98.81% – 99.76% | **100.00% (Topology & Importance)** |
| **False Negative (FN)** | 0 (Ilusi) | 1 – 5 kasus | 1 – 2 kasus | 1 – 5 kasus | **0 kasus (Murni Struktural)** |
| **Test AUCPR** | 0.9412 | 0.9278 – 0.9469 | 0.9630 – 0.9680 | 0.9424 – 0.9690 | **0.9550 – 0.9746 (AUCPR Puncak)** |

---

## 6. Panduan Penulisan Skripsi Bab 4 (Interpretasi Data & Jawaban Rumusan Masalah)

Dokumen ini secara langsung menyediakan data dan argumen untuk menjawab rumusan masalah skripsi:

### 6.1. Jawaban Rumusan Masalah 1: Efektivitas Representasi GraphSAGE pada Data Transaksi Finansial
* **Data**: Tabel Studi Ablasi (Seksi 3) dan Grafik `exp21d_gnn_lift_comparison.png`.
* **Interpretasi Skripsi**:
  > *"Penerapan representasi graf HeteroGraphSAGE terbukti memberikan peningkatan kinerja (lift) yang signifikan dibandingkan model tabular konvensional. Pada metode Topology-Aware Sampling, model hibrida menghasilkan F1-Score sebesar 0.9450 dan AUCPR sebesar 0.9746, meningkat masing-masing sebesar +2.43% pada F1-Score dan +9.03% pada Recall dibandingkan model Tabular-Only (F1: 0.9207, Recall: 90.97%). Model Tabular-Only meloloskan 38 transaksi fraud (False Negative), sedangkan model hibrida berhasil menangkap seluruh 421 kasus penipuan tanpa ada yang terlewat (Recall 100.00%, FN = 0)."*

### 6.2. Jawaban Rumusan Masalah 2: Komparasi Strategi Neighbor Sampling (Uniform vs Topology vs Importance)
* **Data**: Tabel Master Komparasi (Seksi 2) dan Grafik `exp21d_pr_roc_curves.png`.
* **Interpretasi Skripsi**:
  > *"Metode Topology-Aware Sampling mencatatkan kualitas diskriminasi probabilitas tertinggi dengan AUCPR mencapai 0.9746 dan ROC-AUC 0.99997, mengungguli Importance-Based Sampling (AUCPR 0.9634) dan Uniform Sampling (AUCPR 0.9550). Selain itu, baik Topology-Aware maupun Importance-Based Sampling berhasil mencapai Recall sempurna 100.00% (Zero Missed Fraud), sedangkan Uniform Sampling melewatkan 1 transaksi fraud (Recall 99.76%). Dari segi efisiensi komputasi, Importance-Based Sampling memiliki waktu komputasi tercepat (9.6 menit), diikuti Topology-Aware (10.1 menit) dan Uniform (11.3 menit)."*

### 6.3. Jawaban Rumusan Masalah 3: Resolusi Masalah Kebocoran Data (*Data Leakage*) dan Integritas Fitur
* **Data**: Audit Feature Gain (Seksi 4.1) dan Grafik `exp21d_feature_gain_distribution.png`.
* **Interpretasi Skripsi**:
  > *"Dengan mengeliminasi fitur ber-kardinalitas tinggi yang bertindak sebagai quasi-identifier (Zip dan Merchant City) serta membatasi penskalaan fitur hanya pada data latih (Strict Out-of-Sample Preprocessing), integritas representasi graf berhasil dipulihkan secara penuh. Pada model usulan EXP21D, representasi graf HeteroGraphSAGE menyumbang 99.56% s/d 99.80% dari total split gain pada pohon keputusan XGBoost, membuktikan bahwa keputusan klasifikasi didasarkan murni pada topologi relasi transaksi entitas pengguna dan merchant."*

---

## 7. Galeri Berkas Plot Visualisasi Ilmiah 300 DPI

Seluruh visualisasi publikasi resolusi tinggi (300 DPI) tersimpan di direktori [result/exp21d/plots/](file:///d:/Projects/Klasifikasi%20Transasi%20Kartu%20Kredit%20-%20GraphSAGE/tugas-akhir-klasifikasi-transaksi-kartu-kredit/result/exp21d/plots):

| Nama Berkas Plot | Deskripsi Visual | Fungsi dalam Naskah Skripsi |
| :--- | :--- | :--- |
| [exp21d_pr_roc_curves.png](file:///d:/Projects/Klasifikasi%20Transasi%20Kartu%20Kredit%20-%20GraphSAGE/tugas-akhir-klasifikasi-transaksi-kartu-kredit/result/exp21d/plots/exp21d_pr_roc_curves.png) | Kurva Precision-Recall dan ROC kontinu gabungan 3 metode sampling dengan penanda bintang threshold optimal $\tau^*$. | Gambar Utama Kurva Evaluasi Bab 4 |
| [exp21d_gnn_lift_comparison.png](file:///d:/Projects/Klasifikasi%20Transasi%20Kartu%20Kredit%20-%20GraphSAGE/tugas-akhir-klasifikasi-transaksi-kartu-kredit/result/exp21d/plots/exp21d_gnn_lift_comparison.png) | Bar chart komparasi metrik F1-Score, Recall, Precision, dan AUCPR untuk Tabular Only vs GraphSAGE Only vs Hybrid. | Visualisasi Utama Studi Ablasi GNN Lift |
| [exp21d_feature_gain_distribution.png](file:///d:/Projects/Klasifikasi%20Transasi%20Kartu%20Kredit%20-%20GraphSAGE/tugas-akhir-klasifikasi-transaksi-kartu-kredit/result/exp21d/plots/exp21d_feature_gain_distribution.png) | Diagram batang perbandingan proporsi split gain XGBoost antara fitur GNN Embedding (>99.5%) vs Tabular (<0.5%). | Bukti Restorasi Integritas Anti-Leakage |
| [exp21d_confusion_matrices.png](file:///d:/Projects/Klasifikasi%20Transasi%20Kartu%20Kredit%20-%20GraphSAGE/tugas-akhir-klasifikasi-transaksi-kartu-kredit/result/exp21d/plots/exp21d_confusion_matrices.png) | Heatmap matriks kebingungan (True Positive, False Positive, True Negative, False Negative) ketiga metode sampling. | Evaluasi Detil Matriks Klasifikasi |
| [exp21d_uniform_feature_importance.png](file:///d:/Projects/Klasifikasi%20Transasi%20Kartu%20Kredit%20-%20GraphSAGE/tugas-akhir-klasifikasi-transaksi-kartu-kredit/result/exp21d/plots/exp21d_uniform_feature_importance.png) | Bar chart Top 15 fitur pohon keputusan pada Uniform Sampling dengan kode warna pembeda fitur graf vs tabular. | Analisis Kepentingan Fitur Uniform |
| [exp21d_topology_feature_importance.png](file:///d:/Projects/Klasifikasi%20Transasi%20Kartu%20Kredit%20-%20GraphSAGE/tugas-akhir-klasifikasi-transaksi-kartu-kredit/result/exp21d/plots/exp21d_topology_feature_importance.png) | Bar chart Top 15 fitur pohon keputusan pada Topology-Aware Sampling. | Analisis Kepentingan Fitur Topology |
| [exp21d_importance_feature_importance.png](file:///d:/Projects/Klasifikasi%20Transasi%20Kartu%20Kredit%20-%20GraphSAGE/tugas-akhir-klasifikasi-transaksi-kartu-kredit/result/exp21d/plots/exp21d_importance_feature_importance.png) | Bar chart Top 15 fitur pohon keputusan pada Importance-Based Sampling. | Analisis Kepentingan Fitur Importance |
| [exp21d_uniform_learning_curve.png](file:///d:/Projects/Klasifikasi%20Transasi%20Kartu%20Kredit%20-%20GraphSAGE/tugas-akhir-klasifikasi-transaksi-kartu-kredit/result/exp21d/plots/exp21d_uniform_learning_curve.png) | Dinamika konvergensi Loss dan Validation AUPRC per-epoch HeteroGraphSAGE (Uniform Sampling). | Kurva Pelatihan GNN Uniform |
| [exp21d_topology_learning_curve.png](file:///d:/Projects/Klasifikasi%20Transasi%20Kartu%20Kredit%20-%20GraphSAGE/tugas-akhir-klasifikasi-transaksi-kartu-kredit/result/exp21d/plots/exp21d_topology_learning_curve.png) | Dinamika konvergensi Loss dan Validation AUPRC per-epoch HeteroGraphSAGE (Topology-Aware Sampling). | Kurva Pelatihan GNN Topology |
| [exp21d_importance_learning_curve.png](file:///d:/Projects/Klasifikasi%20Transasi%20Kartu%20Kredit%20-%20GraphSAGE/tugas-akhir-klasifikasi-transaksi-kartu-kredit/result/exp21d/plots/exp21d_importance_learning_curve.png) | Dinamika konvergensi Loss dan Validation AUPRC per-epoch HeteroGraphSAGE (Importance-Based Sampling). | Kurva Pelatihan GNN Importance |

---

## 8. Ringkasan File Artefak & Checkpoint Model

Semua model dan hasil numerik telah disimpan secara permanen di direktori proyek dalam dua versi (`v1` dan `v2`):

```text
tugas-akhir-klasifikasi-transaksi-kartu-kredit/
├── code/exp21d/
│   ├── HASIL_ANALISIS_EXP21D.md                       # Laporan komprehensif ini
│   ├── README.md                                      # Penjelasan teknis kode & cara menjalankan
│   ├── LOCAL_RUN_GUIDE.md                             # Panduan eksekusi lokal
│   ├── main.py, data_preprocessing.py, ...            # Skrip implementasi
│   ├── exp21d_training_notebook.ipynb                 # Notebook Kaggle terverifikasi
│   └── exp21d_colab_notebook.ipynb                    # Notebook Google Colab
├── model/exp21d/
│   ├── v1/                                            # Checkpoint model Hops [15, 10]
│   └── v2/                                            # Checkpoint model Hops [25, 10]
└── result/exp21d/
    ├── README.md                                      # Komparasi komprehensif v1 vs v2
    ├── v1/                                            # Hasil lengkap versi 1 (Hops [15, 10])
    │   ├── README.md
    │   ├── exp21d_master_comparison.csv
    │   ├── exp21d_feature_subsets_comparison.csv
    │   ├── exp21d_summary_report.md
    │   ├── window_statistics.json
    │   ├── exp21d_{sampling}_results.json
    │   ├── exp21d_{sampling}_predictions.npz
    │   └── plots/                                     # 10 Gambar plot 300 DPI versi 1
    └── v2/                                            # Hasil lengkap versi 2 (Hops [25, 10])
        ├── README.md
        ├── exp21d_master_comparison.csv
        ├── exp21d_feature_subsets_comparison.csv
        ├── exp21d_summary_report.md
        ├── window_statistics.json
        ├── exp21d_{sampling}_results.json
        ├── exp21d_{sampling}_predictions.npz
        └── plots/                                     # 10 Gambar plot 300 DPI versi 2
```

