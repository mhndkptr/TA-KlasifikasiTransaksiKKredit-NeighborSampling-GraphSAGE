# Eksperimen 21D (EXP21D): The Golden Master Hybrid Pipeline
## *Sintesis Komprehensif: Strict Anti-Leakage, Optimal Regularization, dan Validated Temporal Window*

---

## 1. Deskripsi Eksperimen

Eksperimen **EXP21D** adalah formulasi model hibrida final terbaik (*The Golden Master*) yang memadukan seluruh keunggulan dan pembelajaran empiris dari tiga fase eksperimen sebelumnya:

1. **Dari EXP21A (Anti-Leakage & Clean Feature Engineering)**:
   - Eliminasi total kolom kardinalitas tinggi `Zip` (~25.000 kategori) dan `Merchant City` (~10.000 kategori).
   - *Strict Out-of-Sample Preprocessing*: `StandardScaler` dan pemetaan kategori HANYA di-`fit` pada data latih (Train Window). Kategori baru pada Val/Test ditangani via token `<OOV>` ($=0$).
   - Penambahan fitur temporal siklikal: `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`.
   - Mengembalikan share gain keputusan pohon ke representasi graf GraphSAGE (**$>90\%$ GNN Gain Share**).

2. **Dari EXP21B (Balanced Hyperparameter Tuning & Tree Regularization)**:
   - Meredam pembobotan kelas `scale_pos_weight` dari 732.18 menjadi $\sqrt{\text{neg}/\text{pos}} \approx 27.06$ (atau $10.0$), menghasilkan batas keputusan terkalibrasi dan AUCPR puncak (**0.9680**).
   - Regularisasi pohon anti-memorization: `max_depth = 4`, `min_child_weight = 10`, `colsample_bytree = 0.8`, `subsample = 0.8`.

3. **Dari EXP21C (Validated Temporal Horizon & Embargo Gap)**:
   - Evaluasi difokuskan pada **Window 2018–2019 aktif** (`reference_date: "2018-01-01"`), yang terbukti secara empiris sebagai satu-satunya periode dengan persebaran transaksi fraud seimbang di seluruh partisi.
   - Retensi **Embargo Gap 8 hari** (standar protokol SHINE FRAUD II) yang terbukti memberikan AUCPR optimal (**0.9690** vs 0.8590 pada gap 0 hari).

---

## 2. Struktur Berkas di Direktori Ini

```text
code/exp21d/
├── config.reference.yaml         # Konfigurasi acuan lokal
├── config.kaggle.yaml            # Konfigurasi GPU Kaggle
├── config.colab.yaml             # Konfigurasi GPU Colab
├── config.py                     # Parser konfigurasi YAML
├── data_preprocessing.py         # Modul strict anti-leakage splitting & clean features
├── graph_builder.py              # Pembangun graf heterogen tripartit PyG
├── samplers.py                   # Implementasi Uniform, Topology-Aware, Importance-Based
├── model.py                      # Arsitektur 2-Layer HeteroGraphSAGE ter-regularisasi
├── main.py                       # Pipeline pelatihan, ekstraksi embedding, tuning XGBoost, & evaluasi
├── run_all.py                    # Eksekusi sekuensial 3 metode sampling & pembuatan tabel master
├── visualize_results.py          # Generator plot ilmiah 300 DPI
├── create_nb.py                  # Generator notebook Kaggle mandiri
├── create_colab_nb.py            # Generator notebook Colab mandiri
├── dataset-metadata.json         # Metadata dataset kode Kaggle API
├── kernel-metadata.json          # Metadata kernel Kaggle API
├── LOCAL_RUN_GUIDE.md            # Panduan eksekusi lokal
├── HASIL_ANALISIS_EXP21D.md      # Laporan analisis hasil komprehensif Bab 4 skripsi
└── README.md                     # Berkas dokumentasi ini
```

---

## 3. Cara Menjalankan

### Eksekusi Lokal Cepat:
```powershell
python main.py --config config.reference.yaml --sampling uniform --nrows 50000
```

### Eksekusi Penuh 3 Metode Sampling & Pembuatan Master Table:
```powershell
python run_all.py
```

### Generate Ulang Notebook Kaggle / Colab:
```powershell
python create_nb.py; python create_colab_nb.py
```

---

## 4. Ringkasan Hasil Evaluasi Akhir (The Golden Master)

Pelatihan dan evaluasi penuh telah selesai di Kaggle GPU (`KernelWorkerStatus.COMPLETE`). Seluruh artefak telah diunduh ke `result/exp21d/` dan `model/exp21d/`.

### Tabel Komparasi Utama (Test Set 60 Hari Masa Depan - 283.529 Transaksi):

| Neighbor Sampler | Optimal $\tau^*$ | Test F1-Score | Test Recall | Test Precision | False Negative | Test AUCPR | Test ROC-AUC | GNN Gain Share (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Uniform** | **0.600** | **0.9481** | **99.76%** (420/421) | **90.32%** | 1 kasus | 0.9550 | 0.99995 | **99.80%** |
| **Topology** | **0.380** | **0.9450** | **100.00% (421/421)** | **89.57%** | **0 (Zero Missed)** | **0.9746 (Puncak)** | **0.99997** | **99.56%** |
| **Importance** | **0.415** | **0.9461** | **100.00% (421/421)** | **89.77%** | **0 (Zero Missed)** | **0.9634** | **0.99996** | **99.67%** |

### Poin Temuan Utama:
1. **Restorasi Sempurna Representasi GraphSAGE**: Share gain GNN meningkat drastis dari 0.0001% (pada baseline cacat) menjadi **>99.5%**, membuktikan keputusan model murni berasal dari topologi interaksi subgraf.
2. **Zero Missed Fraud (FN = 0)**: Sampling Topology-Aware dan Importance-Based menangkap **100.00% seluruh 421 transaksi fraud** out-of-sample tanpa ada yang terlewat.
3. **AUCPR Tertinggi pada Topology Sampling (0.9746)**: Memberikan pemisahan probabilitas paling optimal pada ketidakseimbangan kelas ekstrem.
4. **GNN Lift Signifikan**: Memberikan peningkatan Recall hingga **+9.03%** dan F1-Score **+2.76%** dibandingkan baseline Tabular-Only (yang meloloskan 38 transaksi fraud).

Laporan lengkap beserta interpretasi akademik untuk Bab 4 skripsi dapat dibaca di:
👉 **[HASIL_ANALISIS_EXP21D.md](HASIL_ANALISIS_EXP21D.md)**

