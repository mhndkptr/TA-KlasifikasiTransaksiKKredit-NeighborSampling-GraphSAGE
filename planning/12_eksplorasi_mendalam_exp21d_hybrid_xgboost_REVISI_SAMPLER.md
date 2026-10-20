# Rencana Eksplorasi EXP21D — Revisi Terfokus Komparasi Tiga Neighbor Sampler

**Dokumen asal yang dipertahankan**: `planning/12_eksplorasi_mendalam_exp21d_hybrid_xgboost.md`  
**Status dokumen ini**: rencana pengganti untuk implementasi berikutnya  
**Fokus tunggal**: perbandingan `Uniform`, `Topology-Aware`, dan `Importance-Based` sampling  
**Konfigurasi utama**: Hybrid HeteroGraphSAGE–XGBoost, Mean aggregator, fan-out `[15,10]`

## 1. Keputusan Scope

Paper bukan studi pencarian arsitektur GraphSAGE, bukan benchmark classifier, dan bukan studi agregator. Pertanyaan utamanya adalah apakah tiga strategi neighbor sampling menghasilkan subgraf, representasi, dan performa fraud detection yang berbeda ketika seluruh komponen lain dikendalikan.

### Variabel bebas

Hanya strategi sampling:

1. Uniform.
2. Topology-Aware: inverse log counterpart degree.
3. Importance-Based: log counterpart degree.

### Variabel yang wajib dikunci

- Mean aggregator.
- Fan-out `[15,10]`.
- Dua layer HeteroGraphSAGE dan embedding 128-d.
- Fitur tabular 10-d dan hybrid 138-d.
- Temporal split `300d train / 30d validation / 8d embargo / 60d test`.
- Preprocessing, graph construction, loss, early stopping, dan batch size.
- Konfigurasi XGBoost.
- Threshold selection dari validation saja.
- Root transactions, paired seeds, dan urutan evaluasi.

### Di luar scope utama

- Mean-vs-Max aggregator.
- Fan-out search.
- GNN architecture search.
- Benchmark XGBoost vs LightGBM vs Random Forest.
- Broad XGBoost hyperparameter sweep per sampler.
- Embargo-gap optimization menggunakan test.
- Klaim real-time sebelum causal graph evaluation tersedia.

## 2. Hipotesis dan Research Questions

### RQ1 — Mekanisme sampling

Apakah tiga sampler memilih neighborhood yang berbeda pada root dan random stream yang sama?

**H1a**: Topology-Aware menghasilkan counterpart degree terpilih yang lebih rendah daripada Uniform.  
**H1b**: Importance-Based menghasilkan counterpart degree terpilih yang lebih tinggi daripada Uniform.  
**H1c**: Jaccard edge overlap antarsampler lebih rendah pada entitas ber-degree tinggi daripada entitas ber-degree rendah.

### RQ2 — Performa prediktif

Apakah perubahan neighborhood menghasilkan perbedaan AUPRC dan operating-point metrics pada hybrid GraphSAGE–XGBoost?

**Primary metric**: AUPRC.  
**Secondary metrics**: F1, precision, recall, ROC-AUC, FP, dan FN.  
**Operational metrics**: Recall@K dan Precision@K.

### RQ3 — Regime struktural

Apakah keuntungan atau kerugian sampler terkonsentrasi pada user/merchant ber-degree tinggi dan kasus fraud tertentu?

### RQ4 — Robustness

Apakah ranking dan paired delta antarsampler stabil pada lima seeds?

## 3. Kontrak Corrected Sampler

Weighted sampling hanya dinyatakan valid jika seluruh kondisi berikut terpenuhi:

1. `NeighborLoader(..., weight_attr="sampling_weight")` aktif.
2. Sampling tidak menggunakan label.
3. Kandidat transaksi melalui user diberi bobot berdasarkan merchant degree kandidat.
4. Kandidat transaksi melalui merchant diberi bobot berdasarkan user degree kandidat.
5. Uniform memakai bobot satu.
6. Topology dan Importance memakai min-max normalization dengan smoothing floor `0.1`.
7. Audit train, validation, dan test berstatus `passed`.
8. Topology menunjukkan korelasi degree-weight negatif; Importance positif.

EXP21D-v5 telah memenuhi correctness gate tersebut untuk seed `42`.

## 4. Inventaris Implementasi Saat Ini

| Kode | Eksperimen | Hubungan dengan fokus sampler | Status |
|---|---|---|---|
| `exp21d_v4_fairctrl_1a` | 1A MLP 128-d control | Kontrol tambahan | Selesai dijalankan |
| `exp21d_v5_sampler_correctness` | 2A corrected sampler + main outcome | **Inti** | Selesai dijalankan dan dianalisis |
| `exp21d_v6_fairctrl_1c` | 1C frozen GraphSAGE | Kontrol tambahan | Kode dan submission tersedia; bukan dependency eksperimen inti |
| `exp21d_v7_sampler_audit_2b` | 2B sampled neighborhood audit | **Inti** | Kode selesai; belum dijalankan |
| `exp21d_v8_fairctrl_1b` | 1B PCA 10-d control | Kontrol tambahan | Kode selesai; belum dipush |

Kontrol 1A/1B/1C dapat dipindahkan ke supplementary material. Urutan eksperimen inti tidak boleh menunggu ketiganya.

## 5. Hasil Awal Corrected Single-Seed

Hasil EXP 2A, seed `42`:

| Sampler | F1 | Recall | Precision | AUPRC | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| Uniform | 0,9429 | 1,0000 | 0,8919 | 0,9612 | 51 | 0 |
| Topology-Aware | 0,9471 | 1,0000 | 0,8996 | **0,9789** | 47 | 0 |
| Importance-Based | **0,9513** | 0,9976 | **0,9091** | 0,9753 | **42** | 1 |

Interpretasi dibatasi sebagai descriptive single-seed evidence. Topology memimpin AUPRC, sedangkan Importance memimpin F1/precision. Tidak ada klaim bahwa salah satunya unggul universal.

## 6. Eksperimen Inti Berikutnya

### EXP 2B — Sampled Neighborhood Audit

**Kode**: `code/exp21d_v7_sampler_audit_2b`  
**Training GNN/XGBoost**: tidak  
**Dependency**: dataset asli dan corrected sampling core

Output wajib:

- Distribusi degree tetangga terpilih per split, hop, dan kanal.
- Mean, median, kuartil, serta tail degree.
- Paired Jaccard edge overlap antarsampler.
- Effective fraud density post-hoc; label digabung setelah sampling.
- Analisis population roots dan fraud-root cohort secara terpisah.
- NPZ edge-level agar strata atau metrik baru dapat dihitung tanpa sampling ulang.

Acceptance criteria:

- Root dan random stream sama untuk seluruh sampler.
- Self-edge tidak menaikkan homofili secara artifisial.
- Arah degree distribution konsisten dengan weighting function.

### EXP 3S — Sampler Mechanism and Degree-Stratified Outcome

**Nama folder yang disarankan**: `code/exp21d_v9_sampler_mechanism_3s`  
**Training GNN/XGBoost**: tidak  
**Dependency**: hasil lengkap EXP 2A; dataset asli untuk transaction ID, timestamp, user, merchant, dan degree

Cakupan:

1. Rekonstruksi row alignment test secara deterministik.
2. Strata degree yang dibekukan sebelum melihat hasil, misalnya:
   - low: `degree ≤ 15`;
   - medium: `16–200`;
   - high/super-hub: `>200`.
3. Per strata: jumlah transaksi/fraud, AUPRC, F1, precision, recall, FP, dan FN.
4. Paired error overlap:
   - fraud tertangkap semua sampler;
   - hanya tertangkap Topology;
   - hanya tertangkap Importance;
   - terlewat bersama.
5. Latent diagnostics dari blok embedding 128-d:
   - centroid distance;
   - Fisher discriminant summary;
   - visualisasi 2D sebagai eksplorasi, bukan bukti utama.

Catatan: jumlah kasus fraud yang diklaim terselamatkan pada EXP21D lama tidak dipakai. Jumlah dan membership kasus dihitung ulang dari corrected EXP 2A.

Artefak wajib:

- Row-level analysis table dengan transaction ID dan strata.
- Prediksi dan label per sampler.
- Degree-stratified metrics CSV/JSON.
- Fraud overlap membership NPZ/Parquet.
- Plot PNG/PDF 300 DPI dan source data.

### EXP 4S — Paired Multi-Seed Sampler Comparison

**Nama folder yang disarankan**: `code/exp21d_v10_sampler_multiseed_4s`  
**Training GNN/XGBoost**: ya  
**Dependency**: pipeline EXP 2A tervalidasi; EXP 2B disarankan selesai sebelum GPU budget besar digunakan

Desain:

```text
samplers = [uniform, topology, importance]
seeds    = [42, 100, 2026, 3407, 8713]
total    = 15 paired runs
```

Kontrak fairness:

- Seed yang sama membentuk satu paired block.
- Hyperparameter tidak berubah berdasarkan hasil test.
- Tabular-only baseline dihitung satu kali per seed dan harus identik untuk ketiga sampler.
- Test probabilities, labels, IDs, timestamp, dan threshold disimpan.
- Tiap run memiliki manifest, config snapshot, environment, model, history, audit, dan artifact index.

Output utama:

- Per-run metrics.
- Mean, SD, median, dan paired delta.
- Runtime dan peak memory.
- Prediction hash untuk reproducibility.

### EXP 5S — Statistical and Operational Analysis

**Nama folder yang disarankan**: `code/exp21d_v11_sampler_statistics_5s`  
**Training**: tidak  
**Dependency**: EXP 4S

Analisis:

1. Paired seed delta untuk AUPRC dan F1.
2. Confidence interval yang mempertahankan struktur temporal, idealnya block bootstrap per hari.
3. McNemar pada prediksi berpasangan untuk operating point yang telah dibekukan.
4. Recall@K dan Precision@K untuk `K = 100, 200, 300, 421, 500, 1000`.
5. Threshold sensitivity tanpa memilih threshold baru dari test.
6. Win/tie/loss sampler per seed dan metric.

## 7. Eksperimen Kontrol Pendukung

### 1A — MLP 128-d

Menjawab apakah proyeksi neural tabular dengan dimensi sama dapat menjelaskan hasil embedding GraphSAGE. Tidak menentukan pemenang sampler.

### 1B — PCA 10-d

Menjawab kritik bias dimensi 128-d vs 10-d. Berguna sebagai supplementary evidence, tetapi tidak perlu dijalankan sebelum EXP 2B/3S/4S.

### 1C — Frozen GraphSAGE

Mengukur informasi dari agregasi struktur tanpa supervised encoder training. Hasil dapat dilaporkan singkat sebagai kontrol label exposure.

## 8. Eksperimen yang Dihapus dari Roadmap Utama

### Mean vs Max Aggregator

Dihapus karena mengubah dua faktor sekaligus: sampler dan aggregation function. Hal tersebut memperbesar matriks eksperimen dan melemahkan atribusi kausal terhadap sampling.

### Downstream Classifier Benchmark

Tidak menjadi prioritas karena paper mengunci XGBoost sebagai downstream classifier. Benchmark classifier hanya boleh menjadi pekerjaan terpisah atau supplementary jika seluruh eksperimen inti telah selesai.

### Broad XGBoost Grid

Tidak dijalankan per sampler. Jika tuning diperlukan, lakukan sekali pada development protocol yang telah ditentukan, lalu bekukan satu konfigurasi untuk seluruh sampler.

## 9. Dependency dan Paralelisasi Implementasi

| Kode | Dapat diimplementasikan sekarang? | Dapat dijalankan tanpa menunggu kontrol 1C? | Dependency hasil |
|---|---|---|---|
| EXP 2B | Sudah | Ya | Dataset |
| EXP 3S | Ya | Ya | EXP 2A + dataset |
| EXP 4S | Ya | Ya | Pipeline EXP 2A |
| EXP 5S | Skeleton saja | Ya | Hasil lengkap EXP 4S |

Implementasi EXP 3S dan EXP 4S tidak perlu menunggu hasil 1B atau 1C. Eksekusi GPU 15-run EXP 4S sebaiknya dimulai setelah EXP 2B mengonfirmasi bahwa distribusi neighborhood benar-benar berbeda.

## 10. Urutan Eksekusi yang Direkomendasikan

1. Jalankan EXP 2B.
2. Implementasikan dan jalankan EXP 3S dari artefak EXP 2A.
3. Bekukan primary metric, operational budgets, strata degree, seeds, dan reporting rules.
4. Implementasikan EXP 4S.
5. Jalankan 15 paired runs sesuai batas sesi Kaggle.
6. Jalankan EXP 5S tanpa retraining.
7. Masukkan 1A/1B/1C hanya sebagai kontrol tambahan jika diperlukan.

## 11. Rencana Figure dan Table

### Main paper

- Figure 1: pipeline dan tiga weighting functions.
- Figure 2: empirical degree distribution, Jaccard, dan fraud density dari EXP 2B.
- Figure 3: paired sampler performance atau operational curves dengan CI.
- Table I: dataset dan konfigurasi yang dikunci.
- Table II: mean ± SD/CI, FP/FN, dan Recall@K ketiga sampler.

### Supplementary/repository

- Degree-stratified results.
- Fraud capture overlap.
- Precision–recall curves.
- Latent projection.
- Kontrol 1A/1B/1C.

## 12. Stop/Go Criteria

Paper dapat menyatakan adanya pengaruh sampling apabila:

1. EXP 2B menunjukkan neighborhood distribution yang memang berbeda.
2. Perbedaan outcome tidak hanya muncul pada threshold-dependent F1, tetapi juga AUPRC atau Recall@K.
3. Arah paired delta cukup stabil pada seeds.
4. Analisis degree strata memberikan penjelasan yang konsisten dengan mekanisme sampler.

Jika audit subgraf berbeda tetapi performa tidak berbeda stabil, kesimpulan yang sah adalah bahwa bias degree mengubah neighborhood tanpa memberikan keuntungan prediktif yang konsisten. Jika ranking berubah antar-metric, paper harus melaporkan trade-off dan tidak memilih satu pemenang universal.

## 13. Artefact Contract

Semua eksperimen baru harus menyimpan:

- resolved config dan git revision;
- environment/package versions;
- seluruh seeds dan sampler audit;
- append-only event log;
- model dan training history;
- validation/test raw probabilities dan labels;
- transaction ID, timestamp, user/merchant IDs atau hash, serta degree;
- threshold-selection trail;
- feature matrices atau referensi immutable beserta fingerprint;
- CSV/JSON source untuk seluruh figure;
- artifact index.

Kontrak ini memastikan metrik dan visualisasi tambahan dapat dibuat tanpa mengulang training.
