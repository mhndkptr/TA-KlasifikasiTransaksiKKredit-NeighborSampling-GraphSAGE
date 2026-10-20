# Rencana EXP21E-v3 dan Seterusnya

## 1. Tujuan

Roadmap ini mengubah EXP21E dari eksperimen pilot menjadi rangkaian eksperimen yang dapat dipakai sebagai bukti utama paper. Fokus penelitian tetap pada perbandingan tiga strategi neighbor sampling pada Pure End-to-End HeteroGraphSAGE:

1. `uniform`;
2. `topology`;
3. `importance`.

Baseline non-graf yang digunakan hanya **End-to-End Tabular MLP** dengan arsitektur, loss, prosedur threshold, dan seed yang dibuat sebanding dengan model graf.

Hasil EXP21E yang sudah tersedia dipertahankan sebagai **pilot lama**. Hasil tersebut tidak boleh ditimpa dan tidak digunakan sebagai bukti final keunggulan sampler.

---

## 2. Prinsip Eksperimen yang Dikunci

Semua versi mulai V3 harus mengikuti aturan berikut.

1. **Satu perubahan utama per versi.** Perubahan kode, data, dan konfigurasi harus dapat ditelusuri.
2. **Paired comparison.** Seed, split, model initialization policy, urutan root node, dan protokol evaluasi harus sama untuk sampler yang dibandingkan.
3. **Validation-only selection.** Pemilihan checkpoint, threshold, sampler terbaik, dan hyperparameter tidak boleh memakai test set.
4. **Causal evaluation.** Fitur, degree, aggregate, edge, dan candidate neighbor untuk transaksi pada waktu `t` hanya boleh memakai informasi yang tersedia sebelum atau pada cutoff yang ditetapkan.
5. **Deterministic or repeated evaluation.** Evaluasi memakai neighborhood beku, fixed evaluation seed, full-neighborhood inference, atau Monte Carlo inference yang identik bagi seluruh metode.
6. **AUPRC sebagai metrik utama.** F1, precision, recall, FP, FN, Recall@K, runtime, dan peak memory menjadi metrik pendukung.
7. **Tidak menimpa hasil.** Setiap run mempunyai version ID, strategy, seed, window ID, configuration hash, dan run ID tersendiri.
8. **Tidak ada tuning setelah test dibuka.** Perubahan setelah test dibaca harus menghasilkan versi eksperimen baru.

Konfigurasi awal yang dipertahankan kecuali sedang diuji secara eksplisit:

```text
Model             : Pure End-to-End HeteroGraphSAGE
Hidden dimension  : 128
Layers            : 2
Aggregator         : mean
Dropout            : 0.2
Fan-out            : [15, 10]
Loss               : damped weighted BCE
Train/Val/Gap/Test : 300/30/8/60 hari
Primary metric     : validation/test AUPRC
Primary seeds      : [42, 100, 2026, 3407, 8713]
```

---

## 3. Ringkasan Roadmap

| Versi | Prioritas | Tujuan | Training runs baru | Dipakai sebagai hasil final? |
|---|---:|---|---:|---:|
| **V3** | P0 | Koreksi dan validasi sampler, seed, threshold, dan logging | 3 smoke runs | Tidak |
| **V4** | P0 | Pilot causal temporal end-to-end | 3 pilot runs | Tidak |
| **V5** | P0 | Eksperimen utama, 3 sampler x 5 paired seeds | 15 GNN runs | Ya |
| **V6** | P1 | Robustness 3 sampler pada 2 additional temporal windows | 18 GNN runs | Ya |
| **V7** | P1 | Baseline End-to-End Tabular MLP, 5 seeds | 5 MLP runs | Ya |
| **V8** | P1 | Mechanism audit dan analisis statistik | 0 | Ya, post-hoc |
| **V9** | P2 | Ablation smoothing floor sampler terbaik | 9 GNN runs baru | Supplementary |
| **V10** | P2 | Sensitivitas fan-out | 18 GNN runs baru | Supplementary |

Jumlah run:

- **Eksperimen final utama V5–V7:** 38 training runs.
- **Termasuk smoke dan pilot V3–V4:** 44 runs.
- **Jika V9–V10 juga dijalankan:** 71 runs secara keseluruhan.

V3 dan V4 adalah quality gate. Angkanya tidak boleh digabungkan ke tabel hasil final V5–V8.

---

## 4. EXP21E-v3 — Corrected Sampler and Reproducibility Gate

### 4.1. Tujuan

Memastikan ketiga nama sampler benar-benar menghasilkan distribusi sampling yang berbeda sesuai definisinya, bukan hanya memberi atribut bobot yang tidak dipakai loader.

### 4.2. Perubahan wajib

1. Aktifkan atribut bobot secara eksplisit pada neighbor loader.
2. Redesign candidate score agar nilainya bervariasi di antara transaction candidates yang sedang berkompetisi:
   - saat mencuplik transaction melalui user, skor kandidat dapat memakai informasi counterpart merchant;
   - saat mencuplik transaction melalui merchant, skor kandidat dapat memakai informasi counterpart user;
   - seluruh informasi skor harus tersedia secara kausal.
3. Pisahkan state sampler untuk train, validation, dan test.
4. Tambahkan pengaturan seed untuk Python, NumPy, PyTorch CPU/CUDA, data loader, dan evaluation sampler.
5. Ganti pencarian threshold grid terpotong dengan candidate threshold dari validation precision-recall curve.
6. Tambahkan `transaction_id`, `user_id`, `merchant_id`, timestamp, degree, edge ID, dan sampled edge ID ke artefak audit.
7. Tambahkan resolved configuration, environment manifest, Git commit, dan dataset checksum.

### 4.3. Pengujian wajib

1. **Toy graph test:** distribusi probabilitas target diketahui secara analitis.
2. **Empirical sampling test:** lakukan minimal 10.000 draw per relation/case dan bandingkan frekuensi empiris dengan probabilitas target.
3. **Weight validation:** semua bobot finite, non-negative, dan mempunyai positive local sum.
4. **Non-constant candidate test:** neighborhood yang memerlukan sampling harus mempunyai minimal dua candidate weight berbeda untuk sampler berbobot.
5. **Sampler distinction test:** distribusi sampled degree dan sampled edge antara uniform dan sampler berbobot harus berbeda secara terukur.
6. **Reproducibility test:** seed dan konfigurasi identik menghasilkan sampled edge IDs, predictions, dan metrics yang identik dalam toleransi deterministik yang ditentukan.
7. **No-future test:** tidak ada sampled edge atau aggregate feature dengan timestamp melewati cutoff target.

### 4.4. Run

Jalankan tiga smoke runs pada subset tetap:

```text
uniform    x seed 42
topology   x seed 42
importance x seed 42
```

Smoke runs hanya menguji integrasi pipeline, bukan performa.

### 4.5. Acceptance criteria

- Seluruh unit/integration test lulus.
- Weighted sampler terbukti aktif.
- Candidate score tidak konstan pada neighborhood yang benar-benar dicuplik.
- Empirical inclusion frequency mengikuti expected probability dalam toleransi yang dicatat.
- Tidak ditemukan future information.
- Re-run identik tidak membuat artefak baru dengan hasil berbeda.

Jika satu kriteria gagal, V4 tidak boleh dimulai.

---

## 5. EXP21E-v4 — Causal Temporal Pilot

### 5.1. Tujuan

Memastikan pipeline terkoreksi dapat berjalan penuh dengan evaluasi temporal kausal sebelum menghabiskan komputasi untuk multi-seed.

### 5.2. Protokol

Gunakan **daily batch simulation** sebagai rancangan utama:

1. transaksi pada hari `d` dinilai menggunakan graph history sampai akhir hari `d-1`;
2. aggregate user/merchant dihitung hanya dari history tersebut;
3. seluruh transaksi hari `d` dinilai pada frozen historical graph;
4. setelah inference selesai, transaksi hari `d` dimasukkan ke history;
5. label yang belum tersedia tidak boleh dipakai untuk membangun sampling score.

Jika daily batch belum feasible, rolling snapshot boleh digunakan dengan aturan cutoff yang ekuivalen. Pilihan protokol harus dikunci sebelum run.

### 5.3. Run

```text
3 sampler x seed 42 = 3 pilot runs
```

### 5.4. Pemeriksaan hasil

- Pipeline selesai tanpa temporal leakage.
- Validation threshold dapat dihitung dari seluruh candidate threshold yang sah.
- Test hanya dievaluasi satu kali setelah checkpoint dan threshold dibekukan.
- Prediksi memiliki transaction ID dan dapat disejajarkan antarsampler.
- Runtime dan memory masih feasible untuk V5.
- Sampler audit menunjukkan neighborhood ketiga metode memang berbeda.

Jika perlu memperbaiki formula sampler atau protokol temporal setelah melihat hasil pilot, ulangi V4 dengan subversion baru, misalnya `v4.1`. Jangan langsung masuk V5.

---

## 6. EXP21E-v5 — Primary Multi-Seed Experiment

### 6.1. Tujuan

Menghasilkan tabel utama paper untuk perbandingan tiga sampler pada satu locked temporal window.

### 6.2. Matriks run

```text
Strategies : [uniform, topology, importance]
Seeds      : [42, 100, 2026, 3407, 8713]
Total      : 3 x 5 = 15 GNN runs
```

Semua pasangan strategy-seed memakai split manifest, model configuration, training budget, dan evaluation protocol yang sama.

### 6.3. Metrik per run

- validation AUPRC dan best epoch;
- test AUPRC;
- validation-selected threshold;
- F1, precision, recall, FP, FN, dan confusion counts;
- Recall@100, Recall@200, Recall@500, dan Recall@1000;
- alerts per captured fraud;
- training time, inference throughput, dan peak CPU/GPU memory;
- jumlah neighbor aktual per hop dan relation.

### 6.4. Pelaporan agregat

- mean dan standard deviation untuk setiap strategy;
- 95% confidence interval berbasis paired temporal-block bootstrap;
- paired difference terhadap uniform per seed;
- effect size, bukan hanya p-value;
- precision-recall curve agregat beserta confidence band.

### 6.5. Aturan pemilihan sampler terbaik

Sampler terbaik dipilih menggunakan urutan berikut:

1. mean validation AUPRC;
2. jika sangat berdekatan, gunakan operational recall pada validation alert budget;
3. stabilitas antarseed;
4. runtime dan memory sebagai tie-breaker.

Test set tidak dipakai untuk menentukan ranking V5. Karena selisih validation antarsampler kecil
dan tidak konsisten, revisi desain V6 mempertahankan ketiga sampler untuk robustness test.

### 6.6. Acceptance criteria

- Kelima paired seeds lengkap untuk ketiga sampler.
- Tidak ada run yang memakai konfigurasi berbeda tanpa dicatat sebagai failure/rerun.
- Seluruh artefak prediksi dapat disejajarkan berdasarkan transaction ID.
- Summary otomatis cocok dengan metrics JSON setiap run.
- Klaim keunggulan hanya dibuat jika arah perbedaan konsisten dan interval ketidakpastian dilaporkan.

---

## 7. EXP21E-v6 — Rolling-Origin Robustness

### 7.1. Tujuan

Menguji apakah hasil V5 bertahan pada perubahan waktu dan concept drift.

### 7.2. Pemilihan window

1. Buat seluruh candidate anchor yang memenuhi dukungan 300/30/8/60 hari dan memiliki fraud pada validation serta test.
2. Pilih dua additional anchors menggunakan aturan deterministik sebelum membuka performa test.
3. Simpan batas waktu dan jumlah transaksi/fraud ke `fold_manifest.json`.
4. Jangan memilih window karena memberikan metrik yang lebih baik.

### 7.3. Matriks run

Bandingkan ketiga sampler V5 pada dua window. Setiap eksekusi kernel hanya menjalankan satu
window aktif agar durasi aman dan pergantian window cukup dilakukan melalui konfigurasi:

```text
Per kernel: 3 strategies x 1 window x 3 seeds = 9 GNN runs
Keseluruhan: 3 strategies x 2 windows x 3 seeds = 18 GNN runs
Strategies: [uniform, topology, importance]
Seeds: [42, 100, 2026]
```

Jika anggaran komputasi memungkinkan, tingkatkan menjadi lima seed:

```text
3 strategies x 2 windows x 5 seeds = 30 GNN runs
```

Roadmap dan total minimum menggunakan konfigurasi 18 run. Window pertama memakai
`active_window_id: rolling_minus_180d`; window kedua memakai
`active_window_id: rolling_plus_180d` tanpa perubahan lain.

### 7.4. Acceptance criteria

- Semua window memakai protokol dan hyperparameter V5 yang telah dibekukan.
- Setiap kernel menyelesaikan tepat 9 run untuk satu window aktif.
- Tidak ada retuning per window.
- Hasil dilaporkan per window dan secara agregat; window yang buruk tidak boleh dihapus.
- Kesimpulan robustness mempertimbangkan arah efek, variasi antarseed, dan variasi antarwindow.

---

## 8. EXP21E-v7 — End-to-End Tabular MLP Baseline

### 8.1. Tujuan

Mengukur nilai tambah struktur graf terhadap neural baseline non-graf yang setara.

### 8.2. Desain

MLP menggunakan fitur transaksi yang sama dan sedekat mungkin dengan kapasitas model utama:

```text
Input -> Linear(128) -> BatchNorm -> ReLU -> Dropout(0.2)
      -> Linear(128) -> BatchNorm -> ReLU -> Dropout(0.2)
      -> Linear(1)
```

Protokol berikut harus sama dengan V5:

- split dan preprocessing;
- lima paired seeds;
- loss dan positive-class weighting;
- optimizer dan training budget yang adil;
- checkpoint berdasarkan validation AUPRC;
- threshold dipilih hanya dari validation;
- metrik dan alert budget.

### 8.3. Matriks run

```text
1 MLP x 5 seeds = 5 runs
```

### 8.4. Acceptance criteria

- Tidak ada fitur graph-derived atau future aggregate pada input MLP.
- Perbandingan utama memakai mean, standard deviation, dan paired seed policy yang sama.
- Perbedaan kapasitas parameter dicatat secara transparan.

---

## 9. EXP21E-v8 — Mechanism Audit and Statistical Analysis

### 9.1. Tujuan

Menjelaskan apakah perubahan neighborhood benar-benar berhubungan dengan perubahan representasi dan error model. V8 memakai artefak V5–V7 dan tidak melakukan training baru.

### 9.2. Analisis wajib

1. Intended sampling weight versus empirical inclusion probability.
2. Distribusi counterpart degree dari sampled neighbors per hop/relation.
3. Overlap sampled edge antara uniform dan sampler berbobot.
4. Performa berdasarkan user-degree dan merchant-degree quartile.
5. Performa pada super-hub entities.
6. FP/FN dan prediction margin pada setiap degree stratum.
7. Inter-class centroid distance dan Fisher ratio pada embedding 128-d.
8. Precision-recall dan cumulative fraud capture pada beberapa alert budget.
9. Paired bootstrap berdasarkan blok hari.
10. Analisis kasus transaksi yang konsisten salah atau berubah prediksi antarsampler.

Label fraud pada neighborhood hanya boleh digunakan untuk analisis post-hoc, tidak untuk sampling score atau feature construction.

### 9.3. Output utama

- satu main results table;
- precision-recall dan fraud-capture figure;
- sampler mechanism figure;
- degree-stratified error figure/table;
- machine-readable statistical report.

### 9.4. Acceptance criteria

- Setiap gambar dapat diregenerasi hanya dari manifest dan artefak run.
- Semua confidence interval menjelaskan unit resampling.
- Klaim mechanism hanya dibuat jika neighborhood memang berubah dan perubahan tersebut konsisten dengan perubahan representasi/error.

---

## 10. EXP21E-v9 — Smoothing-Floor Ablation (Opsional)

### 10.1. Tujuan

Menguji sensitivitas sampler terbaik terhadap smoothing floor `alpha` tanpa membuka ruang pencarian yang terlalu besar.

### 10.2. Matriks

```text
alpha : [0.00, 0.05, 0.10, 0.20]
seeds : [42, 100, 2026]
```

Konfigurasi utama `alpha = 0.10` dapat memakai tiga run V5 jika seluruh protokol identik. Dengan demikian:

```text
4 alpha x 3 seeds = 12 cells
3 cells reused from V5
9 GNN runs baru
```

V9 hanya dilakukan setelah V5 menunjukkan sampler berbobot layak diteruskan.

---

## 11. EXP21E-v10 — Fan-Out Sensitivity (Opsional)

### 11.1. Tujuan

Menguji apakah kesimpulan bertahan pada perubahan sampling budget.

### 11.2. Matriks

```text
fan-out    : [[5, 5], [10, 5], [15, 10], [25, 15]]
strategies : [uniform, best_weighted_sampler]
seeds      : [42, 100, 2026]
```

Total matriks adalah 24 cells. Enam cells `[15, 10]` dapat memakai V5, sehingga dibutuhkan **18 GNN runs baru**.

V10 ditempatkan pada supplementary/repository kecuali hasilnya sangat penting terhadap klaim utama.

---

## 12. Struktur Direktori dan Penamaan Artefak

Gunakan satu codebase yang diperbaiki, konfigurasi per versi, dan hasil yang immutable:

```text
code/exp21e/
  configs/
    v3_sampler_gate.yaml
    v4_causal_pilot.yaml
    v5_primary.yaml
    v6_rolling.yaml
    v7_mlp.yaml
    v9_alpha_ablation.yaml
    v10_fanout_ablation.yaml
  tests/
  main.py
  samplers.py
  run_experiment.py
  analyze_results.py

result/exp21e/
  legacy_pilot/
  v3_sampler_gate/
  v4_causal_pilot/
  v5_primary/
  v6_rolling/
  v7_mlp/
  v8_mechanism_audit/
  v9_alpha_ablation/
  v10_fanout_ablation/
```

Format run ID:

```text
exp21e_<version>_<window>_<strategy>_seed<seed>_<config_hash>
```

Setiap run minimal menyimpan:

```text
manifest.json
resolved_config.yaml
metrics.json
predictions.parquet atau predictions.npz
training_history.csv
sampler_audit.parquet
environment.json
best_checkpoint.pt
stdout.log
```

Checkpoint dapat dikecualikan dari Git, tetapi lokasi dan checksum-nya tetap dicatat pada manifest.

---

## 13. Urutan Eksekusi dan Stop Rules

```text
V3 sampler gate
  -> lulus: V4 causal pilot
  -> lulus: bekukan protocol
  -> V5 primary 15 runs
  -> pilih sampler berdasarkan validation
  -> V6 rolling robustness
  -> V7 MLP baseline
  -> V8 mechanism/statistical analysis
  -> bila perlu: V9 dan V10
```

Stop rules:

1. Hentikan sebelum V4 jika sampler berbobot belum terbukti aktif.
2. Hentikan sebelum V5 jika masih ditemukan future information atau hasil tidak reproducible.
3. Jangan menambah seed untuk menutupi hasil yang tidak konsisten; laporkan variasinya.
4. Jangan menjalankan seluruh ablation sebelum main experiment lengkap.
5. Jika dua sampler berbobot tidak mengungguli uniform secara stabil, ubah kontribusi paper menjadi audit empiris sampling, bukan klaim peningkatan performa.

---

## 14. Definition of Done untuk Paper

Eksperimen dianggap siap ditulis sebagai hasil utama jika:

- [ ] V3 sampler correctness dan reproducibility gate lulus.
- [ ] V4 causal pilot lulus tanpa future information.
- [ ] V5 mempunyai 15 paired GNN runs lengkap.
- [ ] V6 mempunyai 18 robustness runs lengkap dari dua eksekusi window.
- [ ] V7 mempunyai 5 MLP baseline runs lengkap.
- [ ] V8 menghasilkan mechanism audit dan statistical report yang dapat diregenerasi.
- [ ] Threshold seluruh model dipilih hanya dari validation.
- [ ] Semua test predictions memiliki transaction ID yang sama dan dapat disejajarkan.
- [ ] Main table melaporkan mean, standard deviation, dan confidence interval.
- [ ] Seluruh angka paper berasal dari manifest versi final yang sama.
- [ ] Klaim paper mengikuti kekuatan bukti dan tidak memakai hasil pilot lama sebagai golden result.

