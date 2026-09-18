# Knowledge Base Eksperimen EXP1-EXP14

Dokumen ini adalah sumber pengetahuan utama perjalanan penelitian klasifikasi
fraud kartu kredit dengan neighbor-sampling GraphSAGE. Isinya merangkum struktur,
fitur, protokol, hasil, alasan perubahan, kegagalan, dan batas klaim EXP1-EXP14.
Angka diambil dari artefak lokal `result/`; maksud desain diambil dari README,
konfigurasi, catatan audit, dan kode setiap eksperimen.

## 1. Tujuan penelitian

Pertanyaan utama: apakah neighbor sampling `topology` atau `importance` dapat
mendeteksi fraud lebih baik daripada `uniform` ketika model, data, split,
fanout, seed, training, dan evaluasinya sama?

Tiga strategi yang dibandingkan:

- **Uniform**: seluruh kandidat mempunyai peluang sama.
- **Topology-aware**: bobot menggabungkan kemiripan struktur dan riwayat label
  train lokal.
- **Importance-based**: bobot menggabungkan centrality dan personalized
  PageRank (PPR) lokal.

Selama penelitian muncul masalah pendukung: skala 24 juta transaksi, leakage
temporal, fraud yang sangat langka, validation-test drift, perubahan fraud
online/swipe ke chip, kebutuhan RAM/VRAM, dan pemisahan pengaruh sampler dari
pengaruh fitur serta protokol.

## 2. Data, graf, dan model

Dataset utama adalah `dataset/credit_card_transactions-ibm_v2.csv`, berisi
24.386.900 transaksi. Kolom sumber meliputi user, card, tanggal/waktu, amount,
channel, merchant, lokasi, MCC, error, dan label fraud.

Split full-data EXP9-EXP12:

| Split | Transaksi | Fraud | Periode |
|---|---:|---:|---|
| Train | 17.070.830 | 20.886 | 1991-01-02 - 2015-12-10 |
| Validation | 3.658.035 | 4.417 | 2015-12-10 - 2018-01-27 |
| Test | 3.658.035 | 4.454 | 2018-01-27 - 2020-02-28 |

`max_rows` pada eksperimen awal adalah prefix baris CSV, bukan random sample.
Prefix terutama layak untuk smoke test karena CSV dapat terurut menurut user.
EXP14 membuat batas timestamp tie-safe agar timestamp yang sama tidak masuk dua
split berbeda.

Graf konseptual:

```text
User <-> Transaction <-> Merchant
```

Setiap transaksi punya tepat dua endpoint. Akibatnya, fanout pada langkah
Transaction ke Entity tidak memangkas tetangga; strategi sampling terutama
berbeda ketika Entity memilih transaksi historisnya.

Arsitektur dominan:

- dua layer mean GraphSAGE, hidden 256;
- ReLU dan dropout 0,2;
- BatchNorm pada generasi awal/EXP9/EXP10/EXP13/EXP14 R0; LayerNorm pada EXP11-12;
- classifier linear satu logit;
- Adam, learning rate 0,001;
- fanout `[25, 10]`;
- BCE-with-logits dan sigmoid saat inferensi;
- checkpoint dipilih menggunakan validation, bukan test.

## 3. Definisi sampler

### Uniform

Semua kandidat berbobot satu. Jika degree <= fanout, semua dipakai; jika lebih
besar, dipilih tanpa penggantian. Ini kontrol utama penelitian.

### Topology

Rumus umum:

```text
w(v,u) = alpha * J(v,u) + (1-alpha) * H(v,u), alpha = 0,5
```

Jaccard one-hop literal sering nol pada graf berbeda tipe. EXP1-EXP8 memakai
fallback reciprocal degree. Sejak EXP9, mode `historical` mendefinisikan Jaccard
dari himpunan transaksi historis kedua endpoint. Homophily hanya memakai label
train, leave-one-candidate-out, smoothing 20, dan balanced-prior correction agar
dominasi kelas normal tidak membuat semua entity tampak serupa. Label validation
dan test tidak boleh masuk bobot.

### Importance

```text
w(e,t) = gamma * centrality(t) + (1-gamma) * PPR3(e,t)
gamma = 0,3; restart beta = 0,15
```

PPR dihitung tiga langkah. Degree transaksi literal selalu dua, sehingga versi
berikutnya menyediakan projected transaction degree berdasarkan transaksi yang
berbagi endpoint. EXP7-EXP8 berfokus membuat PPR eksak muat dan cepat di GPU.

Perbandingan sampler hanya fair jika fitur, split, model, root sampling, loss,
seed, threshold, dan protokol sama. Jika komponen itu berbeda, perbandingan
lintas EXP hanya deskriptif.

## 4. Evolusi fitur

### EXP1-EXP9: 9 fitur proposal/legacy

| Kelompok | Fitur | Tujuan |
|---|---|---|
| Nilai | amount | Besaran transaksi |
| Waktu | hour, day-of-week, month, weekend | Pola temporal |
| Kategori | Use Chip, MCC, Errors? | Kanal, merchant category, error |
| Lokasi | merchant location frequency | Sinyal lokasi langka |

Numerik diimputasi/min-max memakai train. Kategori memakai kamus train dan OOV
0. Lokasi memakai frequency encoding train. EXP1 menambah log-degree dan tiga
indikator tipe node, sehingga input homogen berukuran 13. Kelemahannya: kategori
ordinal seolah mempunyai jarak numerik dan tidak ada konteks kebiasaan user/card.

### EXP10: 155 fitur robust

- signed log amount, robust center/IQR, missing, refund;
- hour/day/month sin-cos, weekend;
- online dan has-error;
- location surprisal dan location OOV;
- one-hot `Use Chip`, `MCC`, `Errors?`, termasuk missing/OOV.

Alasannya: sin-cos menjaga siklus waktu, log amount meredam heavy tail, dan
one-hot tidak menciptakan urutan palsu antar kategori. Fitur disimpan float16,
tetapi transformasi dan model memakai float32.

### EXP11: 179 fitur behavioral

EXP11 menambah 24 fitur strictly-past:

- user/card: prior count, cold-start, seconds since previous, count 1h/24h/7d,
  amount delta, amount z-score;
- novelty dan surprisal user terhadap merchant, MCC, channel, dan location.

History memakai `[awal, query_time)`, sehingga transaksi bertimestamp sama tidak
saling membaca. Fitur tanpa label ini menangkap velocity, novelty, dan anomali
relatif terhadap perilaku pemilik kartu.

### EXP12 dan EXP14: 220 fitur contextual

EXP12 menambah 41 fitur:

- baseline amount 7 hari user/card: delta, z-score, support;
- hour deviation/concentration dan aktivitas 24h relatif terhadap 7d;
- context card-channel, card-MCC, user-merchant, merchant-channel: support,
  cold-start, recency, amount delta, amount z-score;
- novelty/surprisal card terhadap merchant/MCC/channel/location;
- indikator card missing.

Alasannya: lifetime baseline EXP11 dapat didominasi transaksi lama. Context
lokal lebih peka terhadap perubahan terbaru dan fraud chip. EXP14 menyimpan 220
fitur sebagai memmap float16; merchant-channel dihitung pada pass terpisah.

### EXP13: 15 fitur compact

EXP13 sengaja memakai fitur proposal-aligned yang lebih kecil agar aman pada
Kaggle T4. Karena berbeda dari 220 fitur EXP12, hasilnya tidak boleh dibandingkan
seolah hanya sampler atau hardware yang berubah. Tidak ada metrik lokal EXP13.

## 5. Evolusi training dan evaluasi

| EXP | Imbalance/root | Selection dan threshold | History graph |
|---|---|---|---|
| 1-2 | Root natural + inverse-frequency weight | AP validation; threshold 0,5 | Graf semua periode |
| 3-4 | Sama | RNG eval tetap, 3 pass, threshold F1 validation | Graf semua periode |
| 5 | Sama | Threshold 0,5 | Graf semua periode |
| 6-8 | Full positive weight + gradient clip | Threshold validation | Warisan graf awal |
| 9 | Root natural + positive weight | AP, threshold utama 0,5 | Frozen train graph |
| 10 | Balanced 20 normal/fraud; no double weight | Temporal AP lift; recent calibration | Frozen train graph |
| 11 | Balanced 50:1 + subtype weight | Recent selection + separate F2 calibration | Frozen graph, moving behavior |
| 12 | Balanced roots + ablation rasio | Recent weekly interleaved split | Frozen graph, contextual behavior |
| 13 | Root natural + train pos_weight | Separate AP selection/F2 calibration | Frozen, tie-safe |
| 14 R0 | n30 + corrected effective weight | Selection, FPR 0,1% calibration, assessment | Frozen cutoff, tie-safe |

AP/AUPRC menilai ranking dan tidak berubah oleh threshold. Precision, recall,
F1, FP, dan FN sangat bergantung threshold. Accuracy tidak cukup informatif
karena prevalensi fraud sekitar 0,12%.

## 6. Detail setiap eksperimen

### EXP1 - Starter GraphSAGE

**Dicoba:** baseline proposal dengan split 70/15/15, graf tiga tipe, dua layer,
fanout 25/10, weighted BCE, dan tiga sampler custom CPU dalam satu `run.py`.

**Hasil tersedia:** hanya uniform lima seed. AP test rata-rata 0,002955; terbaik
0,003299. Validation AP rata-rata 0,3283, sehingga gap sangat besar.

**Masalah:** sampler stateful yang sama dipakai train/validation/test. Validation
berubah tiap epoch dan menggeser RNG training. Struktur graf semua periode dapat
membawa konteks masa depan. Prefix 200 ribu tidak representatif.

**Alasan EXP2:** bottleneck sampling/training CPU; perlu menguji jalur GPU.

### EXP2 - GPU-bound

**Perubahan:** fitur, label, CSR, sampling, subgraph, forward/backward pindah ke
CUDA; AMP; batch 4.096.

**Hasil:** uniform lima seed, AP rata-rata 0,002411; terbaik 0,002827. Latensi
turun sekitar 30,67 ke 1,77 ms/1.000 node.

**Masalah:** candidate Gumbel top-k aproksimatif, kandidat dapat berulang sebelum
deduplikasi, VRAM naik, dan uniform tidak identik dengan EXP1. Kecepatan tidak
memperbaiki generalisasi.

**Alasan EXP3:** pisahkan masalah reliabilitas evaluasi dari optimasi GPU.

### EXP3 - Stable training/evaluation

**Perubahan:** sampler train/eval terpisah, seed evaluasi tetap, tiga pass
dirata-rata, `min_delta`, patience lebih panjang, threshold F1 dari validation.

**Hasil:** AP rata-rata 0,002865; terbaik 0,003483. Run terbaik mempunyai
recall/F1 nol pada threshold validation.

**Pelajaran:** evaluasi lebih reproducible, tetapi threshold sangat tinggi tidak
transfer ke test. Temporal drift tetap ada.

**Alasan EXP4:** gabungkan stabilitas EXP3 dengan throughput EXP2.

### EXP4 - GPU-bound stable evaluation

**Perubahan:** GPU aproksimatif EXP2 + kebijakan evaluasi EXP3.

**Hasil:** AP rata-rata 0,002640; terbaik 0,003089; F1/recall nol. Latensi sekitar
5,38 ms/1.000 node.

**Masalah:** sampler masih aproksimatif dan threshold validation gagal transfer.

**Alasan EXP5:** buat sampler GPU exact tanpa replacement dan kontrol threshold.

### EXP5 - Corrected GPU sampler

**Perubahan:** seluruh adjacency, exact without-replacement, duplicate edge
antar-hop dipertahankan, batch 1.024, FP32, RNG eval terpisah, threshold 0,5.

**Hasil tersimpan:** AP 0,012105, recall 0,72093, precision 0,002798, F1 0,005574.

**Pelajaran:** threshold 0,5 memberi recall tinggi tetapi false positive sangat
banyak. Run full yang dianalisis EXP6 mempunyai validation AP sekitar 0,668
namun test AP sekitar 0,024.

**Alasan EXP6:** kalibrasi threshold, gradient clipping, full-data monitoring,
dan pelaporan validation-test gap.

### EXP6 - Imbalance calibration

**Perubahan:** exact GPU sampler; threshold validation; gradient clip 5; satu
pass eval deterministik; TQDM; full class weight dipertahankan sebagai kontrol.

**Hasil seed 42:** uniform AP 0,023460; topology 0,023791. Topology precision
0,76984, recall 0,02178, F1 0,04236. Gap AP sekitar 0,64.

**Pelajaran:** topology hanya naik sekitar 1,4% relatif dan lebih mahal: peak
VRAM 7,75 vs 6,41 GiB serta durasi lebih panjang. Drift temporal lebih dominan.

**Alasan EXP7:** importance/PPR full-data berisiko OOM.

### EXP7 - Memory-bounded importance

**Perubahan:** PPR di-chunk; last hop memakai reverse adjacency; preprocessing
dapat di-cache.

**Hasil artefak subset/smoke:** AP uniform 0,010800; topology 0,010802;
importance 0,011209. Importance sekitar 382,93 ms/1.000 node vs uniform 3,48.

**Kegagalan:** OOM teratasi, tetapi loop per-root dan `.item()` membuat GPU
menunggu Python.

**Alasan EXP8:** batch root adaptif dan cache bobot tanpa mengubah rumus.

### EXP8 - Adaptive batched importance

**Perubahan:** adaptive root batching, slice hub, hilangkan sinkronisasi per-root,
cache bobot PPR+centrality per CSR edge.

**Hasil engineering:** smoke 100 ribu turun 568,89 menjadi 8,78 detik, 64,8x
lebih cepat, dengan VRAM hampir sama.

**Hasil full-data:** importance AP 0,023619, precision 0,77419, recall 0,02155,
F1 0,04194; gap validation-test 0,63623.

**Pelajaran:** performa komputasi membaik besar, kualitas tidak. Audit menemukan
edge antar-root tercampur dan graph context mencakup masa depan.

**Alasan EXP9:** tulis ulang modular, matematis eksplisit, dan anti-leakage.

### EXP9 - Reference sampling

**Perubahan:** modul data/graph/features/sampling/minibatch/model/trainer/eval;
frozen train graph; timestamp menit/detik; directed block; topology literal vs
historical; closed-form PPR tiga langkah; without-replacement.

**Fitur:** tetap 9 legacy agar koreksi sampler/protokol dapat diamati.

**Hasil seed 42:** uniform 0,025288; topology 0,025862; importance 0,025448.
Recall 14,4%-15,9%, tetapi precision sekitar 0,36%-0,43%.

**Masalah:** frozen history valid tetapi stale untuk test jauh di masa depan;
fitur legacy lemah; gap AP masih 0,62-0,63.

**Alasan EXP10:** perbaiki representasi fitur, root imbalance, normalization,
dan checkpoint temporal.

### EXP10 - Temporal robust

**Perubahan:** 155 fitur robust, storage float16, balanced roots 20:1,
`pos_weight=1`, weight decay, factorized mean-SAGE, refresh sampler per epoch,
selection temporal geometric AP lift.

**Hasil:** uniform AP 0,035268; topology 0,028505; importance 0,029190. Uniform
menang. Recall uniform 0,06533; F1 0,04786; gap AP sekitar 0,688.

**Pelajaran:** robust features/objective membantu baseline, custom sampler belum.
Audit menemukan perubahan subtype besar menuju fraud chip.

**Alasan EXP11:** tambah fitur anomali strictly-past dan validation lebih baru.

### EXP11 - Behavioral temporal

**Perubahan:** +24 behavioral features, LayerNorm, balanced roots 50:1, train-only
channel weight, minimum 20 epoch, recent selection dan calibration terpisah.

**Hasil:** uniform 0,174721; topology 0,187791; importance 0,153851. Topology F1
tertinggi 0,24707; importance recall tertinggi 0,25550.

**Interpretasi:** lompatan 4,96x-6,59x dari EXP10 menunjukkan fitur perilaku
membawa sinyal besar; bukan bukti sampler sendirian.

**Masalah:** selection hanya 25 fraud: 20 online, 3 swipe, 2 chip. Lifetime
baseline dapat didominasi histori lama; AP chip tetap rendah.

**Alasan EXP12:** recent 7-day context, conditional context, dan validation yang
lebih mewakili chip.

### EXP12 - Recent context

**Perubahan:** 220 fitur; per-row card missing; 25% validation terbaru dibagi
blok mingguan interleaved untuk checkpoint dan calibration; banyak ablation.

**Cohort fair seed 42** (`comparison_id=7bdae5fe466252c7`):

| Sampler | AP | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Uniform | 0,207345 | 0,251586 | 0,267176 | 0,259146 |
| Topology | 0,220823 | 0,304762 | 0,251459 | 0,275557 |
| Importance | 0,224387 | 0,294223 | 0,258419 | 0,275161 |

Importance naik 8,22% dan topology 6,50% AP relatif terhadap uniform, tetapi
hanya satu seed.

**Best tersimpan:** run uniform lain AP 0,248571, precision 0,42874, recall
0,23978, F1 0,30756. Ini hasil sweep, bukan cohort fair. Memilih N:P=25 dari AP
test adalah leakage; validation justru memilih N:P=30.

**Ablation pilot:** control AP 0,062566, features-only 0,166574, self-only
0,119762, main 0,166574. Contextual features adalah kontributor besar dan graph
memberi sinyal tambahan, tetapi prefix pilot bukan estimasi populasi.

**Masalah:** full run utama hanya seed 42, frozen entity context stale, test sudah
berulang kali dilihat, dan perubahan EXP9-12 confounded.

**Alasan EXP13:** deployment T4 out-of-core, tie-safe split, root natural,
multi-seed, dan artefak reproducible.

### EXP13 - Kaggle deployment

**Tujuan:** paket mandiri DuckDB external sort, memmap CPU, CSR, mini-batch GPU,
fitur compact, dan output audit untuk Kaggle T4.

**Perubahan:** 15 fitur, BatchNorm, root natural, dynamic train `pos_weight`,
projected degree importance, tie-safe split, selection/calibration terpisah.

**Status:** tidak ada metrik lokal di `result/`. EXP13 adalah eksperimen
engineering/deployment yang belum punya bukti empiris lokal, bukan skor nol.

**Alasan EXP14:** jalankan lokal dengan 220 fitur kuat dan protokol lebih ketat.

### EXP14 - Local temporal protocol

**R0:** membandingkan ketiga sampler dengan 220 fitur, model, split, fanout,
seed, objective, dan threshold yang sama.

**Engineering:** DuckDB external sort; memmap sekitar 11,33 GiB; 24.386.900
transaksi, 2.000 user, 100.343 merchant; preprocessing full sekitar 1.680,7
detik; observed peak RSS sekitar 11,8 GiB.

**Protokol:** checkpoint pada selection, threshold FPR 0,1% pada calibration,
kemudian assessment. R0 memakai static frozen graph, balanced n30, dan effective
class weight terkoreksi agar tidak double-count imbalance.

**Hasil R0 seed 42:**

| Sampler | AP | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Uniform | 0,187102 | 0,258621 | 0,259317 | 0,258969 |
| Topology | 0,157930 | 0,218816 | 0,232375 | 0,225392 |
| Importance | 0,143271 | 0,198164 | 0,232600 | 0,214005 |

Uniform menang pada seed ini; topology/importance tidak membuktikan peningkatan.
Research gate tetap memerlukan seed 42-46.

**Rolling analysis:** audit hanya menemukan satu origin 90 hari yang memenuhi
minimal 25 fraud dan 25 chip fraud per role sebelum gap label 2017. Satu origin
tidak cukup untuk klaim lintas periode.

**Batas:** R0 EXP14 tidak identik dengan EXP12 karena split tie-safe, threshold
FPR, effective weighting, dan assessment berbeda. Test 2018-2020 sudah diketahui.

## 7. Hasil terbaik per nomor EXP

Tabel ini untuk navigasi, **bukan leaderboard kausal**, karena protokol dan
kelengkapan seed berbeda.

| EXP | Run terpilih | AP | Precision | Recall | F1 |
|---:|---|---:|---:|---:|---:|
| 1 | uniform seed 45 | 0,003299 | 0,003745 | 0,050847 | 0,006977 |
| 2 | uniform seed 46 | 0,002827 | 0,004499 | 0,067797 | 0,008439 |
| 3 | uniform seed 42 | 0,003483 | 0 | 0 | 0 |
| 4 | uniform seed 42 | 0,003089 | 0 | 0 | 0 |
| 5 | uniform seed 42 | 0,012105 | 0,002798 | 0,720930 | 0,005574 |
| 6 | topology seed 42 | 0,023791 | 0,769841 | 0,021778 | 0,042358 |
| 7 | importance seed 42 | 0,011209 | 0,010101 | 0,023256 | 0,014085 |
| 8 | importance seed 42 | 0,023619 | 0,774194 | 0,021554 | 0,041940 |
| 9 | topology seed 42 | 0,025862 | 0,004139 | 0,144365 | 0,008048 |
| 10 | uniform seed 42 | 0,035268 | 0,037763 | 0,065335 | 0,047862 |
| 11 | topology seed 42 | 0,187791 | 0,303934 | 0,208128 | 0,247068 |
| 12 | uniform sweep seed 42 | 0,248571 | 0,428743 | 0,239784 | 0,307559 |
| 13 | belum ada hasil | - | - | - | - |
| 14 | uniform R0 seed 42 | 0,187102 | 0,258621 | 0,259317 | 0,258969 |

## 8. Rantai keputusan

```text
EXP1 baseline
 -> EXP2 GPU
 -> EXP3 stable evaluation
 -> EXP4 GPU + stable evaluation
 -> EXP5 exact GPU sampler
 -> EXP6 threshold/imbalance calibration
 -> EXP7 memory-bounded PPR
 -> EXP8 adaptive/cached PPR
 -> EXP9 modular frozen graph
 -> EXP10 robust features + balanced roots
 -> EXP11 strictly-past behavior
 -> EXP12 recent/conditional context
 -> EXP13 Kaggle out-of-core deployment
 -> EXP14 local full-feature temporal protocol
```

EXP2/4/7/8/13 terutama eksperimen engineering. EXP3/5/6/9-12/14 terutama
eksperimen validitas, fitur, imbalance, temporal generalization, dan sampler.

## 9. Penyebab kegagalan yang berulang

1. **Temporal drift.** Fraud train chip/online/swipe sekitar 256/14.771/5.859,
   sedangkan test 3.907/128/419. Model online yang kuat gagal pada chip.
2. **Validation mismatch.** EXP3/4 menghasilkan TP=0; selection EXP11 hampir
   tidak memuat chip. Threshold/checkpoint tidak mewakili target terbaru.
3. **Extreme imbalance.** Threshold 0,5 memberi recall tetapi FP besar; threshold
   tinggi memberi precision tetapi recall sekitar 2%. Balanced roots tidak boleh
   digabung dengan full inverse-frequency weight tanpa koreksi.
4. **Fitur awal lemah.** Fitur global tidak mengukur anomali relatif user/card.
   Lompatan EXP11-12 menunjukkan velocity, novelty, dan recency penting.
5. **Frozen graph stale.** Aman dari leakage, tetapi konteks entity test tetap
   berhenti pada cutoff train. Moving behavioral feature tidak memperbarui graph.
6. **Sampler bukan faktor dominan.** Perbedaannya sering kecil dibanding fitur
   dan protokol; EXP14 seed 42 justru dimenangkan uniform.
7. **Rumus literal degenerat.** Jaccard beda tipe nol; degree transaksi selalu
   dua. Adaptasi historical/projected harus dinyatakan eksplisit.
8. **Confounding.** EXP9-12 mengubah fitur, root ratio, normalization, loss, dan
   validation. Kenaikan lintas EXP tidak dapat seluruhnya dikreditkan ke sampler.
9. **Test reuse.** Test telah memandu desain dan sweep; klaim final perlu data
   atau periode baru.
10. **Kurang replikasi.** Mayoritas full custom-sampler baru seed 42; selisih
    kecil belum robust tanpa paired multi-seed.

## 10. Aturan interpretasi

1. Bandingkan sampler hanya dalam comparison/protokol yang sama.
2. Gunakan AP/AUPRC sebagai metrik ranking utama.
3. Sertakan precision, recall, F1, TP/FP/FN, FPR, dan alert rate.
4. Pisahkan metrik global, channel, dan periode.
5. Jangan membandingkan smoke prefix dengan full-data.
6. Jangan sebut hasil terbaik sweep sebagai fair sampler comparison.
7. Gunakan seed 42-46, mean/std, dan paired delta.
8. Pisahkan waktu preprocessing, weight build, refresh, training, dan inference.
9. Nyatakan graph frozen, moving, atau causal.
10. Perlakukan test 2018-2020 sebagai exploratory.

## 11. Kesimpulan

- Pipeline berkembang dari baseline CPU menjadi GPU/out-of-core yang dapat diaudit.
- Keberhasilan engineering terbesar: percepatan importance EXP8 dan local
  preprocessing 220 fitur EXP14.
- Kenaikan akurasi paling jelas terjadi saat fitur perilaku/context ditambahkan
  pada EXP11-12, bukan saat hanya mengganti sampler.
- EXP12 fair cohort seed 42 mendukung topology/importance; EXP14 R0 seed 42
  menghasilkan kebalikan. Protokolnya berbeda dan satu seed tidak cukup.
- Belum ada bukti final bahwa satu sampler selalu unggul.

Kesimpulan paling aman: **weighted neighbor sampling masih merupakan hipotesis,
sedangkan fitur temporal-perilaku dan desain evaluasi mempunyai pengaruh yang
lebih jelas pada hasil yang tersedia.**

## 12. Pekerjaan lanjutan prioritas

1. Selesaikan EXP14 R0 ketiga strategi pada seed 42-46.
2. Bekukan keputusan dari selection/development, bukan assessment/test.
3. Hitung paired delta dan bootstrap blok waktu/episode fraud.
4. Tambah self-only/MLP dan tree baseline pada fitur yang sama.
5. Uji rolling causal history pada beberapa origin atau dataset baru.
6. Lakukan ablation satu perubahan per run.
7. Audit chip fraud pada fixed FPR/alert budget.

## 13. Peta artefak

| Informasi | Sumber |
|---|---|
| Metrik semua run | `recap/experiment_metrics_detail.csv` |
| Best per EXP | `recap/experiment_best_by_experiment.csv` |
| Agregat strategi | `recap/experiment_strategy_summary.csv` |
| Grafik | `recap/figures/` |
| Ringkasan kegagalan | `recap/experiment_differences_and_failures.md` |
| Generator | `recap/generate_recap.py` |
| EXP1-8 | `code/exp*/README.md`, `config.yaml`, `run.py` |
| Pipeline EXP9 | `code/exp9_reference_sampling/` |
| Audit EXP9-12 | `ANALISIS_EXP9.md`, `ANALISIS_EXP10.md`, `ANALISIS_EXP11.md` |
| Audit EXP13 | `code/exp13_try_kaggle/PRIOR_EXPERIMENT_AUDIT.md` |
| EXP14 | `code/exp14_local_temporal/PLAN.md`, `README.md`, `VALIDATION.md` |
| Metrik mentah | `result/exp*/` |

## 14. Glosarium

- **AP/AUPRC**: area precision-recall; kualitas ranking pada kelas langka.
- **Root**: transaksi target mini-batch.
- **Context**: node historis untuk message passing.
- **Fanout**: batas tetangga per hop.
- **Frozen graph**: adjacency berhenti di cutoff train.
- **Strictly-past**: hanya memakai observasi sebelum waktu query.
- **Selection**: subset untuk checkpoint.
- **Calibration**: subset terpisah untuk threshold.
- **Assessment**: dibuka setelah model dan threshold terkunci.
- **n30**: seluruh fraud dan maksimal 30 normal per fraud sebagai root train.
- **OOV**: kategori tidak terlihat saat fit train.
- **Concept drift**: hubungan fitur-label berubah terhadap waktu.
- **Leakage**: masa depan/holdout memengaruhi training atau pemilihan keputusan.

