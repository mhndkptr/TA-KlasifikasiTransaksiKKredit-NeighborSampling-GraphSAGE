# EXP9 - GraphSAGE dengan sampling berbasis referensi

EXP9 mengimplementasikan penelitian **Klasifikasi Transaksi Penipuan Kartu Kredit
Skala Besar Menggunakan Pendekatan Neighbor Sampling GraphSAGE** dalam proposal
Muhammad Hendika Putra. Tujuannya adalah memperbaiki dasar implementasi EXP8 dan
menyediakan perbandingan `uniform`, `topology`, dan `importance` pada **model
GraphSAGE yang sama**.

Kode EXP9 mandiri: tidak mengimpor `run.py` eksperimen sebelumnya dan tidak
memerlukan instalasi TensorFlow/DGL/StellarGraph dari repository referensi.
`run.py` hanya entry point; setiap tahap memiliki modul tersendiri.

**Status:** implementasi dan pengujian pipeline tersedia. Peningkatan AUPRC, F1,
atau recall atas EXP8 pada seluruh dataset **belum terbukti**. Smoke test adalah
pemeriksaan teknis, bukan hasil akhir penelitian. Lihat [VALIDATION.md](VALIDATION.md).

## 1. Temuan EXP8 dan alasan perbaikan

Sumber historis: [exp8_importance_seed42.json](../../result/exp8_importance_seed42.json).

| Ukuran | EXP8 tersimpan, full-data importance seed 42 |
|---|---:|
| Train / validation / test | 17.070.830 / 3.658.035 / 3.658.035 transaksi |
| Fraud train / validation / test | 20.886 / 4.417 / 4.454 |
| Best validation AUPRC | 0,659854 |
| Test AUPRC | 0,023619 |
| Test recall pada threshold validation 0,998965 | 0,021554 |
| Test F1 pada threshold validation | 0,041940 |
| Test recall pada threshold 0,5 | 0,091379 |
| Test F1 pada threshold 0,5 | 0,004909 |
| Durasi run | 87.891,97 detik, sekitar 24,4 jam |
| Peak VRAM yang dicatat | 6,4098 GiB |

Selisih validation-test sangat besar. Kalibrasi threshold dapat mengubah
precision/recall, tetapi tidak memperbaiki ranking yang dinilai AUPRC. GPU run
historis tidak diidentifikasi lengkap dalam JSON; jangan mengasumsikan perangkatnya
sama dengan perangkat smoke test EXP9.

Audit kode menemukan beberapa hal berikut:

1. **Edge antar-root tercampur.** `ExactGPUSampler.sample()` yang diwarisi EXP8
   menggabungkan edge seluruh root, menambahkan arah balik, dan mempertahankan
   duplikasi antar-hop. Hub bisa menerima pesan dari transaksi root lain dalam
   batch meskipun transaksi itu tidak dipilih sebagai konteks hub. Batas fanout
   tidak lagi langsung menjadi batas pesan per lapisan.
2. **Konteks graf mencakup masa depan.** Degree dan adjacency EXP8 berasal dari
   seluruh dataset. Walaupun label validation/test dimasking, fitur dan struktur
   transaksi tersebut dapat masuk message passing training. Masking label saja
   belum mewujudkan evaluasi induktif dengan periode masa depan yang terpisah.
3. **Pekerjaan konteks berulang.** Cache bobot PPR EXP8 membantu, tetapi sampling
   adjacency hub dan penghitungan embedding konteks masih diulang di banyak batch.
4. **Menit transaksi diabaikan.** Parser lama menggunakan jam saja saat mengurutkan
   waktu. EXP9 mempertahankan menit dan detik jika tersedia.
5. **Label entitas buatan.** Majority vote pada user/merchant mudah menghasilkan
   hampir semua label normal. Proposal hanya memberi label pada transaksi;
   homophily antar-endpoint bertipe berbeda perlu definisi operasional yang jelas.
6. **Pemilihan checkpoint.** EXP9 menyimpan setiap maksimum validation AUPRC baru;
   `min_delta` hanya mengendalikan patience. Audit drift juga dibuat lebih lengkap.

Ini alasan teknis perbaikan, bukan bukti bahwa satu persoalan tertentu menyebabkan
seluruh selisih generalisasi pada hasil EXP8.

## 2. Pemetaan ke proposal

Dokumen acuan adalah **Proposal Final - Muhammad Hendika Putra.pdf** yang
dilampirkan pengguna. Nomor halaman berikut memakai nomor tercetak, bukan indeks PDF.

| Bagian proposal | Implementasi EXP9 |
|---|---|
| §1.3, hlm. 4 | Klasifikasi transaksi dengan GraphSAGE; model tidak diganti menjadi CARE-GNN/PC-GNN/SplitGNN |
| §3.2 dan Tabel 3.1, hlm. 24-25 | IBM CSV, split temporal 70/15/15, encoder fit train |
| §3.3, hlm. 25-27 | User--Transaction--Merchant; satu user dan merchant per transaksi; HeteroData tersedia untuk audit |
| §3.4, hlm. 28 | Dua mean GraphSAGE, hidden 256, BatchNorm, ReLU, classifier skalar |
| §3.5, hlm. 30-34 | Tiga strategi, tanpa penggantian, fanout 25/10, alpha 0,5, gamma 0,3, beta PPR 0,15 |
| §3.6, hlm. 33-35 | Batch 1.024, weighted BCE invers frekuensi train, Adam 0,001, scheduler faktor 0,5/patience 5, early stopping patience 15 |
| §3.7, hlm. 35-38 | F1, recall, AUPRC, confusion matrix, long-tail, latensi per 1.000 transaksi, lima seed |
| Penjelasan Gambar 3.6, hlm. 37 | Threshold utama **0,5**; threshold validation menjadi evaluasi tambahan terpisah |

Keputusan implementasi yang perlu dicatat dalam laporan penelitian:

- **70/15/15 versus 80/10/10:** Algoritma 5 menyebut 80/10/10; Tabel 3.1
  menjelaskan 70/15/15. Default mengikuti tabel metodologi. Perubahan split harus
  berlaku pada semua strategi dan memerlukan run ulang.
- **Dimensi fitur:** Gambar 3.2 menulis `R^10`, tetapi daftar rekayasa fitur
  eksplisit, ditambah `Errors?` seperti EXP8, menghasilkan sembilan fitur transaksi.
  EXP9 mencatat nama aktual dan tidak menambahkan fitur kesepuluh tanpa definisi.
  Padding degree dan tipe menambah empat dimensi: `input_channels=13`.
- **Dropout 0,2** dipertahankan dari EXP8 untuk semua strategi.
- **Custom sampler:** EXP9 menggunakan sampler sendiri dengan `SAGEConv` PyG,
  bukan subclass internal `NeighborSampler`. Blok pesan dan distribusi sampling
  tetap eksplisit, tanpa ekstensi sampling native tambahan.
- **Topology historical** adalah adaptasi operasional pada graf bertipe,
  dijelaskan di §6. Mode literal juga tersedia. Keduanya dibedakan dalam artefak.
- **PPR tiga langkah** mengikuti EXP8. Nilai langkah selain tiga ditolak karena
  rumus tertutup yang diimplementasikan memang khusus untuk tiga langkah.

## 3. Penggunaan ketujuh repository referensi

Referensi digunakan untuk memilih komponen yang sesuai ruang lingkup. Kode EXP9
ditulis sebagai implementasi mandiri, bukan gabungan seluruh arsitektur referensi.
Permalink berikut memuat commit checkout lokal yang ditinjau.

| Referensi dan sumber kode | Penerapan | Batas penerapan |
|---|---|---|
| [GraphSAGE/models.py](https://github.com/williamleif/GraphSAGE/blob/a0fdef95dca7b456dab01cb35034717c8b6dd017/graphsage/models.py), `aggregators.py`, `minibatch.py` | Pemisahan model/sampler/minibatch, backward expansion lalu forward aggregation per lapisan, jalur self dan mean, pemisahan graf train | Tidak mengimpor TensorFlow atau mengganti mean dengan LSTM/pooling |
| [CARE-GNN/layers.py](https://github.com/YingtongDou/CARE-GNN/blob/a64ff7523e187a24251f7ca88435d2c9d8f7dcd9/layers.py), `README.md` | Prinsip seleksi tetangga dengan informasi label yang tersedia; pemisahan selection dan aggregation | EXP9 menggunakan statistik history, bukan classifier similarity CARE; tanpa RL/top-p/inter-relation aggregator |
| [DGFraud/GraphConsis/neigh_samplers.py](https://github.com/safe-graph/DGFraud/blob/22b72d75f81dd057762f0c7225a4558a25095b8f/algorithms/GraphConsis/neigh_samplers.py), struktur `algorithms/GraphSage/` | Organisasi modul dan perbandingan dengan konfigurasi/evaluasi konsisten; distribusi sampling harus dinyatakan jelas | Distance sampler GraphConsis tidak dijadikan strategi keempat |
| [GraphSAINT/minibatch.py](https://github.com/GraphSAINT/GraphSAINT/blob/c9b1e340d7b951465ac4a9251eef93832e68b003/graphsaint/pytorch_version/minibatch.py) | CSR, graf train terpisah, mengeluarkan komputasi berulang dari minibatch, memisahkan biaya persiapan dari inferensi | Tidak mengklaim sampler ini sebagai GraphSAINT; tidak memakai estimator/normalisasi subgraph GraphSAINT |
| [Inductive GRL/timeframes.py](https://github.com/Charlesvandamme/Inductive-Graph-Representation-Learning-for-Fraud-Detection/blob/b60d1fa93dce25b536aaf90c7860076f8d9377b1/inductiveGRL/timeframes.py), `graphconstruction.py`, `README.md` | Graf tiga tipe, komponen pipeline modular, pemisahan periode train dari transaksi induktif | Tidak memakai classifier XGBoost, rolling-window, atau graph undersampling pada eksperimen utama |
| [PC-GNN/src/utils.py](https://github.com/PonderLY/PC-GNN/blob/9d7d7fae491081178b6e12e193fb89cfc330f2be/src/utils.py), `pick_step`, `layers.py` | Prinsip koreksi frekuensi kelas train diadaptasi menjadi balanced-prior affinity pada topology | Bukan algoritma pick-step PC-GNN; root tetap diacak tanpa oversampling dan loss sama untuk semua strategi |
| [SplitGNN/src/model.py](https://github.com/Split-GNN/SplitGNN/blob/32fbdb836781d5c02523844a7079425f89be8831/src/model.py), `README.md` | Landasan audit heterophily/long-tail dan perhatian terhadap sinyal self transaksi | Tidak memakai spectral filter, relation classifier, atau cabang tambahan; self path tetap milik GraphSAGE standar |

Semua repository referensi dibaca dari checkout lokal `Ref/` dan tidak diubah.
Semantik GraphSAGE juga diperiksa terhadap
[paper asli](https://cs.stanford.edu/~jure/pubs/graphsage-nips17.pdf) dan dukungan
[masukan bipartit SAGEConv](https://pytorch-geometric.readthedocs.io/en/stable/cheatsheet/gnn_cheatsheet.html).

## 4. Struktur kode

```text
exp9_reference_sampling/
  run.py                         # entry point pendek
  compare.py                     # membandingkan JSON pada threshold 0,5
  config.yaml                    # full-data, tiga strategi x lima seed
  config.smoke.yaml              # 100.000 baris, dua epoch, satu seed
  config.literal.yaml            # audit topology literal
  config.exp8_batch.yaml          # ablation batch size 4.096
  config.unbalanced_topology.yaml # ablation koreksi prior kelas
  requirements.txt
  exp9/
    cli.py                       # CLI dan urutan eksperimen
    config.py                    # inheritance, path, validasi konfigurasi
    data.py                      # CSV, waktu, fit/transform fitur
    graph.py                     # TransactionGraph, CSR, HeteroData, FeatureStore
    sampling/
      weights.py                 # uniform/topology/PPR closed-form
      tables.py                  # Gumbel top-k, chunk dan streaming hub
    minibatch.py                 # directed Block per lapisan
    model.py                     # GraphSAGE yang sama untuk semua strategi
    trainer.py                   # loss, optimizer, checkpoint, early stopping
    evaluation.py                # cache embedding, prediksi, latensi
    metrics.py                   # metrik, kalibrasi, long-tail, temporal bins
    artifacts.py                 # fingerprint, cache, JSON, checkpoint, ringkasan
    runtime.py                   # seed, perangkat, versi, timer
  tests/test_exp9.py              # pengujian matematis dan integrasi
```

Alur eksekusi: `CLI -> preprocessing/cache -> frozen training graph -> bobot ->
tabel tetangga -> blok per lapisan -> GraphSAGE -> validation/checkpoint ->
kalibrasi validation -> test -> laporan`.

Konfigurasi mendukung `extends`. Setiap path relatif dihitung terhadap file
konfigurasi yang menuliskannya; perintah bisa dijalankan dari root atau folder EXP9.

## 5. Preprocessing dan protokol temporal

### Preprocessing

- `User` dan `Merchant Name` hanya menjadi kunci graf. `Card` tidak menjadi fitur
  langsung atau tipe node tambahan.
- Waktu diurutkan stabil sampai menit/detik; timestamp invalid ditolak.
  `source_rows` menyimpan nomor baris CSV untuk audit.
- Amount dibersihkan dari `$` dan pemisah ribuan. Median/modus imputasi hanya fit
  train. Numerik yang seluruhnya kosong pada train mendapat fallback 0 tersimpan.
- Numerik diskalakan dengan min-max train seperti EXP8. Nilai holdout di luar
  rentang train tidak dipotong diam-diam; persentasenya dicatat sebagai drift.
- `Use Chip`, `MCC`, `Errors?` memakai label encoding train. Kategori baru berkode
  OOV 0. Gabungan lokasi merchant memakai frequency encoding train.
- Setiap split harus berisi normal dan fraud; subset tanpa fraud ditolak.

`max_rows` berarti **prefix baris CSV, kemudian diurutkan**. Ini bukan sampel acak
representatif atau N transaksi paling awal dari seluruh CSV. IBM CSV dapat
terurut berdasarkan pengguna; subset terutama dipakai untuk pemeriksaan pipeline.
Default penelitian adalah `max_rows: null`.

### Frozen training history

CSR hanya memuat relasi **entitas -> transaksi train**. Transaksi query tetap
terhubung ke user dan merchant, tetapi tidak ditambahkan ke history entitas.

| Operasi | Informasi yang digunakan |
|---|---|
| Scaler, kategori, frequency encoding | Train |
| Degree, adjacency history, jumlah pasangan user-merchant | Train |
| Homophily, prior kelas, PPR | Train |
| Update model dan BatchNorm | Root dan konteks train |
| Checkpoint, scheduler, threshold tambahan | Validation |
| Metrik akhir dan diagnostik test | Test setelah checkpoint terpilih |

Entitas baru memiliki degree 0 dan tetangga kosong; jalur self, tipe node dan bias
tetap memberi representasi yang terdefinisi. Tingkat entitas baru dicatat.

Ini **protokol induktif dengan satu snapshot training yang dibekukan**, bukan
simulasi kausal per transaksi dalam training. Transaksi train bisa memakai
transaksi lain dari periode train yang sama. Validation tidak ditambahkan sebagai
history test. Konteks test dapat menjadi usang; rolling-window perlu eksperimen
tersendiri dengan aturan yang sama untuk semua strategi.

Jika timestamp sama tepat di batas split, pembagian tetap memakai jumlah baris
dengan urutan stabil. `tie_at_split_boundary` mencatatnya: waktu tidak menurun,
tetapi pemisahan timestamp tidak selalu ketat.

### Memori

Graf menyimpan fitur transaksi float32, dua ID entitas per transaksi, CSR entitas,
label, timestamp, source-row index, degree dan pair count. Tidak ada adjacency
Python untuk setiap transaksi atau empat salinan edge penuh di GPU. `HeteroData`
dapat dimaterialisasi untuk audit dengan `--export-heterodata`.

Default `feature_device: cpu` menyimpan fitur dalam RAM; hanya fitur batch masuk
GPU. Opsi `cuda` dapat mengurangi transfer jika VRAM mencukupi. Gunakan opsi sama
untuk perbandingan latensi.

Preprocessing masih membaca dan mengurutkan CSV dalam RAM: **belum out-of-core**.
Fitur 24,39 juta x 9 float32 saja sekitar 0,88 GB desimal, belum termasuk DataFrame
string, sorting, ID dan CSR. RAM preprocessing bisa jauh lebih besar. Batas chunk
sampler bukan batas seluruh penggunaan RAM.

## 6. Definisi tiga strategi

Semua strategi mengambil `min(k, degree)` tetangga tanpa penggantian. Tidak ada
oversampling fraud, edge buatan, focal loss, atau arsitektur tambahan pada
eksperimen utama. Loss, model, batch size, split dan seed sama.

### Uniform

Bobot setiap kandidat adalah 1. Tetangga diambil semua jika degree <= fanout.
Slot kosong bertanda -1 selalu dimasking sebelum agregasi.

### Topology literal

Algoritma 3 proposal menuliskan:

```text
J(v,u) = |N(v) intersect N(u)| / |N(v) union N(u)|
H(v,u) = |{w in N(u): L[w] = L[v]}| / (|N(u)| + 1)
w(v,u) = alpha * J(v,u) + (1-alpha) * H(v,u)
```

Pada Transaction--Entity, tetangga endpoint berada pada partisi berbeda, sehingga
Jaccard langsung nol. User/merchant juga tidak memiliki label ground-truth.
Root yang memerlukan sampling berdegree besar adalah entitas; transaksi hanya
memiliki dua tetangga.

Mode `literal` tidak menganggap dua label unknown sebagai kecocokan. Semua bobot
menjadi nol dan fallback yang terdefinisi adalah uniform. Tidak dibuat label
normal palsu pada entitas. Mode ini memperlihatkan keterbatasan rumus pada graf
penelitian, bukan menyembunyikannya sebagai peningkatan topology.

### Topology historical: adaptasi utama

Default `historical` mempertahankan kombinasi kemiripan struktur dan riwayat label,
dengan definisi yang dapat dipakai pada graf bertipe. Untuk entitas pusat `e`,
kandidat transaksi train `t`, dan entitas lain `o(t)`:

```text
d_e, d_o = jumlah transaksi train pada kedua entitas
c_eo = jumlah transaksi train yang menghubungkan pasangan e dan o
J_hist(e,t) = c_eo / (d_e + d_o - c_eo)
```

Jaccard ini membandingkan himpunan transaksi dua entitas. Proyeksi hanya dipakai
untuk skor; message passing tetap pada User--Transaction--Merchant.

Untuk label, `f_e` adalah jumlah fraud train pada entitas, `y_t` label kandidat,
`F/N` prevalensi train, dan `s=20` pseudo-count:

```text
pi_minus_t = (F-y_t)/(N-1)
p_e = (f_e-y_t + s*pi_minus_t)/(d_e-1+s)
p_o = (f_o-y_t + s*pi_minus_t)/(d_o-1+s)
```

Label kandidat dikeluarkan dari kedua hitungan dan prior. Ini
**leave-one-candidate-out**, bukan cross-fitting atau penghapusan semua label root
dari seluruh konteks. Prior diberi batas numerik `[1e-9, 1-1e-9]`.

Dengan `topology_balance: true`, probabilitas diubah ke prior kelas seimbang:

```text
q(p) = p*(1-pi) / (p*(1-pi) + (1-p)*pi)
H_hist(e,t) = q(p_e)*q(p_o) + (1-q(p_e))*(1-q(p_o))
w(e,t) = alpha*J_hist(e,t) + (1-alpha)*H_hist(e,t)
```

Tanpa koreksi, hampir semua entitas terlihat mirip karena sama-sama didominasi
normal. Koreksi menjaga variasi relatif riwayat fraud. Jika `topology_balance`
false, H langsung memakai p. Preset unbalanced disediakan untuk mengukur kontribusi
koreksi tersebut.

Historical topology adalah adaptasi EXP9 yang terinspirasi seleksi CARE-GNN dan
perhatian pada imbalance PC-GNN. **Ini bukan rumus CARE-GNN, PC-GNN, atau salinan
persis Algoritma 3 proposal.** Mode dan parameter disimpan dalam hasil, serta perlu
dituliskan pada laporan TA. Skor tinggi tidak menjamin tetangga fraud ataupun
peningkatan recall; keduanya tetap hipotesis yang diuji.

### Importance: PPR tiga langkah eksak

Komposisi bobot tetap mengikuti proposal:

```text
w(e,t) = gamma * degree(t)/(V_train-1) + (1-gamma) * PPR_3(e,t)
gamma = 0.3; beta = 0.15; a = 1-beta
PPR_3(e,t) = a*beta/d_e + a^3/(2*d_e) * (1+c_eo/d_o)
```

Turunannya memakai restart propagation yang sama dengan EXP8. Setelah dua
langkah, untuk entitas `e'`:

```text
p_2[e'] = beta * 1[e'=e] + a^2*c(e,e')/(2*d_e)
p_3[t] = a * (p_2[e]/d_e + p_2[o]/d_o)
```

Untuk `e'=e`, jumlah transaksi bersama adalah `d_e`; substitusi menghasilkan rumus
tertutup di atas. Rumus eksak secara matematis untuk **graf history tak berarah,
transaksi degree dua, tiga langkah, kandidat transaksi**. Nilainya diuji terhadap
propagasi matriks dense independen. Tidak ada Monte Carlo PPR, pruning tambahan,
atau pemotongan kandidat hub.

`V_train` hanya menghitung transaksi dan entitas aktif training; entitas masa
depan tidak memengaruhi centrality. Semua kandidat transaksi memiliki degree dua,
sehingga centrality konstan dalam baris dan perbedaan importance terutama berasal
dari struktur pasangan pada PPR. Jika struktur serupa, distribusi dapat mendekati
uniform. Batas representasi ini perlu dilaporkan, bukan menjanjikan importance
selalu unggul. Rumus tidak berlaku otomatis pada tipe edge atau jumlah langkah lain.

## 7. Tabel sampling, blok per lapisan, dan cache inferensi

### Tabel per epoch

Skor Gumbel `log(weight)-log(-log(U))` diurutkan dan k terbesar dipilih. Distribusi
weighted without replacement sama secara matematis dengan EXP8, dengan batas
presisi komputer. Yang berubah adalah jadwal: satu sampel per entitas dipakai
bersama sepanjang satu epoch, kemudian diganti. Marginal sampling tetap sesuai
bobot, tetapi korelasi antar-batch berbeda. EXP9 bukan reproduksi RNG atau dinamika
training EXP8. Default `refresh_epochs: 1`.

Row kecil dikemas sampai `edge_chunk_size`; hub besar di-stream per slice sambil
menyimpan k skor terbesar sejauh ini. Semua kandidat tetap berpeluang. Tensor
temporer dibatasi sekitar `edge_chunk_size + fanout`; bobot CPU tetap O(E_train)
dan tabel O(jumlah_entitas * fanout).

### Fanout dan blok pesan

`fanouts: [25,10]` mengikuti **urutan forward layer**. Lapisan pertama mengagregasi
maksimum 25 transaksi history per entitas; lapisan kedua mengambil maksimum 10
entitas per transaksi, tetapi graf hanya menyediakan dua sehingga keduanya diambil.

Ekspansi dilakukan dari output menuju input, kemudian forward dijalankan
sebaliknya. Setiap blok memiliki source, destination dan self index. Edge pesan
hanya dari source ke destination pada blok terkait. Sharing node konteks tidak
menambahkan root batch lain atau edge balik ke konteks sebuah entitas.

### Cache embedding evaluasi

Pada `model.eval()`, tabel validation/test tetap untuk setiap seed dan pass.
Embedding lapisan pertama entitas dihitung sekali dari tabel tersebut. Setiap
query menghitung jalur self lapisan pertama menggunakan fitur transaksi dan mean
fitur mentah dua entitas, lalu lapisan kedua menggunakan mean embedding entitas
yang sudah dihitung.

Hasil ekuivalen dengan forward dua blok pada tabel sama; tidak ada aggregator atau
lapisan yang diganti. Cache dibangun ulang setiap epoch dan setelah best checkpoint
dimuat. Cache tidak dipakai untuk training karena parameter dan BatchNorm berubah
di setiap optimizer step. Dengan tabel tetap dan BatchNorm evaluation, prediksi
query tidak bergantung pada ukuran/komposisi batch, selain pembulatan numerik.

## 8. Training, metrik dan latensi

Setiap root train muncul sekali dalam shuffle tiap epoch. Batch terakhir berisi
satu transaksi digabungkan ke batch sebelumnya agar BatchNorm dapat bekerja;
transaksi fraud tidak dibuang.

Bobot positif default `(N_train-N_fraud)/N_fraud`; weighted BCE memakai FP32 tanpa
AMP. Power dan cap bobot tersedia untuk ablation tetapi default tetap full ratio.
Checkpoint memakai maksimum AUPRC validation. Scheduler, early stopping dan
kalibrasi hanya membaca validation. Threshold utama 0,5; threshold tambahan
memaksimalkan F-beta validation, beta 1, dengan tie memilih threshold lebih tinggi.

Keluaran mencakup AP (`average_precision_score`, sesuai persamaan proposal),
precision, recall, F1, accuracy, ROC-AUC, TN/FP/FN/TP, dua kelompok threshold,
selisih validation-test, drift OOV/numerik/entitas baru, lima bagian waktu test,
long-tail recall, durasi tahap, serta peak VRAM.

### Long-tail pada graf bertipe

Karena tetangga langsung adalah entitas tanpa label, long-tail memakai **union
transaksi train dua-hop** yang berbagi user atau merchant dengan query. Transaksi
yang berbagi keduanya dihitung sekali. Fraud query termasuk long-tail jika
lingkungannya **lebih dari 70% normal**. Query tanpa history dilaporkan terpisah,
bukan dianggap lingkungan normal. Label test hanya menentukan bahwa query adalah
fraud; komposisi lingkungannya berasal dari training.

### Ruang lingkup timer

`inference_ms_per_1000` adalah durasi seluruh test dibagi jumlah transaksi dan
dinormalisasi ke 1.000 transaksi, dalam ms. Termasuk transfer fitur, forward query,
dan pengambilan probabilitas; embedding entitas sudah disiapkan.
`inference_with_preparation_ms_per_1000` juga memasukkan persiapan bobot, tabel
evaluation dan embedding entitas akhir.

Benchmark tambahan memilih 1.000 transaksi test tanpa melihat label, melakukan
warmup lalu lima pengulangan, dan melaporkan median. CUDA disinkronisasikan di
batas timer. Jangan membandingkan latensi query cache dengan EXP8 yang menyampling
dan menghitung konteks tiap batch tanpa menyatakan perbedaan ruang lingkup ini.

## 9. Instalasi dan penggunaan

Gunakan Python >=3.10 dan dependensi di `requirements.txt`. Untuk GPU, gunakan
build PyTorch CUDA yang sesuai perangkat. Environment EXP6 yang sudah ada dapat
dipakai; tidak perlu memasang dependency repository referensi.

Dari root repository:

```powershell
& .\code\exp6_gpu_tqdm\.venv\Scripts\python.exe -m pip install -r .\code\exp9_reference_sampling\requirements.txt
& .\code\exp6_gpu_tqdm\.venv\Scripts\python.exe .\code\exp9_reference_sampling\run.py --help
```

Atau buat environment sendiri:

```powershell
cd code/exp9_reference_sampling
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Perintah selanjutnya dijalankan dari folder EXP9, memakai `python` environment
yang aktif. Dataset default `../../dataset/credit_card_transactions-ibm_v2.csv`.

```powershell
# Pengujian otomatis.
python -m unittest discover -s tests -v

# Smoke test: 100.000 baris, dua epoch, tiga strategi, seed 42.
python run.py --config config.smoke.yaml --device cpu --no-progress
python run.py --config config.smoke.yaml --device cuda --no-progress

# Preprocessing full-data, tanpa training.
python run.py --config config.yaml --preprocess-only
python run.py --config config.yaml --preprocess-only --export-heterodata

# Satu strategi/seed pada seluruh data.
python run.py --config config.yaml --strategy uniform --seed 42
python run.py --config config.yaml --strategy topology --seed 42
python run.py --config config.yaml --strategy importance --seed 42

# Seluruh eksperimen: tiga strategi x lima seed = 15 run.
python run.py --config config.yaml
```

Hasil smoke disimpan di `outputs/smoke/` dan diabaikan Git. CPU ditujukan untuk uji
kecil. Ekspor HeteroData hanya berisi snapshot training dan membutuhkan memori
tambahan. Cache EXP9 terpisah dari EXP8 karena preprocessing, struktur dan payload
berbeda. Pakai `--rebuild-cache` untuk membangun ulang.

Fingerprint data mencakup path, ukuran, mtime CSV, batas subset, split dan source
preprocessing. Ini bukan checksum seluruh CSV; perubahan isi yang mempertahankan
ukuran/mtime memerlukan rebuild eksplisit.

Untuk subset tambahan dengan nama terpisah:

```powershell
python run.py --config config.yaml --max-rows 1000000 --epochs 20 --seed 42 --name exp9_subset_1m
```

Folder run yang sudah ada ditolak agar hasil tidak tertukar. Gunakan nama baru
atau `--overwrite` untuk secara eksplisit mengganti run sama. Resume optimizer
belum tersedia; `best.pt` merupakan checkpoint model terbaik untuk evaluasi.

### Ablation

```powershell
# Audit rumus literal, menghasilkan fallback uniform.
python run.py --config config.literal.yaml

# Kontribusi balanced-prior affinity pada topology historical.
python run.py --config config.unbalanced_topology.yaml

# Pengaruh batch 4.096 dengan pipeline EXP9 yang sama.
python run.py --config config.exp8_batch.yaml
```

Preset menjalankan ketiga strategi dengan aturan sama. Mengubah batch size adalah
perubahan training, bukan penghematan memori tanpa konsekuensi. Jangan mengubahnya
hanya untuk satu strategi dalam perbandingan ilmiah.

## 10. Artefak dan reproduksibilitas

```text
model/exp9/
  cache/graph_<data_hash>.pt
  <name>_<comparison_id>_<strategy>_seed<seed>/best.pt
result/exp9/
  preprocessing_<data_hash>.json
  <name>_<comparison_id>_<strategy>_seed<seed>/
    config.json
    status.json
    history.json
    metrics.json
  runs.csv
  summary.csv
```

Checkpoint menyimpan model/BatchNorm, konfigurasi, best epoch, seed/strategi,
encoder, nama fitur, threshold, seed tabel evaluation, fingerprint data dan hash
source. Cache graf memuat mapping entitas, fitur, timestamp dan source-row index.

`comparison_id` membedakan nama eksperimen, data, source kode, konfigurasi ilmiah
dan lingkungan eksekusi. Strategi/seed dalam eksperimen sama tetap dikelompokkan
bersama. Ringkasan menghitung count, mean dan **sample standard deviation**.
Satu seed tidak mempunyai std yang bermakna; kolom std dibiarkan kosong.
Subset, batch size atau source berbeda tidak dicampur dalam satu rata-rata.

Seed ditetapkan untuk training/tabel/evaluation. CUDA scatter/reduction tetap
dapat memberi selisih pembulatan; reproduksibilitas bitwise lintas GPU/versi tidak
dijanjikan. Versi runtime dan hash source disimpan untuk audit.

### Perbandingan dengan EXP8

```powershell
python compare.py ../../result/exp8_importance_seed42.json ../../result/exp9/<folder-run>/metrics.json
```

Ganti `<folder-run>` dengan folder aktual. Script membandingkan metrik pada
threshold **0,5**, dan menandai EXP8-versus-EXP9 sebagai perbandingan historis.
Script tidak memilih parameter atau run berdasarkan test.

Perbandingan sampling terkontrol harus memakai subset, encoder, history, model,
loss, fanout, batch, checkpoint policy dan evaluation policy yang sama. Gunakan
semua seed 42-46 dan laporkan mean/std serta selisih per seed terhadap uniform.
Perbedaan EXP8-versus-EXP9 tidak boleh seluruhnya diatribusikan pada bobot sampling
karena protokol graf dan komputasi juga berubah.

## 11. Pengujian dan batas kesimpulan

Unit/integration test mencakup parser waktu, encoder train-only/OOV, cache dan
invalidasinya, isolasi label holdout, PPR versus matriks dense, distribusi weighted
sampling versus probabilitas Plackett-Luce, hub melampaui anggaran chunk, entitas
kosong/kecil, literal fallback, batas pesan per lapisan, kesetaraan cache/blok,
invariansi batch query, checkpoint roundtrip, batch singleton, deduplikasi
long-tail, training tiga strategi, guard overwrite dan ringkasan CSV.

Hasil aktual ada di [VALIDATION.md](VALIDATION.md). Full-data 15 run belum
dijalankan pada tahap implementasi. Lulus tes membuktikan perilaku yang diperiksa,
bukan membuktikan peningkatan deteksi fraud pada seluruh populasi.

Untuk menilai keberhasilan, lihat AUPRC, recall/F1/precision pada threshold 0,5 dan
threshold validation secara terpisah, false positive, long-tail beserta jumlah
kasusnya, drift dan metrik per waktu. Untuk efisiensi, tampilkan biaya persiapan,
query setelah cache, training, dan VRAM pada perangkat sama. Jika hasil belum
meningkat, laporkan hasil negatif dan batasi penyesuaian pada validation. Test
yang dipakai berulang untuk memilih konfigurasi tidak lagi menjadi holdout yang
sepenuhnya independen.
