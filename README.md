# Klasifikasi Transaksi Penipuan Kartu Kredit dengan GraphSAGE

Implementasi terbaru tersedia di **[EXP11 - Fitur Perilaku dan Validation Terbaru](code/exp11_behavioral_temporal/README.md)**.
EXP11 menambahkan histori transaksi user/kartu tanpa label, memisahkan pemilihan
checkpoint dari kalibrasi threshold, dan memperjelas alasan early stopping.
[Analisis full-data EXP10](code/exp11_behavioral_temporal/ANALISIS_EXP10.md) dan
[validasi EXP11](code/exp11_behavioral_temporal/VALIDATION.md) mencatat bukti serta batas klaim.

Versi sebelumnya: **[EXP10 - Temporal Robustness](code/exp10_temporal_robust/README.md)**.
EXP10 memperbaiki fitur nominal, penanganan imbalance pada root training, pemilihan
checkpoint temporal, dan pelaporan metrik. [Analisis EXP9](code/exp10_temporal_robust/ANALISIS_EXP9.md)
menjelaskan gap validation–test serta perbedaan protokol dengan tujuh referensi.
EXP9 dan hasil aslinya tetap tersedia. Peningkatan full-data EXP10 perlu diuji;
perubahan metodologi dan ablation didokumentasikan terpisah dari proposal awal.

Repository ini berisi implementasi tugas akhir tentang **klasifikasi transaksi
penipuan kartu kredit berskala besar menggunakan GraphSAGE dengan optimasi
neighbor sampling**. Penelitian tidak hanya menilai apakah GraphSAGE mampu
mengenali transaksi fraud, tetapi secara khusus membandingkan bagaimana cara
memilih tetangga memengaruhi kualitas deteksi pada graf yang sangat besar,
tidak seimbang, dan berpotensi heterofilik.

Dataset transaksi yang digunakan adalah **IBM Synthetic Credit Card
Transactions Dataset**. Dataset lengkap memuat sekitar 24,3 juta transaksi,
15 kolom, dan proporsi fraud sekitar 0,12%. Karakteristik tersebut membuat
akurasi saja tidak cukup untuk mengevaluasi model: model yang selalu menebak
transaksi normal pun dapat terlihat sangat akurat tanpa benar-benar mendeteksi
fraud.

## Permasalahan penelitian

Model machine learning tabular umumnya memproses setiap transaksi secara
terpisah. Pendekatan itu dapat memanfaatkan nominal, waktu, lokasi, atau metode
pembayaran, tetapi tidak secara langsung mempelajari hubungan antara pengguna,
transaksi, dan merchant.

Penelitian ini mengubah data tersebut menjadi graf agar model dapat mempelajari
fitur transaksi sekaligus konteks relasionalnya. Tantangan utamanya adalah:

- **Extreme class imbalance**: transaksi fraud jauh lebih sedikit daripada
  transaksi normal.
- **Representation dilution**: sinyal fraud yang langka dapat tertutup oleh
  mayoritas tetangga normal ketika informasi graf diagregasi.
- **Heterophily dan kamuflase**: transaksi fraud dapat berada di lingkungan
  yang didominasi entitas atau transaksi normal.
- **Skalabilitas**: seluruh tetangga dari jutaan node tidak praktis diproses
  sekaligus, sehingga diperlukan neighbor sampling.

Pertanyaan utama penelitian adalah apakah pemilihan tetangga yang
mempertimbangkan struktur graf atau kepentingan node dapat meningkatkan
Recall, F1-Score, dan AUPRC dibandingkan pengambilan sampel acak, tanpa
menimbulkan biaya inferensi yang tidak proporsional.

## Representasi graf

Setiap baris pada dataset direpresentasikan sebagai node transaksi. Graf
memiliki tiga tipe node:

- **User**: pengguna pemilik kartu;
- **Transaction**: satu kejadian transaksi;
- **Merchant**: merchant tempat transaksi berlangsung.

Setiap node transaksi terhubung ke satu user dan satu merchant:

```text
User ── Transaction ── Merchant
          │
          ├── fitur transaksi
          └── label normal/fraud
```

Label klasifikasi hanya melekat pada node transaksi. Fitur awal node user dan
merchant berupa degree embedding, sedangkan node transaksi menggunakan fitur
hasil preprocessing. Graf heterogen disimpan dalam format `HeteroData` untuk
audit. Pada pelatihan, graf dikonversi menjadi representasi homogen agar semua
strategi sampling dibandingkan menggunakan arsitektur GraphSAGE yang identik.

## Preprocessing dan rekayasa fitur

Pipeline melakukan preprocessing berikut:

1. Mengurutkan transaksi secara kronologis.
2. Mempertahankan `User` dan `Merchant Name` sebagai kunci pembentuk graf,
   bukan sebagai fitur numerik langsung.
3. Membersihkan dan menormalisasi `Amount`.
4. Mengekstrak fitur jam, hari dalam minggu, bulan, dan indikator akhir pekan.
5. Mengodekan `Use Chip`, `MCC`, dan `Errors?` sebagai fitur kategorikal.
6. Menggabungkan kota, negara bagian, dan kode pos merchant, lalu menerapkan
   frequency encoding berdasarkan data training.
7. Memisahkan data secara temporal menjadi **70% training, 15% validation,
   dan 15% test**.

Statistik preprocessing dan encoding hanya dipelajari dari bagian training
untuk mengurangi risiko data leakage. Proposal juga menyebut pembagian
80/10/10 pada salah satu algoritma, tetapi implementasi mengikuti pembagian
70/15/15 pada tabel metodologi yang lebih rinci.

## Model GraphSAGE

Semua eksperimen menggunakan kapasitas model yang sama agar perubahan hasil
dapat dikaitkan dengan strategi sampling, bukan perbedaan arsitektur:

- dua lapisan GraphSAGE dengan mean aggregator;
- hidden dimension 256;
- aktivasi ReLU dan batch normalization;
- dropout 0,2;
- classifier linear untuk menghasilkan logit fraud;
- fan-out sampling 25 tetangga pada lapisan pertama dan 10 pada lapisan kedua.

Model dilatih dengan weighted binary cross-entropy untuk memberi perhatian
lebih besar pada kelas fraud, optimizer Adam dengan learning rate 0,001,
scheduler berbasis AUPRC validation, dan early stopping. Setiap strategi
dirancang untuk dijalankan dengan lima random seed: 42, 43, 44, 45, dan 46.

## Perkembangan eksperimen

Setiap folder eksperimen mempertahankan tujuan penelitian yang sama, tetapi
memperbaiki aspek implementasi atau evaluasi secara bertahap:

| Eksperimen | Fokus perubahan |
|---|---|
| EXP1 | Baseline GraphSAGE dan weighted neighbor sampling berbasis CPU |
| EXP2 | Pemindahan training dan sampling ke GPU |
| EXP3 | Early stopping dan evaluasi sampling yang lebih stabil |
| EXP4 | Penggabungan GPU-bound training dengan evaluasi deterministik |
| EXP5 | Exact GPU sampler dengan semantik sampling dan multigraph EXP1 |
| EXP6 | Kalibrasi threshold validation, audit generalization gap, dan monitoring TQDM |
| EXP7 | Sampling importance dengan batas memori |
| EXP8 | Adaptive batched PPR dan cache bobot importance |
| EXP9 | Struktur modular, frozen training history, blok per lapisan, PPR closed-form, dan audit proposal |
| EXP10 | Audit perubahan pola fraud, fitur nominal, balanced roots, checkpoint temporal, dan metrik referensi |

EXP6 merupakan tahap pengembangan sebelum EXP7, EXP8, dan EXP9. Exact
GPU sampler tetap berasal dari EXP5 agar perubahan hasil dapat dikaitkan dengan
kalibrasi/evaluasi, bukan pergantian algoritma sampling.

## Strategi neighbor sampling

Penelitian membandingkan tiga pendekatan berikut.

### 1. Uniform Random Sampling

Baseline GraphSAGE. Semua tetangga memiliki peluang yang sama untuk dipilih
tanpa mempertimbangkan struktur atau kepentingannya.

### 2. Topology-Aware Sampling

Tetangga diberi bobot berdasarkan kombinasi kemiripan lingkungan dan local
homophily. Label yang digunakan untuk komponen homophily dibatasi pada label
training; label validation dan test tidak digunakan untuk menentukan bobot.

Pada graf bipartit bertipe, endpoint langsung mempunyai tipe berbeda sehingga
Jaccard satu-hop dapat selalu bernilai nol. Implementasi menggunakan reciprocal
degree sebagai fallback kekuatan struktural ketika kondisi tersebut terjadi.
Keputusan ini dicatat secara eksplisit agar hasil eksperimen dapat diaudit.

### 3. Importance-Based Sampling

Tetangga diprioritaskan menggunakan gabungan degree centrality dan pendekatan
Personalized PageRank lokal dengan propagasi terbatas. Tujuannya adalah memilih
node yang diperkirakan memberi informasi lebih besar tanpa menghitung PageRank
global untuk keseluruhan graf pada setiap batch.

## Evaluasi

Metrik utama yang dicatat adalah:

- **Recall**, untuk menilai berapa banyak transaksi fraud yang berhasil
  ditemukan;
- **Precision**, untuk mengukur proporsi prediksi fraud yang benar;
- **F1-Score**, untuk menyeimbangkan precision dan recall;
- **AUPRC**, yang lebih informatif daripada accuracy pada data dengan kelas
  sangat tidak seimbang;
- **confusion matrix** (`TN`, `FP`, `FN`, dan `TP`);
- **inference time per 1.000 node**;
- **long-tail fraud recall**, untuk subset fraud yang lingkungan berlabelnya
  lebih dari 70% normal.

Checkpoint dipilih berdasarkan AUPRC validation terbaik. Hasil dari beberapa
seed harus dianalisis menggunakan rata-rata dan standar deviasi sebelum
menarik kesimpulan final. Berkas yang saat ini berada di `model/` dan `result/`
dapat berupa hasil percobaan atau smoke test dan bukan otomatis hasil final
penelitian.

Pada EXP6, threshold keputusan dipilih dengan memaksimalkan F1 pada validation
set saja. Test set tidak digunakan untuk memilih threshold. Output juga
menyimpan metrik test pada threshold tetap 0,5 sebagai pembanding, statistik
probabilitas validation/test, serta selisih AUPRC validation terhadap test.

### Temuan full-data EXP5 seed 42

Run uniform sampling pada seluruh data menghasilkan:

- 17.070.830 node transaksi train dengan 20.886 fraud;
- 3.658.035 node validation dengan 4.417 fraud;
- 3.658.035 node test dengan 4.454 fraud;
- best validation AUPRC 0,6679 pada epoch 30;
- test AUPRC 0,0237;
- 179.533 false positive dan recall 0,0855 pada threshold tetap 0,5.

Perbedaan besar antara validation dan test menunjukkan temporal generalization
gap. Karena validation ranking sudah kuat, EXP6 mempertahankan full imbalance
class weight EXP5 dan memusatkan perubahan pada kalibrasi threshold serta audit
gap tersebut. Hasil ini masih berasal dari satu strategi dan satu seed sehingga
belum cukup untuk menjadi kesimpulan akhir penelitian.

## Struktur repository

```text
repository/
├── code/
│   ├── exp1_starter/
│   │   ├── config.yaml       # konfigurasi eksperimen
│   │   ├── requirements.txt  # dependensi Python
│   │   ├── run.py            # preprocessing, graph, training, dan evaluasi
│   │   └── README.md         # petunjuk khusus Experiment 1
│   ├── exp2_gpu_bound/        # varian training dan sampling berbasis GPU
│   │   ├── config.yaml
│   │   ├── requirements.txt
│   │   ├── run.py
│   │   └── README.md
│   ├── exp3_stable_training/  # early stopping dan evaluasi deterministik
│   │   ├── config.yaml
│   │   ├── requirements.txt
│   │   ├── run.py
│   │   └── README.md
│   ├── exp4_gpu_stable/       # GPU-bound dengan evaluasi deterministik
│   │   ├── config.yaml
│   │   ├── requirements.txt
│   │   ├── run.py
│   │   └── README.md
│   ├── exp5_gpu_corrected/    # GPU-bound dengan semantik sampling EXP1
│   │   ├── config.yaml
│   │   ├── requirements.txt
│   │   ├── run.py
│   │   ├── test_sampler.py
│   │   └── README.md
│   └── exp6_gpu_tqdm/         # full-data, threshold validation, dan TQDM
│       ├── config.yaml
│       ├── requirements.txt
│       ├── run.py
│       ├── test_exp6.py
│       └── README.md
├── dataset/                  # dataset lokal, diabaikan oleh Git
│   └── .gitkeep
├── model/                    # cache graf dan checkpoint terbaik
├── result/                   # metrik per run dan ringkasan CSV
├── .gitignore
└── README.md
```

Dataset tidak dimasukkan ke Git karena ukurannya besar. File transaksi utama
yang diharapkan pipeline adalah:

```text
dataset/credit_card_transactions-ibm_v2.csv
```

## Instalasi

Direkomendasikan menggunakan Python 3.10 atau lebih baru dan PyTorch dengan
dukungan CUDA. Dari root repository, instal dependensi EXP6:

```powershell
cd code/exp6_gpu_tqdm
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Pastikan file dataset utama sudah tersedia di folder `dataset/` sebelum
menjalankan eksperimen.

## Menjalankan eksperimen

Perintah berikut dijalankan dari folder `code/exp6_gpu_tqdm` setelah virtual
environment diaktifkan.

### Validasi subset

Gunakan 100.000 transaksi untuk memvalidasi pipeline. Subset yang terlalu kecil
dapat tidak memiliki fraud pada salah satu temporal split dan akan dihentikan:

```powershell
python run.py --config config.yaml --strategy uniform --seed 42 --max-rows 100000
```

### Menjalankan satu strategi pada full data

`config.yaml` EXP6 menggunakan `max_rows: null`, sehingga perintah berikut
memakai seluruh dataset:

```powershell
python run.py --config config.yaml --strategy uniform --seed 42
python run.py --config config.yaml --strategy topology --seed 42
python run.py --config config.yaml --strategy importance --seed 42
```

### Menjalankan konfigurasi lengkap

```powershell
python run.py --config config.yaml
```

Perintah terakhir menjalankan seluruh strategi dan seed yang didefinisikan di
`config.yaml`, yaitu 3 strategi x 5 seed atau 15 run. Eksperimen penuh
membutuhkan RAM, VRAM, ruang penyimpanan, dan waktu komputasi yang besar.

Progress bar TQDM menampilkan progres run, epoch, training batch,
validation/test batch, loss, learning rate, dan penggunaan VRAM. Logger memakai
handler yang kompatibel dengan TQDM agar pesan tidak merusak progress bar.
Progress bar dapat dinonaktifkan ketika output diarahkan ke file:

```powershell
python run.py --config config.yaml --strategy uniform --seed 42 --no-progress
```

## Artefak keluaran

Pipeline menghasilkan:

```text
model/exp6_<strategy>_seed<seed>.pt
result/exp6_<strategy>_seed<seed>.json
result/exp6_summary.csv
```

Checkpoint menyimpan model terbaik, konfigurasi, seed, strategi, dan threshold
keputusan. File JSON menyimpan metrik utama, confusion matrix, statistik split,
riwayat training, threshold validation, metrik pembanding threshold 0,5,
generalization gap, latensi inferensi, serta peak VRAM. CSV ringkasan digunakan
untuk membandingkan strategi dan seed secara berdampingan.

## Reproduksibilitas dan catatan eksperimen

- Gunakan subset temporal, konfigurasi model, dan threshold yang sama ketika
  membandingkan strategi.
- Jika threshold otomatis digunakan, pilih threshold hanya dari validation set
  dan jangan mengoptimalkannya menggunakan test set.
- Jangan menggunakan label validation atau test untuk menghitung bobot
  sampler maupun fitur preprocessing.
- Catat versi Python, PyTorch, PyTorch Geometric, jenis GPU/CPU, RAM, dan waktu
  eksperimen ketika menghasilkan laporan final.
- Jangan menyimpulkan performa dari satu seed atau smoke test.
- Periksa jumlah fraud pada setiap split. Pada subset kecil yang hanya mengambil
  bagian awal kronologi, test set mungkin tidak mengandung fraud sehingga
  Recall atau AUPRC tidak representatif.

## Ruang lingkup

Fokus repository saat ini adalah eksperimen komparatif neighbor sampling pada
GraphSAGE. Penelitian tidak berfokus pada perubahan fungsi agregasi, graph data
augmentation, atau perbandingan banyak arsitektur GNN. Pembatasan ini menjaga
agar pengaruh mekanisme pemilihan tetangga dapat diamati secara lebih adil.
