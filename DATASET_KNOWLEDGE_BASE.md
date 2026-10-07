# Knowledge Base Dataset Tugas Akhir

Dokumen ini adalah rujukan utama mengenai dataset yang digunakan pada tugas
akhir **klasifikasi transaksi penipuan kartu kredit dengan GraphSAGE dan
neighbor sampling**. Dokumen memisahkan fakta dari dataset mentah, hasil audit
lokal, dan keputusan pemodelan di repositori agar ketiganya tidak tercampur
dalam penulisan laporan.

Terakhir diverifikasi: **24 September 2026**.

## 1. Identitas dataset

| Atribut | Nilai |
|---|---|
| Nama | IBM Synthetic Credit Card Transactions Dataset |
| Pembuat/penerbit | IBM Research, melalui proyek TabFormer |
| Sifat data | Sintetis; bukan catatan transaksi nasabah nyata |
| Tugas pada penelitian ini | Klasifikasi biner transaksi normal dan fraud |
| File transaksi lokal | `dataset/credit_card_transactions-ibm_v2.csv` |
| Ukuran file lokal | 2.350.744.057 byte (sekitar 2,35 GB desimal) |
| SHA-256 file lokal | `B01FA323C98522F8C710C7F7242581860C97C50183F1C5FA9E772E5C674A7F15` |
| Jumlah baris | 24.386.900 transaksi |
| Jumlah kolom fisik CSV | 15 |
| Rentang waktu | 2 Januari 1991 07:10 sampai 28 Februari 2020 23:58 |
| Target | `Is Fraud?`: `Yes` = fraud, `No` = normal |

Repositori resmi TabFormer menyebut sekitar 24 juta record dan 12 field. File
lokal penelitian memiliki 15 kolom fisik karena tanggal disimpan sebagai tiga
kolom (`Year`, `Month`, dan `Day`) dan rincian file dapat dihitung dengan
konvensi berbeda. Untuk eksperimen dan laporan tugas akhir ini, skema aktual
CSV lokal yang berisi 15 kolom adalah sumber kebenaran operasional.

Folder lokal juga memuat `sd254_users.csv`, `sd254_cards.csv`, dan satu subset
`User0_credit_card_transactions.csv`. Ketiganya bukan input utama eksperimen
GraphSAGE saat ini: pipeline membaca `credit_card_transactions-ibm_v2.csv` dan
tidak menggabungkan atribut demografis user atau metadata kartu. Karena itu,
atribut dari file pelengkap tidak boleh disebut sebagai fitur model kecuali
pipeline kelak benar-benar diubah dan diaudit ulang.

Sumber primer:

- [Repositori resmi IBM TabFormer](https://github.com/IBM/TabFormer)
- [Paper TabFormer di arXiv](https://arxiv.org/abs/2011.01843)
- [Publikasi ICASSP 2021 di IEEE](https://ieeexplore.ieee.org/document/9414142)

## 2. Ringkasan hasil audit full-data

Angka berikut berasal dari pemindaian seluruh CSV lokal, bukan perkiraan dari
subset.

| Statistik | Nilai |
|---|---:|
| Transaksi total | 24.386.900 |
| Transaksi normal | 24.357.143 |
| Transaksi fraud | 29.757 |
| Proporsi fraud | 0,122020% |
| Rasio normal : fraud | sekitar 818,53 : 1 |
| User unik | 2.000 |
| Pasangan `User::Card` unik | 6.139 |
| Merchant unik | 100.343 |
| MCC unik | 109 |
| Kanal transaksi | 3 |

Ketimpangan kelas sangat ekstrem. Accuracy tidak layak dijadikan metrik utama
karena prediksi seluruh transaksi sebagai normal sudah menghasilkan accuracy
sekitar 99,878% tanpa menemukan satu pun fraud. Metrik utama yang lebih relevan
adalah Average Precision/AUPRC, recall fraud, precision fraud, F1 fraud, serta
TP, FP, FN, FPR, dan alert rate pada operating point yang jelas.

### Distribusi kanal

| `Use Chip` | Transaksi | Bagian data | Fraud | Fraud rate kanal |
|---|---:|---:|---:|---:|
| Swipe Transaction | 15.386.082 | 63,0916% | 6.572 | 0,042714% |
| Chip Transaction | 6.287.598 | 25,7827% | 4.836 | 0,076913% |
| Online Transaction | 2.713.220 | 11,1257% | 18.349 | 0,676281% |
| **Total** | **24.386.900** | **100%** | **29.757** | **0,122020%** |

Agregat full-data di atas tidak berarti distribusi tersebut stabil sepanjang
waktu. Audit eksperimen menemukan perubahan komposisi fraud yang besar:
fraud training lama didominasi transaksi online dan swipe, sedangkan test
2018–2020 didominasi fraud chip. Karena itu, kanal harus dilaporkan juga per
periode/split dan tidak hanya secara global.

## 3. Kamus data mentah

| Kolom | Bentuk/Contoh | Makna dan perlakuan di proyek |
|---|---|---|
| `User` | ID seperti `0` | Identitas user anonim. Dipakai sebagai kategori/kunci node, bukan besaran numerik. |
| `Card` | ID kartu per user seperti `0` | Identitas kartu milik user. Digabung dengan `User` bila diperlukan agar kunci kartu unik secara global. Pada jalur utama EXP14, kartu dipakai untuk fitur perilaku, bukan tipe node terpisah. |
| `Year` | `2002` | Tahun transaksi; bagian pembentuk timestamp. |
| `Month` | `9` | Bulan transaksi; bagian timestamp dan fitur musiman. |
| `Day` | `1` | Hari dalam bulan; bagian timestamp. |
| `Time` | `06:21` | Jam dan menit transaksi. Digabung dengan tanggal; parsing yang gagal harus dianggap error, bukan ditebak. |
| `Amount` | `$134.09` | Nilai transaksi dalam string berformat mata uang. Tanda `$` dan pemisah ribuan dibuang lalu dikonversi menjadi numerik. Nilai negatif dipertahankan sebagai indikasi refund. |
| `Use Chip` | `Swipe Transaction`, `Chip Transaction`, `Online Transaction` | Kanal/metode transaksi. Ini kategori nominal, bukan ordinal. |
| `Merchant Name` | ID numerik anonim, dapat negatif | Identitas merchant. Meskipun tampak numerik, nilainya adalah kategori/kunci node dan tidak boleh diperlakukan sebagai ukuran kontinu. |
| `Merchant City` | Nama kota atau `ONLINE` | Lokasi merchant. Nilai `ONLINE` juga dipakai untuk membentuk indikator transaksi online. |
| `Merchant State` | Kode wilayah atau kosong | Negara bagian/wilayah merchant. Kekosongan banyak berkaitan dengan jenis/lokasi transaksi dan tidak boleh langsung dianggap error acak. |
| `Zip` | Kode pos atau kosong | Lokasi pos merchant. Perlakukan sebagai kategori/string, bukan angka kontinu. |
| `MCC` | Kode empat digit | Merchant Category Code. Fitur kategori nominal; bukan nilai ordinal. |
| `Errors?` | Deskripsi error atau kosong | Error transaksi. Dalam pipeline, kosong dimaknai sebagai tidak ada error dan tetap dibuat kategori eksplisit. |
| `Is Fraud?` | `Yes` atau `No` | Label target. Nilai selain dua kategori tersebut harus menghentikan preprocessing agar label tak dikenal tidak diam-diam dianggap normal. |

Contoh struktur baris:

```csv
User,Card,Year,Month,Day,Time,Amount,Use Chip,Merchant Name,Merchant City,Merchant State,Zip,MCC,Errors?,Is Fraud?
0,0,2002,9,1,06:21,$134.09,Swipe Transaction,3527213246127876953,La Verne,CA,91750.0,5300,,No
```

## 4. Kelengkapan data

Audit nilai kosong pada 24.386.900 baris:

| Kolom | Kosong | Persentase kosong | Terisi |
|---|---:|---:|---:|
| `Merchant State` | 2.720.821 | 11,1569% | 21.666.079 |
| `Zip` | 2.878.135 | 11,8020% | 21.508.765 |
| `Errors?` | 23.998.469 | 98,4072% | 388.431 |
| 12 kolom lainnya | 0 | 0% | 24.386.900 per kolom |

Catatan interpretasi:

- Kekosongan `Errors?` merupakan kondisi dominan dan diperlakukan sebagai
  kategori “tanpa error”, bukan diimputasi dengan error yang paling sering.
- `Merchant State` dan `Zip` dapat tidak berlaku pada transaksi online atau
  lokasi tertentu. Pipeline membuat kategori missing secara eksplisit.
- Identitas user dan merchant tidak boleh kosong karena keduanya diperlukan
  untuk membangun graf.
- Audit missing tidak dengan sendirinya membuktikan kebenaran semantik setiap
  nilai. Validasi domain, kategori asing/OOV, dan rentang numerik tetap dicatat
  saat transformasi.

## 5. Representasi graf pada penelitian

Jalur eksperimen utama EXP1–EXP14 menggunakan graf heterogen konseptual:

```text
User <──> Transaction <──> Merchant
             │
             ├── fitur transaksi
             └── label normal/fraud
```

- Setiap baris CSV menjadi satu node `Transaction`.
- Setiap transaksi terhubung ke tepat satu node `User` dan satu node
  `Merchant`.
- Label hanya berada pada node transaksi.
- `User` dan `Merchant Name` menjadi identifier pembentuk graf, bukan fitur
  numerik langsung.
- `Card` digunakan dalam rekayasa fitur perilaku/kontekstual pada pipeline
  terbaru, tetapi bukan tipe node tersendiri pada jalur utama EXP14.
- Karena transaksi mempunyai dua endpoint, perbedaan strategi neighbor
  sampling terutama muncul ketika node entitas memilih transaksi historisnya.

EXP15 adalah replikasi/adaptasi referensi yang mendefinisikan client sebagai
kunci komposit `User::Card`. Hasilnya tidak boleh dianggap identik dengan graf
utama berbasis node `User`, kecuali perbedaan definisi entitas tersebut sudah
dikontrol dan dijelaskan.

## 6. Preprocessing yang disepakati

Urutan minimum preprocessing:

1. Baca 15 kolom dan tambahkan ID baris sumber untuk audit.
2. Gabungkan `Year`, `Month`, `Day`, dan `Time` menjadi timestamp.
3. Validasi timestamp dan label; hentikan proses jika ada nilai tidak valid.
4. Urutkan timestamp secara stabil dengan ID baris sumber sebagai tie-breaker.
5. Bersihkan `Amount` menjadi numerik tanpa menghilangkan nilai negatif.
6. Bentuk split secara temporal, bukan random.
7. Fit statistik, kamus kategori, imputasi, dan scaling hanya pada data train.
8. Terapkan kategori `<missing>` dan `<oov>` secara eksplisit.
9. Bangun fitur perilaku hanya dari observasi sebelum waktu transaksi
   (`[awal_histori, waktu_query)`), sehingga transaksi pada timestamp yang sama
   tidak saling membaca.
10. Bangun graf/histori hanya sampai cutoff yang diizinkan oleh protokol.

Evolusi dimensi fitur dalam repositori:

| Versi | Dimensi | Ringkasan |
|---|---:|---|
| EXP1–EXP9 | 9 fitur transaksi dasar | Amount, waktu, kanal, MCC, error, frekuensi lokasi |
| EXP10 | 155 | Encoding nominal, waktu siklik, amount robust, lokasi/OOV |
| EXP11 | 179 | Tambahan fitur perilaku user/kartu yang strictly-past |
| EXP12/EXP14 | 220 | Tambahan konteks recent dan conditional |
| EXP13 | 15 | Versi compact khusus deployment; tidak setara dengan 220 fitur |

Pernyataan “dataset memiliki 15 kolom” tidak sama dengan “model menerima 15
fitur”. Jalur utama terbaru menghasilkan **220 fitur turunan dari 15 kolom
mentah**, dan hal itu harus ditulis eksplisit dalam metodologi.

## 7. Pembagian temporal dan risiko leakage

Protokol historis utama memakai rasio 70% train, 15% validation, dan 15% test
setelah pengurutan kronologis:

| Split historis | Transaksi | Fraud | Periode kira-kira |
|---|---:|---:|---|
| Train | 17.070.830 | 20.886 | 1991-01-02 – 2015-12-10 |
| Validation | 3.658.035 | 4.417 | 2015-12-10 – 2018-01-27 |
| Test | 3.658.035 | 4.454 | 2018-01-27 – 2020-02-28 |

EXP14 memperbaiki batas menjadi **timestamp tie-safe**, sehingga semua
transaksi dengan timestamp sama tetap berada pada satu sisi split. Akibatnya,
jumlah baris dapat berbeda beberapa transaksi dari tabel historis. Manifest
EXP14 mencatat `train_end = 17.070.833` dan `test_start = 20.728.867`. Saat
melaporkan hasil, gunakan angka dari manifest run yang benar-benar dieksekusi,
bukan mencampur angka dari versi split berbeda.

Aturan anti-leakage:

- Jangan memakai random split untuk klaim performa temporal.
- Jangan fit encoder atau normalisasi pada validation/test.
- Jangan memakai label validation/test untuk bobot topology/homophily.
- Jangan memakai transaksi masa depan dalam fitur histori atau graf train.
- Pilih checkpoint pada selection/validation, kalibrasi threshold pada subset
  calibration terpisah bila protokol menyediakannya, lalu baca assessment/test
  terakhir.
- Jangan memilih threshold, fitur, atau hyperparameter berdasarkan hasil test.

## 8. Karakteristik temporal penting

- Fraud berlabel pertama pada audit harian muncul 5 Juli 1996 dan yang terakhir
  muncul 27 Oktober 2019, walaupun keseluruhan transaksi mencakup 1991–2020.
- Dari 10.599 tanggal yang mempunyai transaksi, 3.241 tanggal mempunyai
  sedikitnya satu fraud dan 7.358 tanggal tidak mempunyai fraud.
- Audit EXP14 menemukan periode Januari–Oktober 2017 tanpa fraud berlabel.
- Komposisi kanal fraud berubah kuat antarperiode. Oleh sebab itu, performa
  global dapat menyembunyikan kegagalan pada kanal atau rezim waktu tertentu.
- Transaksi fraud yang berdekatan pada user/kartu yang sama dapat berasal dari
  satu episode. Jumlah transaksi tidak selalu sama dengan jumlah kejadian
  fraud independen.

Konsekuensinya, evaluasi sebaiknya melaporkan metrik per waktu dan per kanal,
dukungan jumlah fraud tiap subset, serta jumlah user/kartu/episode fraud bila
tersedia. Subgroup tanpa kasus positif dilaporkan sebagai `NA`, bukan nol.

## 9. Keterbatasan dan batas klaim

1. Dataset bersifat sintetis, sehingga performa pada data ini tidak otomatis
   mewakili sistem pembayaran nyata atau institusi tertentu.
2. Data berlangsung hampir 30 tahun dan mengalami concept/label drift yang
   besar. Asumsi distribusi identik antarperiode tidak berlaku.
3. Label sangat langka dan tidak tersebar merata dalam waktu; satu split atau
   satu seed tidak cukup untuk klaim yang kuat.
4. ID user, kartu, merchant, ZIP, dan MCC bersifat kategorikal walaupun tampak
   numerik. Memperlakukan ID sebagai bilangan kontinu menambahkan hubungan
   ordinal palsu.
5. `max_rows` mengambil prefix file, bukan sampel acak yang representatif.
   Prefix terutama berguna untuk smoke test dan tidak boleh dilaporkan sebagai
   hasil penelitian utama.
6. Test 2018–2020 telah dipakai dalam proses pengembangan eksperimen lama.
   Hasil pada test tersebut bersifat eksploratif; klaim final idealnya memakai
   periode atau dataset baru yang belum digunakan untuk keputusan desain.
7. Perbandingan strategi sampling hanya adil jika data, split, fitur, graf,
   root sampling, model, fanout, seed, loss, checkpoint, dan threshold sama.

## 10. Artefak lokal yang menjadi sumber angka

| Artefak | Fungsi |
|---|---|
| `result/exp15/dataset_audit.json` | Total baris, fraud, entitas, rentang waktu, dan missing value |
| `result/exp15/daily_support.csv` | Jumlah transaksi dan fraud per tanggal |
| `result/exp14/fold_manifest.json` | Batas split tie-safe dan dukungan rolling fold |
| `result/exp14/temporal_support.json` | Dukungan bulanan, kanal, user, kartu, dan episode fraud |
| `code/exp15_reference_replication/exp15/data.py` | Kontrak 15 kolom dan normalisasi audit/replikasi |
| `code/exp12_recent_context/exp12/data.py` | Parsing, validasi, dan encoder fitur robust |
| `code/exp12_recent_context/exp12/behavior_v2.py` | Definisi fitur strictly-past recent/conditional |
| `code/exp14_local_temporal/exp14/feature_store.py` | Pipeline out-of-core 220 fitur dan metadata graf |

Urutan prioritas ketika angka berbeda adalah: artefak run yang sedang
dilaporkan, kode dan konfigurasi run tersebut, audit full-data, lalu ringkasan
naratif. Jangan menggabungkan statistik dari protokol eksperimen berbeda tanpa
memberi label yang jelas.

## 11. Deskripsi siap pakai untuk laporan

> Penelitian ini menggunakan IBM Synthetic Credit Card Transactions Dataset
> yang dipublikasikan bersama proyek TabFormer oleh IBM Research. Salinan lokal
> yang diaudit terdiri atas 24.386.900 transaksi dengan 15 kolom dan rentang
> waktu 2 Januari 1991 hingga 28 Februari 2020. Sebanyak 29.757 transaksi
> berlabel fraud, atau sekitar 0,122% dari seluruh data, sehingga rasio kelas
> normal terhadap fraud mencapai sekitar 819:1. Setiap transaksi dibentuk
> sebagai node yang terhubung dengan node user dan merchant. Pembagian data
> dilakukan secara kronologis untuk meniru prediksi terhadap transaksi masa
> depan dan mengurangi kebocoran temporal. Statistik preprocessing dipelajari
> hanya dari data training.

Sesuaikan jumlah split, dimensi fitur, dan definisi node dengan eksperimen yang
benar-benar dilaporkan. Jangan menggunakan paragraf ini sebagai pengganti
sitasi sumber dataset dan penjelasan metodologi.

## 12. Sitasi yang disarankan

```bibtex
@inproceedings{padhi2021tabular,
  title     = {Tabular Transformers for Modeling Multivariate Time Series},
  author    = {Padhi, Inkit and Schiff, Yair and Melnyk, Igor and
               Rigotti, Mattia and Mroueh, Youssef and Dognin, Pierre and
               Ross, Jerret and Nair, Ravi and Altman, Erik},
  booktitle = {2021 IEEE International Conference on Acoustics, Speech and
               Signal Processing (ICASSP)},
  pages     = {3565--3569},
  year      = {2021},
  doi       = {10.1109/ICASSP39728.2021.9414142}
}
```

Untuk reproducibility, laporan juga sebaiknya menyebut nama file lokal, jumlah
baris, tanggal audit, dan SHA-256 yang tercantum pada dokumen ini.
