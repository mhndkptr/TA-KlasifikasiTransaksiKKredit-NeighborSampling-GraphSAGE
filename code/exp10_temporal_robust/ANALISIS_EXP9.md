# Analisis hasil EXP9 dan rancangan EXP10

Audit 8 September 2026 memakai tiga `metrics.json` EXP9, `summary.csv`,
laporan preprocessing, implementasi lokal ketujuh repository, dan pembacaan
agregat **seluruh 24.386.900 transaksi IBM**. Angka audit dapat direproduksi
dengan [audit_exp9.py](audit_exp9.py); hasilnya di [exp9_audit.json](validation/exp9_audit.json).

## 1. Hasil EXP9 tidak seburuk yang terlihat dari perbandingan metrik, tetapi gagal beradaptasi pada test

`auprc` adalah **average precision (AP)** dari `average_precision_score`, bukan
ROC-AUC dan bukan integral trapezoidal kurva PR. `f1` adalah **F1 kelas fraud**,
bukan F1-macro. Berikut perbandingan dengan definisi metrik yang sama.
F1-macro dan GMean dihitung kembali dari confusion matrix tersimpan, tanpa training ulang.

| Strategi, seed 42 | Val AP checkpoint | Test AP | Test ROC-AUC | F1 fraud @0,5 | F1-macro @0,5 | GMean @0,5 |
|---|---:|---:|---:|---:|---:|---:|
| Uniform | 0,649150 | 0,025288 | 0,642789 | 0,007009 | 0,489383 | 0,388063 |
| Topology | 0,653045 | 0,025862 | 0,657691 | 0,008048 | 0,492949 | 0,371822 |
| Importance | 0,652308 | 0,025448 | 0,658640 | 0,008360 | 0,492939 | 0,381679 |

ROC-AUC EXP9 sekitar 0,64–0,66, bukan 0,025. Secara numerik ini berada di
sekitar beberapa baseline finansial yang dikutip, tetapi **beda dataset berarti
bukan bukti menyamai/mengalahkan paper**. Prevalensi fraud test 0,00121759,
sehingga AP sekitar 20,77–21,24 kali baseline ranking acak. Kegunaan operasionalnya
tetap rendah: uniform menghasilkan **709 TP dan 197.159 FP** pada threshold 0,5.
F1-macro mendekati 0,5 juga bukan keberhasilan; kelas normal mendominasi dan
classifier selalu-normal pada data ini sudah memiliki F1-macro sekitar 0,4997.

Ketiga strategi baru **satu seed**, semuanya memilih epoch 43 dari 58 epoch.
Kolom standar deviasi kosong berarti belum dapat dihitung, bukan nol.
Keunggulan AP topology atas uniform hanya 0,0005734 atau sekitar 2,27% relatif;
belum ada dasar untuk klaim signifikansi atau strategi terbaik.

## 2. Bukti utama: hubungan kanal pembayaran dengan fraud berubah drastis

| Kanal di antara transaksi **fraud** | Train | Validation | Test |
|---|---:|---:|---:|
| Online | 14.771 / 20.886 = 70,72% | 3.450 / 4.417 = 78,11% | 128 / 4.454 = 2,87% |
| Chip | 256 / 20.886 = 1,23% | 673 / 4.417 = 15,24% | 3.907 / 4.454 = 87,72% |
| Swipe | 5.859 / 20.886 = 28,05% | 294 / 4.417 = 6,66% | 419 / 4.454 = 9,41% |

Pada **kelas normal**, proporsi online validation dan test justru hampir sama,
sekitar 12,38% dan 12,44%. Jadi rasio fraud keseluruhan yang serupa (~0,12%)
menutupi perubahan besar hubungan fitur–label. Kategori tidak perlu menjadi OOV
agar hubungan tersebut berubah. Laporan OOV EXP9 yang mendekati nol tidak
membuktikan tidak ada distribution shift.

Ini bukti kuat adanya perubahan pola fraud, konsisten dengan AP validation
0,65 yang jatuh menjadi 0,025. Mekanisme ketergantungan model pada setiap fitur
masih perlu ablation/attribution untuk dinyatakan sebagai sebab tunggal.
Tidak ada bukti bahwa CUDA atau arsitektur GraphSAGE saja menyebabkan gap ini.

Histori graf EXP9 berhenti pada **10 Desember 2015**. Validation mencakup akhir
2015–Januari 2018, sedangkan test Januari 2018–Februari 2020 tetap memakai
histori training yang sama. User yang tidak punya histori training bertambah
dari 0,80% validation menjadi 3,82% test. Kedua hal ini dapat menambah kesulitan
generalization; besarnya kontribusi masing-masing belum terisolasi.

Audit streaming memakai batas timestamp yang sama. Ada transaksi pada menit
batas split, sehingga beberapa transaksi **normal** berpindah kelompok dibanding
pembagian indeks tepat EXP9. Seluruh jumlah fraud per split pada tabel cocok
dengan artefak EXP9. Pada CSV ini tidak ditemukan label fraud pada baris tahun
2020; itu fakta isi file, bukan kesimpulan tentang fraud dunia nyata.

## 3. Threshold dan preprocessing memperburuk cara hasil dibaca

`trainer.py` EXP9 menghitung threshold validation, tetapi `metrics` utama,
`runs.csv`, dan `summary.csv` tetap menggunakan `evaluation.threshold=0.5`.
Kalibrasi tidak hilang; hasilnya ada di `test_metrics_at_validation_threshold`.

| Strategi | Threshold validation | F1 fraud test terkalibrasi | Recall test terkalibrasi |
|---|---:|---:|---:|
| Uniform | 0,999394 | 0,042275 | 0,021778 |
| Topology | 0,999168 | 0,042917 | 0,022003 |
| Importance | 0,999440 | 0,041921 | 0,021554 |

Kalibrasi meningkatkan precision/F1, tetapi hanya menemukan sekitar **2% fraud**.
Threshold-moving tidak dapat memperbaiki ranking AP/ROC-AUC. Menyalin threshold
0,2 atau 0,008 dari paper ke probabilitas EXP9 bukan perbaikan yang valid.

Weight positif EXP9 **816,33** berasal dari inverse-frequency BCE. Ini membuat
skor sigmoid bukan probabilitas fraud populasi yang terkalibrasi. Secara ideal,
weighted BCE mengestimasi `q = w*p / (1-p+w*p)`; threshold q=0,5 setara p=1/(w+1),
bukan p=0,5. Ini menjelaskan mengapa threshold tinggi mungkin diperlukan,
tetapi tidak membuktikan bahwa mengurangi weight pasti menaikkan AP.

Kelemahan representasi yang ditemukan di kode:

- `Use Chip`, `MCC`, dan `Errors?` diberi kode ordinal yang kemudian dinormalisasi.
  Jarak numerik antar-kategori tidak memiliki makna nominal yang sesuai.
- `Errors?` kosong diisi mode non-null. Pada full training mode-nya
  `Insufficient Balance`: tidak tercatat error menjadi sama dengan error tersebut.
- Amount min-max rentan terhadap rentang besar/outlier; lokasi hanya memiliki
  frequency encoding sehingga makna `ONLINE` tidak tersedia sebagai indikator eksplisit.

## 4. Mengapa tujuh repository bukan target angka yang langsung dapat direplikasi

| Referensi lokal | Protokol/komponen yang relevan | Perbedaan penting dengan EXP9 |
|---|---|---|
| CARE-GNN | `train.py`: random stratified 40/60; undersampling train; filter tetangga berbasis label | Yelp/Amazon, graf multi-relasi, bukan split waktu IBM 70/15/15 |
| PC-GNN | `src/model_handler.py`: stratified random split; Pick dan Choose | Root/graph balancing dan pemilihan tetangga tidak sama dengan topology EXP9 |
| DGFraud | `README.md`, `algorithms/GraphSage/`: toolbox berbagai model | Bukan satu algoritma atau satu angka benchmark IBM; baseline perlu dipilih spesifik |
| GraphSAGE | `README.md`, `graphsage/minibatch.py`: neighbor sampling; Reddit/PPI; split dari graf | Memberi algoritma dasar, bukan jaminan performa fraud pada IBM |
| GraphSAINT | `README.md`: sampling subgraph dengan koreksi normalisasi | Bukan sekadar mengganti distribusi pemilihan tetangga GraphSAGE |
| Inductive Graph Representation | `README.md`, `inductiveGRL/hinsage.py`: graf tripartit, rolling window, HinSAGE dan XGBoost | Data riil dengan fraud 0,65%; graph undersampling; classifier dan waktu berbeda |
| SplitGNN | `README.md`, `src/train.py`: spektral/heterophily; Yelp/Amazon/FDCompCN | Arsitektur serta dataset berbeda; bukan mean GraphSAGE IBM |

Sumber primer: [PC-GNN paper, tabel 6](https://yliu.site/pub/PCGNN_WWW2021.pdf),
[CARE-GNN](https://github.com/YingtongDou/CARE-GNN),
[DGFraud](https://github.com/safe-graph/DGFraud),
[GraphSAGE](https://github.com/williamleif/GraphSAGE),
[GraphSAINT](https://github.com/GraphSAINT/GraphSAINT),
[Inductive Graph Representation](https://github.com/Charlesvandamme/Inductive-Graph-Representation-Learning-for-Fraud-Detection).
Kode lokal adalah dasar pembacaan implementasi; versi paper/repository dapat berbeda.

Koreksi pada kutipan angka:

- PC-GNN menamai baris 0,4405 / 0,5439 / 0,2589 sebagai **GraphSAGE(T)**,
  yaitu threshold-moving. Baris GraphSAGE tanpa `(T)` berbeda pada F1-macro/GMean.
- [HOGRL tabel 2](https://www.ijcai.org/proceedings/2024/0839.pdf) memberi baris
  GraphSAGE CCFD berurutan **F1-macro 0,5104, AUC 0,5444, GMean 0,5011**.
  Urutan metrik CCFD pada kutipan pengguna tertukar. Angka Yelp/Amazon yang
  dikutip merupakan baseline GraphSAGE dalam paper itu, bukan model HOGRL.
- Angka HHLN-GNN, GTAN/RGTAN, SCN-GNN, dan DAGNN dalam kutipan belum diverifikasi
  seluruhnya; jangan menjadikannya target replikasi tanpa tabel serta protokol asli.

Pada graf U–T–M, setiap transaksi hanya mempunyai **dua tetangga langsung**.
Fanout 25/10 selalu mengambil keduanya. Strategi baru berpengaruh ketika entitas
memilih transaksi histori. Selain itu, degree centrality kandidat transaksi
konstan (=2) pada importance EXP9. Banyak variasi skor struktural menjadi kecil.
Ini menjelaskan mengapa mengganti sampler saja memiliki ruang pengaruh terbatas;
EXP9 tidak mengimplementasikan keseluruhan mekanisme CARE-GNN/PC-GNN/SplitGNN.

## 5. Hipotesis yang diuji EXP10

EXP10 mempertahankan graf, tiga strategi, dua mean-SAGE layer, hidden 256,
dropout 0,2, BatchNorm default, fanout 25/10, dan split utama 70/15/15.
Perubahan fitur, root sampler, regularisasi, dan evaluasi merupakan **ekstensi
metodologi**, bukan klaim sudah identik dengan proposal/ketujuh referensi.

1. One-hot nominal train-only, missing/OOV terpisah, signed-log amount dengan
   median/IQR, waktu sin/cos, serta indikator online/refund/error.
2. Root sampling: semua fraud train sekali per epoch + maksimal 20 normal per
   fraud tanpa replacement; negatif diacak ulang. Histori graf tetap seluruh
   train, semua strategi memakai root sampler sama. `pos_weight=1` mencegah
   penyeimbangan ganda. Untuk full data: 438.606 roots/epoch dan bobot positif
   efektif relatif populasi sekitar 40,82, dibanding 816,33 pada EXP9.
3. Checkpoint berdasarkan geometric mean AP/prevalensi dari tiga blok waktu
   validation. Periode yang lemah lebih diperhatikan daripada AP gabungan saja.
   Ini hipotesis robust selection, bukan metode yang diklaim berasal dari paper.
4. Threshold F1 dipilih pada 50% validation paling akhir; dipakai konsisten oleh
   metrics/CSV/checkpoint. Pembanding threshold 0,5 tetap tersedia di JSON.
5. Laporkan ROC-AUC, AP/lift, F1 fraud, F1-macro, GMean, FPR/alert rate,
   distribusi skor, periode waktu, dan hasil per kanal pembayaran.

Lihat [README](README.md) untuk ablation bertahap. `config.recent_refit.yaml`
menambahkan label training yang lebih baru dengan split **80/5/15** dan test
akhir yang sama. Itu **protokol label-budget berbeda**, bukan kenaikan akibat
sampling saja. Implementasi ini masih snapshot frozen history, belum rolling
retraining periodik atau sistem real-time dengan delayed labels.

Karena hasil test EXP9 sudah dipakai untuk diagnosis dan merancang EXP10,
peningkatan pada test yang sama harus diperlakukan sebagai hasil pengembangan.
Untuk klaim final generalisasi, gunakan evaluasi rolling yang ditetapkan lebih
dulu atau periode/dataset baru yang belum dipakai dalam desain. Tidak ada
threshold atau epoch EXP10 yang dipilih menggunakan label test di pipeline.

**Status:** perubahan kode dan pengujian teknis bukan bukti peningkatan full-data.
Hasil pengujian yang benar-benar dijalankan dicatat di [VALIDATION.md](VALIDATION.md).
