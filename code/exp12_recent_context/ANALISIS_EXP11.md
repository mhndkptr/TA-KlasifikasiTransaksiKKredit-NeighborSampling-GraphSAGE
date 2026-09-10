# Evaluasi EXP11 dan rancangan EXP12

Analisis ini memakai artefak full-data seed 42 yang tersedia dan audit streaming
CSV 24.386.900 transaksi. Angka test telah dipakai untuk pengembangan beberapa
eksperimen, sehingga temuan ini bersifat diagnostik, bukan estimasi final pada
data yang belum pernah dilihat.

## Hasil yang benar-benar membaik

| Strategi | AP test EXP10 | AP test EXP11 | F1 EXP11 | Recall EXP11 |
|---|---:|---:|---:|---:|
| Uniform | 0,035268 | 0,174721 | 0,220554 | 23,42% |
| Topology | 0,028505 | 0,187791 | 0,247068 | 20,81% |
| Importance | 0,029190 | 0,153851 | 0,215062 | 25,55% |

EXP11 menghasilkan peningkatan AP sekitar 4,96×–6,59× dan recall sekitar
3,2×–4,2× terhadap EXP10. Fitur perilaku memang memberi sinyal yang sebelumnya
hilang. Topology memiliki AP/F1 tertinggi pada seed ini, sedangkan importance
memiliki recall tertinggi. Satu seed belum cukup untuk menyatakan pemenang.

Gap tetap besar: AP validation checkpoint EXP11 berada pada 0,805–0,832,
sedangkan AP test hanya 0,154–0,188. Pada test, AP fraud online tetap tinggi
(0,801–0,874), sementara AP fraud chip hanya 0,142–0,162. Masalah utama masih
generalisasi temporal ke fraud chip.

## Masalah pada preprocessing dan fitur EXP11

Fitur lifetime EXP11 menghitung mean/variance nominal dari seluruh histori user
atau kartu. Pada rentang 1991–2020, transaksi lama dapat mendominasi baseline
dan menyamarkan perubahan perilaku terbaru. EXP11 juga belum memiliki:

1. baseline nominal lokal, misalnya tujuh hari terakhir;
2. kebiasaan jam per user/kartu dan burst 24 jam relatif terhadap tujuh hari;
3. nominal bersyarat kartu–kanal dan kartu–MCC, yang relevan untuk membedakan
   fraud chip dari pola kartu itu sendiri;
4. nominal bersyarat user–merchant dan merchant–kanal;
5. novelty merchant/MCC/kanal/lokasi pada level kartu.

EXP12 menambahkan semuanya secara strictly earlier dan tanpa label. Transaksi
dengan timestamp sama tidak saling membaca. Statistik robust fitur dasar tetap
fit hanya pada train. Fitur streaming validation/test boleh membaca transaksi
sebelumnya tanpa label, sama seperti kebijakan deployment online EXP11.

Logika kartu EXP11 juga bersifat all-or-nothing: bila ada satu `Card` kosong,
seluruh fitur card memakai histori user. EXP12 mempertahankan kartu yang valid
dan mengelompokkan missing-card per user.

## Masalah pada checkpoint validation

EXP11 dirancang memilih checkpoint dari `val[50%:75%]`, tetapi window diperluas
mundur sampai memperoleh minimum 25 fraud. Audit menunjukkan window akhirnya
menjadi 18 Desember 2016–17 Juli 2017, dengan komposisi:

| Kanal | Fraud selection EXP11 |
|---|---:|
| Online | 20 |
| Swipe | 3 |
| Chip | 2 |

Sementara itu, 25% validation paling akhir berisi 379 fraud dan mayoritas chip,
namun hanya dipakai untuk threshold. Checkpoint karena itu dipilih dengan sinyal
ranking yang hampir tidak mengandung subtype target terbaru.

EXP12 mengambil 25% validation terbaru, lalu membagi blok waktu tujuh hari
secara bergantian. Pembagian ditentukan oleh parity blok sejak Unix epoch,
sehingga label tidak menentukan keanggotaan. Audit full CSV menghasilkan:

| Partisi EXP12 | Transaksi | Fraud | Fraud chip |
|---|---:|---:|---:|
| Checkpoint selection | 451.858 | 210 | 196 |
| Threshold calibration | 462.653 | 169 | 153 |

Kedua partisi tidak overlap. Desain ini menggunakan pola terbaru untuk kedua
keputusan sambil menjaga label calibration tidak memilih checkpoint.

## Evaluasi logic GraphSAGE

Implementasi factorized EXP11 telah diuji setara dengan ekspansi block untuk
output, gradient, BatchNorm/LayerNorm, dan dropout. Bentuk layer sesuai mean
GraphSAGE: transformasi fitur root ditambah transformasi mean tetangga. Jadi
tidak ditemukan bug aritmetika utama pada forward pass.

Keterbatasan ada pada informasi graf. Setiap transaksi hanya mempunyai dua
tetangga langsung, user dan merchant, sehingga fanout 25 pada langkah pertama
selalu mengambil keduanya. Variasi sampler baru terjadi ketika entity memilih
transaksi historis. Graf entity–transaction tetap membeku pada akhir train;
fitur perilaku held-out bergerak, tetapi embedding context entity tidak.
`config.self_only.yaml` disediakan untuk mengukur apakah graf masih menambah AP
di atas fitur transaksi/context yang sudah kuat.

EXP12 belum mengubah arsitektur menjadi HinSAGE, rolling graph, atau XGBoost,
karena perubahan itu akan mencampur kontribusi fitur, validation, dan model.

## Mengapa angka repository referensi tidak langsung sebanding

Repository GraphSAGE asli memberi algoritma umum dan benchmark seperti Reddit/
PPI, bukan target fraud IBM. CARE-GNN, PC-GNN, dan SplitGNN memakai Yelp/Amazon
atau graf multi-relasi dengan random stratified split. Repository
Inductive-Graph-Representation-Learning-for-Fraud-Detection paling dekat karena
memakai graf tripartit, tetapi menggunakan HinSAGE, rolling timeframe,
graph-level undersampling, data riil berprevalensi fraud 0,65%, dan XGBoost
setelah embedding.

Karena dataset, prevalensi, split, classifier, dan akses histori berbeda, selisih
angka tidak membuktikan implementasi GraphSAGE EXP11 salah. Perbandingan yang
valid memerlukan protokol serta data identik. EXP12 tetap memakai split temporal
IBM 70/15/15 agar perbaikan dapat dibandingkan dengan EXP10/11.

## Batas klaim EXP12

Fitur baru menambah kebutuhan RAM, dan graph context masih frozen-train. Audit
menunjukkan window yang lebih relevan, tetapi peningkatan full-data belum dapat
diklaim sebelum run selesai. Jalankan kontrol, feature-only, logic-only, main,
dan self-only; kemudian ulangi tiga strategi pada beberapa seed. Jangan memilih
preset dari AP test seed 42 saja.

