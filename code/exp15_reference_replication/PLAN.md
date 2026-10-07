**Rencana EXP15 — Replikasi Inductive Graph Representation Learning pada dataset IBM**

Tanggal: 18 September 2026. Status: implementasi awal tersedia; training full-data EXP15 belum dijalankan.

Tujuan EXP15 adalah menjalankan kembali pipeline referensi **Inductive-Graph-Representation-Learning-for-Fraud-Detection** menggunakan `dataset/credit_card_transactions-ibm_v2.csv`. Pengguna telah memilih cakupan lengkap: **HinSAGE + XGBoost dan FI-GRL + XGBoost**, termasuk baseline fitur transaksi. Susunan kode, konfigurasi, progress bar, log, cache, dan hasil mengikuti pola EXP12/EXP14.

Arti replikasi dalam rencana ini adalah mempertahankan algoritma, urutan tahap, parameter eksplisit, dan protokol notebook eksperimen asli. Penggantian dataset memerlukan adapter fitur dan kalender. Kesamaan preprocessing asli, versi library historis, dan hasil numerik tidak dapat dijanjikan: repository tidak menyertakan `preprocessed_ccf.csv`, pembuat file tersebut, atau dependency lock lengkap. Semua keputusan pengganti harus tercatat sebelum training.

**1. Acuan implementasi dan temuan audit**

Sumber utama adalah checkout lokal pada commit `b60d1fa93dce25b536aaf90c7860076f8d9377b1`, yang bersih saat diperiksa:

| Sumber | Peran dalam EXP15 |
| --- | --- |
| [Experimental Pipeline.ipynb](../../../Ref/Inductive-Graph-Representation-Learning-for-Fraud-Detection/Experimental%20Pipeline.ipynb) | Urutan eksperimen, window, undersampling, hyperparameter, XGBoost, dan evaluasi |
| [hinsage.py](../../../Ref/Inductive-Graph-Representation-Learning-for-Fraud-Detection/inductiveGRL/hinsage.py) | Training supervised, pembagian internal 80/20, ekstraksi embedding, inferensi |
| [graphconstruction.py](../../../Ref/Inductive-Graph-Representation-Learning-for-Fraud-Detection/inductiveGRL/graphconstruction.py) | Graf tripartit tidak berarah |
| [timeframes.py](../../../Ref/Inductive-Graph-Representation-Learning-for-Fraud-Detection/inductiveGRL/timeframes.py) | Rolling window dan batas holdout berbasis tanggal |
| [FIGRL.m](../../../Ref/Inductive-Graph-Representation-Learning-for-Fraud-Detection/Demo/FIGRL.m) | Implementasi FI-GRL MATLAB yang dipanggil notebook eksperimen |
| [evaluation.py](../../../Ref/Inductive-Graph-Representation-Learning-for-Fraud-Detection/inductiveGRL/evaluation.py) | Average Precision, kurva precision–recall, Lift@1% |
| [README EXP12](../exp12_recent_context/README.md), [README EXP14](../exp14_local_temporal/README.md) | Pola struktur proyek dan pengalaman menjalankan eksperimen |

Nomor cell notebook yang disebut selanjutnya dihitung mulai dari nol. Notebook demo GraphSAGE memakai split 60/40 dan batch 5; angka tersebut **bukan** acuan eksperimen utama. Notebook eksperimen memakai rolling window dan batch 50.

Audit dataset dan environment yang sudah dilakukan:

- CSV utama tersedia, ukuran **2.350.744.057 byte**, dengan 15 kolom. File `sd254_cards.csv`, `sd254_users.csv`, dan `User0_credit_card_transactions.csv` juga tersedia. Input utama tetap CSV penuh; tidak menambahkan atribut profil atau menggantinya dengan subset User0.
- Cache full-data EXP14 mencatat **24.386.900 transaksi**, **29.757 fraud**, waktu **2 Januari 1991–28 Februari 2020**. Angka jumlah fraud dan rentang waktu diperiksa dari `labels.npy` dan `timestamps.npy`; ukuran dan waktu modifikasi CSV cocok dengan metadata cache. Ini bukan audit ulang seluruh isi CSV atau verifikasi hash penuh.
- Python pada environment EXP6 yang dipakai EXP12/EXP14 adalah **3.13.7**. TensorFlow, StellarGraph, XGBoost, dan modul MATLAB tidak ditemukan pada environment tersebut. `matlab` tidak ditemukan di PATH; keberadaan instalasi/lisensi di lokasi lain belum diperiksa.
- Rilis PyPI StellarGraph 1.2.1 mensyaratkan Python `>=3.6,<3.9`. Karena itu lingkungan referensi perlu dipisahkan dari environment proyek saat ini. [Metadata resmi StellarGraph](https://pypi.org/project/stellargraph/).

**2. Kontrak metode yang dipertahankan**

| Komponen | Keputusan EXP15 berdasarkan source referensi |
| --- | --- |
| Pipeline | Data numerik → pemilihan timeframe → undersampling train → graf → embedding → XGBoost → evaluasi holdout |
| Fold utama | Lima timeframe berurutan; notebook mengoperasikan satu timeframe per eksekusi, contoh `timeframe=4` |
| Kalender | Window 17 hari, bergeser 5 hari, holdout 5 hari terakhir; pada window lengkap berarti 12 hari train + 5 hari holdout |
| Graf | Tiga tipe node: client, merchant, transaction; edge client–transaction dan merchant–transaction; tidak berarah |
| Fitur node | Transaction memakai fitur transaksi; client dan merchant masing-masing atribut konstan `[1]` |
| HinSAGE | Dua layer, `layer_sizes=[64,64]`, embedding transaksi 64 dimensi, `num_samples=[2,32]`, dropout 0 |
| Supervisi HinSAGE | Dense sigmoid satu unit, binary cross-entropy biasa, Adam learning rate 0,001 |
| Training HinSAGE | Batch 50, tepat 10 epoch; pembagian root internal 80/20 menurut urutan input |
| Pemilihan model | Bobot akhir epoch 10; referensi tidak memakai early stopping, scheduler, atau seleksi checkpoint berdasarkan AP |
| FI-GRL | MATLAB; intermediate dimension 400, final dimension 64 |
| Inferensi embedding | Bangun graf gabungan sampled train + seluruh transaksi holdout; gunakan model/faktor hasil train |
| XGBoost | `XGBClassifier(n_estimators=100)`; parameter default lainnya harus dibekukan lewat versi dan manifest |
| Fitur classifier | Embedding saja atau embedding + fitur transaksi; baseline hanya fitur transaksi |
| Evaluasi utama | Average Precision (AP), precision–recall curve, Lift@1% |

HinSAGE menggunakan agregasi heterogen berdasarkan tipe, sehingga model homogen EXP12/EXP14 tidak dapat dipakai sebagai pengganti langsung. Default `MeanHinAggregator`, aktivasi `relu, linear`, dan normalisasi L2 perlu dikunci sesuai versi StellarGraph yang dipilih. Semantik sampling `[2,32]` mengikuti generator bertipe, termasuk pengulangan tetangga dan urutan tensor; jangan menafsirkannya sebagai fanout homogen 2/32. [Source resmi HinSAGE 1.2.1](https://stellargraph.readthedocs.io/en/v1.2.1/_modules/stellargraph/layer/hinsage.html).

EXP15 menggunakan pola operasional EXP12/EXP14. Fitur perilaku 65 kolom, hidden 256, dropout 0,2, weighted BCE, root n30 per epoch, threshold F2/FPR, serta tiga strategi neighbor sampling dari eksperimen tersebut bukan bagian preset replikasi. Graph-level undersampling referensi mengubah isi graf training itu sendiri.

**3. Adapter dataset IBM**

| Data IBM | Representasi EXP15 | Aturan |
| --- | --- | --- |
| Nomor baris CSV | `transaction_id` | ID stabil, unik, tidak menjadi fitur transaksi |
| `User` + `Card` | `CARD_PAN_ID` / client | Kunci komposit kartu per user, sebagai padanan identitas kartu pada notebook; `Card` sendiri tidak unik global |
| `Merchant Name` | `TERM_MIDUID` / merchant | Baca sebagai string/integer utuh; jangan melalui float yang menghilangkan presisi ID |
| `Year`, `Month`, `Day`, `Time` | `TX_DATETIME` | Timestamp lokal dataset; tidak mengarang zona waktu sumber |
| `Is Fraud?` | `TX_FRAUD` | `Yes=1`, `No=0`; label tidak masuk fitur atau atribut graf |
| `Amount` | Fitur nominal | Parse simbol mata uang, pertahankan nilai negatif, dokumentasikan nilai kosong |
| `Use Chip`, `MCC`, `Errors?` | Fitur kategorikal | Encoder eksplisit dan konsisten train/holdout |
| `Merchant City`, `Merchant State`, `Zip` | Fitur lokasi | Pertahankan sebagai informasi kategorikal; tidak mengarang padanan country/acceptance yang tidak tersedia |

Pemakaian kunci kartu adalah keputusan adapter karena source memakai `CARD_PAN_ID`. Perbedaan dari node User EXP12/EXP14 harus terlihat di README dan manifest, dengan `client_key: user_card`. Nilai Card kosong memakai sentinel per user. Kunci node ketiga tipe harus memiliki namespace berbeda, lalu dipetakan menjadi ID integer padat; ID mentah yang sama lintas tipe tidak boleh menyatukan node.

Rancangan encoder IBM awal: nominal numerik distandardisasi memakai statistik train; komponen kalender numerik sederhana; kategori di-one-hot dengan kategori unknown/missing; jumlah dimensi dicatat. Fit dilakukan pada train timeframe sebelum undersampling agar seluruh varian rasio memakai ruang fitur yang sama. Encoder tidak membaca distribusi holdout. Ini merupakan **spesifikasi adapter IBM**, bukan klaim bahwa preprocessing privat penulis menggunakan transformasi yang sama. Hindari fitur agregat perilaku tambahan agar perubahan input tetap mudah dilacak.

Pisahkan dua matriks input:

- `transaction_features`: tidak memuat label, timestamp mentah, ID transaksi, `CARD_PAN_ID`, atau `TERM_MIDUID`; dipakai HinSAGE.
- `classifier_features`: mengikuti cell 43 yang hanya membuang label dan timestamp, sehingga **ID client dan merchant turut masuk** baseline dan varian `+features`. Gunakan kode numerik kompak yang konsisten, mapping train dan kode unknown untuk entitas baru; simpan daftar fitur. Dampak ID arbitrer dilaporkan, tidak dihilangkan diam-diam.

Raw CSV harus diurutkan stabil berdasarkan `(timestamp, source_row_id)` sebelum pemilihan window. Pembacaan beberapa baris pertama membuktikan urutan file tidak dapat dianggap sebagai urutan global waktu; `source_row_id` tetap dipertahankan setelah sort maupun sampling. Tidak memakai cache fitur EXP12/EXP14 sebagai input model EXP15.

**4. Kalender IBM dan pemisahan evaluasi**

Untuk origin `s`, timeframe ke-`i` memakai interval `[s + 5(i-1), s + 5(i-1) + 17)` hari. Batas train/holdout berada pada hari ke-12 untuk window lengkap. Lima fold mencakup bentang 37 hari, dengan holdout yang tidak overlap dan train yang boleh overlap. Model, encoder, dan classifier dilatih ulang setiap fold. Lima fold waktu berbeda dari lima random seed.

Ada adaptasi kalender yang wajib terlihat. Jika lima window pertama dimulai dari timestamp minimum dataset IBM, dukungannya sebagai berikut, berdasarkan cache full-data yang cocok dengan CSV lokal:

| Fold | Awal window | Transaksi train | Fraud train | Transaksi holdout | Fraud holdout |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | 1991-01-02 | 16 | 0 | 8 | 0 |
| 2 | 1991-01-07 | 13 | 0 | 14 | 0 |
| 3 | 1991-01-12 | 23 | 0 | 10 | 0 |
| 4 | 1991-01-17 | 30 | 0 | 12 | 0 |
| 5 | 1991-01-22 | 28 | 0 | 7 | 0 |

Transaksi fraud pertama pada cache bertanggal 5 Juli 1996. Memakai lima window paling awal secara literal tidak menghasilkan benchmark fraud yang bisa dievaluasi.

Rencana penetapan kalender:

1. Audit seluruh CSV dengan agregasi harian, tanpa melatih model. Simpan jumlah transaksi/fraud, jumlah client/merchant, missing values, dan rentang data.
2. Catat lima window literal di atas sebagai `ineligible`, beserta alasannya. Tetap pertahankan durasi 17/5/5 hari untuk benchmark IBM.
3. Tentukan anchor benchmark dengan aturan deklaratif: scan origin pada grid 5 hari dari tanggal minimum, pilih origin pertama yang train 12 harinya memiliki sedikitnya 25 fraud dan 25 normal serta window 17 hari lengkap. Angka 25 adalah guard dukungan IBM, bukan hyperparameter referensi. Pilih berdasarkan train, bukan skor model atau banyaknya fraud holdout.
4. Kunci origin terpilih dan empat origin berikutnya dalam `fold_manifest.json` sebelum menjalankan model. Fold berikutnya yang tidak layak tetap dicatat; tidak mengganti fold berdasarkan performa atau mencari holdout dengan fraud lebih banyak. Jika tidak tercapai lima fold layak, laporkan cakupan aktual; benchmark periode lain harus diberi identitas berbeda.
5. Untuk holdout tanpa fraud, simpan prediksi dan jumlah false alert tetapi AP/Lift fraud dilaporkan tidak terdefinisi. Jangan menyulap fold itu menjadi nilai sempurna atau menghapusnya dari laporan kelayakan.

Audit full CSV pada 18 September 2026 mengunci anchor **13 Juni 2001**. Fold 1 dan fold 5 memenuhi minimum 25 fraud train; fold 2–4 tetap terkunci tetapi berturut-turut hanya mempunyai 19, 11, dan 9 fraud train. Secara default CLI menjalankan fold eligible; fold rendah dukungan dapat dijalankan hanya dengan `--fold` eksplisit dan harus tetap dilaporkan sebagai rendah dukungan. Aturan kalender tersebut adalah adaptasi yang diperlukan ketika periode asal diganti dataset berdurasi hampir tiga dekade; seluruh hasilnya bersifat benchmark eksploratif.

`Timeframes.train_inductive_split` menghitung cutoff dari akhir hari transaksi maksimum dalam window. Adapter wajib memeriksa kelengkapan window dan merekam batas aktual; window parsial di ujung dataset tidak boleh diam-diam dianggap memiliki 12/5 hari penuh. Timestamp yang sama tidak dipecah di batas outer train/holdout.

Pembagian **internal** HinSAGE mengikuti source: ambil `round(0.8 * n)` ID pertama untuk gradient training dan sisanya untuk validation loss, sesudah graph-level undersampling. Validation internal tetap berada di graf train, dan embedding seluruh sampled train dipakai melatih XGBoost. Dengan demikian, subset internal tersebut bukan holdout untuk XGBoost dan tidak boleh dilaporkan sebagai hasil generalisasi akhir.

Perlu audit khusus urutan keluaran `RandomUnderSampler`: dapat mengelompokkan kelas, sehingga split posisi 80/20 tidak otomatis temporal atau stratified dan bisa memiliki satu kelas. Simpan ID serta dukungan kelas kedua bagian. Untuk parity, pertahankan perilaku source; bila nantinya dibuat perbaikan urutan/split, gunakan preset berbeda dan nyatakan perubahan metodologinya.

**5. Graph-level undersampling dan matriks eksperimen**

Notebook cell 14 menggunakan `RandomUnderSampler(sampling_strategy=ratio)` pada train sebelum graf dibangun. Rasio berarti **jumlah fraud / jumlah normal setelah sampling**, bukan persentase normal yang diambil. Misalnya `0.1` berarti sekitar 1 fraud : 10 normal. Transaksi normal yang dibuang tidak tetap tersedia sebagai tetangga graf. Holdout menggunakan prevalensi dan transaksi asli.

Gunakan sampled transaction IDs yang identik untuk HinSAGE, FI-GRL, dan baseline pada pasangan fold/rasio/seed yang sama. Pertahankan indeks sumber setelah sampling. Rasio yang tidak feasible, train satu kelas, dan graf terlalu kecil untuk SVD 64 dimensi menghasilkan status eksplisit, bukan rasio atau dimensi pengganti otomatis.

| Varian | Input XGBoost | Tujuan |
| --- | --- | --- |
| `features_only` | Fitur transaksi classifier | Baseline referensi |
| `hinsage_embedding` | Embedding HinSAGE 64 dimensi | Mengukur informasi embedding |
| `hinsage_plus_features` | Embedding HinSAGE + fitur classifier | Pipeline utama HinSAGE |
| `figrl_embedding` | Embedding FI-GRL 64 dimensi | Mengukur informasi embedding |
| `figrl_plus_features` | Embedding FI-GRL + fitur classifier | Pipeline utama FI-GRL |

Kelima varian dijalankan pada fold, baris training, encoder, dan holdout yang sama. Embedding dapat digunakan ulang untuk dua varian classifier dengan identitas embedding yang sama; masing-masing classifier mempunyai instance dan model tersimpan sendiri. Baseline cukup dilatih sekali per fold/rasio/seed.

Konfigurasi yang terbukti eksplisit di notebook adalah `undersampling_rate=None`, `embedding_size=64`, dan `add_additional_data=True`. Maka urutan eksekusi dimulai dari tiga varian `features_only`, `hinsage_plus_features`, dan `figrl_plus_features` tanpa undersampling, lalu dua varian embedding-only.

Repository memperlihatkan mekanisme undersampling tetapi **tidak mencantumkan seluruh grid rasio paper**. Tahap audit referensi harus mencari grid dari paper/artefak penulis dan menyimpan sumbernya sebelum sweep diklaim sama dengan paper. Jangan mengisi grid tebakan lalu menyebutnya parameter asli. Bila grid historis tetap tidak tersedia, notebook dengan rasio eksplisit yang terverifikasi tetap dapat direplikasi; sweep IBM tambahan harus diberi label pilihan EXP15.

Seed awal EXP15 adalah 42; seed 43–46 dapat menjadi replikasi tambahan setelah baseline lulus. Seed eksplisit merupakan tambahan untuk keterulangan karena notebook tidak menetapkannya. Simpan seed Python, NumPy, TensorFlow, sampler, XGBoost, dan MATLAB secara terpisah dengan derivasi deterministik. Lima seed tidak menggantikan lima fold.

**6. Dua cabang representasi dan inferensi**

HinSAGE dilatih dengan library StellarGraph dalam environment referensi. Simpan classifier sigmoid untuk audit training dan encoder embedding secara terpisah. Prediksi akhir eksperimen berasal dari **XGBoost**, bukan output sigmoid head pelatihan HinSAGE. Model akhir menggunakan bobot epoch 10; checkpoint per epoch boleh disimpan untuk pemulihan tanpa mengubah aturan pemilihan model.

FI-GRL utama menggunakan `Demo/FIGRL.m` melalui MATLAB Engine, sesuai notebook eksperimen. Simpan faktor SVD, urutan node, seed/state RNG, konfigurasi, serta model XGBoost. Matriks adjacency tetap sparse; ID MATLAB perlu integer positif padat dan mapping yang menjaga ID train saat menambah node holdout. Jangan menggunakan nilai Merchant Name sebagai nomor indeks matriks.

Ada beberapa detail source FI-GRL yang perlu dipertahankan dan diuji secara eksplisit:

- Training memakai normalisasi adjacency berbasis degree, Gaussian sketch berdimensi 400, lalu SVD 64 dimensi.
- Langkah inductive membentuk **Gaussian sketch baru**, menggunakan faktor hasil train, lalu menerapkan faktor degree tambahan pada embedding akhir. Jangan otomatis menggantinya dengan sketch train yang diperluas.
- `Demo/FIGRL.py` merupakan implementasi demo lain dan menjelaskan modifikasi untuk kecepatan; bukan pengganti MATLAB yang diasumsikan identik. Port Python hanya boleh diberi status alternatif setelah uji parity formula, indeks, dan hasil pada input terkendali. Jika dipakai tanpa MATLAB, jangan menyebutnya replikasi runtime asli.
- Konversi 0-based/1-based, node tanpa edge, singular value nol, dan rank graf harus diuji sebelum data nyata. Kasus tidak layak ditandai, bukan diperbaiki dengan perubahan rumus tersembunyi.

Untuk kedua cabang, graf inferensi berisi **sampled train + seluruh holdout lima hari**. Bobot encoder dan XGBoost dibekukan; label holdout tidak diserahkan kepada pembentukan fitur, graf, atau prediksi. Karena seluruh struktur dan fitur holdout hadir bersamaan, protokolnya adalah **batch inductive** dan dapat memanfaatkan konteks transaksi lain di dalam holdout, termasuk yang waktunya lebih akhir. Ini mengikuti notebook; jangan mengklaim inferensi online strictly-past. Eksperimen online kausal, bila dibutuhkan kemudian, merupakan protokol lain.

**7. Environment dan penggunaan sumber daya**

Siapkan environment EXP15 tersendiri. Kandidat awal adalah Python 3.8 dengan StellarGraph 1.2.1, kemudian tentukan pasangan TensorFlow/Keras, NumPy, pandas, NetworkX, scikit-learn, imbalanced-learn, dan XGBoost yang lolos smoke. Versi tersebut adalah pilihan rekonstruksi environment, bukan versi historis yang sudah dibuktikan. Seluruh versi final dikunci setelah pengujian dan dicatat bersama default XGBoost yang benar-benar digunakan.

Adapter kompatibilitas dapat menangani import Keras, constructor StellarGraph, atau nama argumen API yang berubah. Catat setiap patch beserta bukti bahwa graph, tensor, bobot, dan urutan komputasinya tetap setara. `setup.py` referensi tidak cukup menjadi lock dan mengimpor dependensi tambahan yang perlu diaudit, misalnya `dateparser` dan `imbalanced-learn`.

MATLAB Engine harus cocok dengan versi MATLAB dan Python yang tersedia. Jika tidak bisa berbagi environment HinSAGE, jalankan worker MATLAB/Python terpisah dengan pertukaran edge list, mapping ID, seed, dan faktor melalui format yang terdokumentasi. `doctor` memeriksa jalur executable, import, kemampuan memulai engine, dan fungsi FI-GRL. Tanpa keberhasilan pemeriksaan ini, cakupan replikasi lengkap tetap belum selesai.

CSV penuh dipindai bertahap untuk audit dan partisi waktu. Bangun graf hanya untuk window yang sedang dijalankan, kemudian lepaskan memorinya sebelum fold berikutnya. Backend preprocessing boleh terpisah dari environment model jika format perantara dan tipe data diverifikasi. Hindari NetworkX berisi seluruh 24 juta transaksi ketika eksperimen sebenarnya bekerja per window.

Estimasi memori berdasarkan jumlah node window: satu matriks dense float64 berukuran `N × 400` memerlukan `N × 400 × 8` byte; pada 100.000 node sekitar 305 MiB. FI-GRL memerlukan lebih dari satu matriks seperti ini dan workspace SVD. Audit RAM/disk perlu menghitung puncak gabungan, bukan hanya ukuran embedding 64 dimensi. Jangan menurunkan embedding, mengubah epoch, atau mengganti precision secara otomatis untuk mengatasi OOM pada preset referensi.

**8. Struktur yang direncanakan**

Struktur berikut menjadi kontrak implementasi. Modul inti, konfigurasi, dokumentasi, worker MATLAB, dan test sudah tersedia; cache serta folder run dibuat saat perintah dijalankan.

```text
code/exp15_reference_replication/
|-- PLAN.md
|-- README.md
|-- VALIDATION.md
|-- REFERENCE_PARITY.md
|-- run.py
|-- config.reference.yaml
|-- config.smoke.yaml
|-- config.multiseed.yaml
|-- environment.reference.yml
|-- requirements.txt
|-- requirements.lock.txt
|-- exp15/
|   |-- __init__.py
|   |-- cli.py
|   |-- config.py
|   |-- data.py
|   |-- features.py
|   |-- timeframes.py
|   |-- sampling.py
|   |-- graph.py
|   |-- hinsage.py
|   |-- figrl.py
|   |-- classifier.py
|   |-- trainer.py
|   |-- evaluation.py
|   |-- artifacts.py
|   |-- runtime.py
|   `-- summary.py
|-- matlab/
|   `-- FIGRL.m
|-- tests/
|   |-- test_data_timeframes.py
|   |-- test_graph_sampling.py
|   |-- test_reference_parity.py
|   `-- test_pipeline.py
`-- validation/

model/exp15/
|-- cache/<data_and_preprocess_fingerprint>/
`-- <run_id>/
    |-- encoder/
    |-- hinsage/
    |-- figrl/
    `-- xgboost/

result/exp15/
|-- run.log
|-- dataset_audit.json
|-- reference_manifest.json
|-- fold_manifest.json
|-- summary.json
|-- runs.csv
|-- summary.csv
`-- <run_id>/
    |-- config.json
    |-- environment.json
    |-- source_manifest.json
    |-- status.json
    |-- history.json
    |-- graph_stats.json
    |-- metrics.json
    |-- predictions.parquet
    |-- timing.json
    `-- plots/
```

`run_id` mengikat fold, rasio, seed, varian classifier, backend, dan fingerprint konfigurasi. Cache juga mengikat dataset, schema fitur, fold, sampled IDs, source, dan versi dependensi yang relevan. Resume/skip run lengkap memerlukan identitas sama. Semua penulisan JSON/checkpoint menggunakan pola atomik dan retry Windows seperti EXP12/EXP14.

Modul logging/artefak yang relevan boleh diadaptasi dari EXP12/EXP14 dengan asal source tercatat. EXP15 tidak mengimpor trainer/model lama melalui shim `legacy.py`, agar preset replikasi tidak mewarisi perubahan metode dari eksperimen sebelumnya. Salinan kode referensi menyertakan asal commit dan atribusi; sebelum distribusi, periksa file lisensi karena metadata `setup.py` saja belum menggantikan teks lisensi.

CLI menyediakan `doctor`, `audit-reference`, `audit-data`, `preprocess`, `run`, dan `summarize`. Opsi utama: `--config`, `--fold`, `--variant`, `--seed`, `--undersampling-rate`, `--results-dir`, dan `--no-progress`.

Progress bar/log menampilkan pembacaan CSV, preprocessing, konstruksi graf, epoch/batch HinSAGE, ekstraksi embedding, tahap MATLAB/SVD, training XGBoost, serta evaluasi. Untuk operasi MATLAB yang tidak menyediakan progres granular, tampilkan mulai/selesai, elapsed time, dan penggunaan memori yang dapat diukur; jangan menampilkan ETA buatan.

**9. Evaluasi dan artefak ilmiah**

Metrik utama mengikuti source: **AP**, kurva precision–recall, dan **Lift@1%**. AP dihitung dengan `average_precision_score`; label `auprc` untuk kompatibilitas laporan harus menjelaskan bahwa nilainya AP, bukan integrasi trapesium.

Lift@1% mengikuti `k = round(0.01 × n_holdout)` dan `precision(top-k) / prevalence(holdout)`. Audit perilaku tie skor dan urutan sort menurut environment referensi, lalu simpan kebijakan yang dipakai. Tambahkan penanganan kasus top-k tanpa fraud/normal, `k=0`, dan holdout satu kelas agar tidak crash akibat indexing `value_counts`; perubahan penanganan edge case dicatat.

Precision, recall, F1, confusion matrix, ROC-AUC, dan metrik per kanal boleh menjadi keluaran tambahan untuk keterbacaan seperti EXP12/EXP14. Gunakan threshold tetap 0,5 untuk metrik klasifikasi tambahan dan simpan nilainya. Threshold tidak dituning menggunakan holdout dan tidak menggantikan AP/Lift sebagai hasil utama.

Simpan prediksi per transaksi beserta fold, ID, label evaluasi, skor, dan kanal. Simpan sampled IDs, ID gradient/validation internal, feature names, peta node, graph counts, dan prevalensi sebelum/sesudah sampling. Artefak ini memungkinkan metrik serta alignment embedding–label dihitung ulang tanpa retraining.

Ringkasan membandingkan semua varian pada fold/rasio/seed yang sama, melaporkan nilai setiap fold, mean/std, jumlah fold layak, dan run gagal. Jika ada prediksi gabungan antar-fold, beri label metrik pooled karena berbeda dari rerata per-fold. Seed tambahan diringkas terpisah dari variasi waktu. Training windows yang overlap membatasi klaim independensi antar-fold.

Waktu dipecah menjadi preprocessing, graph construction, training representasi, embedding train, embedding holdout, training XGBoost, dan prediksi XGBoost. Waktu pipeline end-to-end serta startup MATLAB dicatat terpisah. Jangan hanya membandingkan forward XGBoost dengan inferensi GraphSAGE lama yang sudah mencakup sampling/fitur.

Hasil EXP12/EXP14 dapat ditampilkan sebagai konteks historis. Perubahan kalender, unit client, fitur, model, dan classifier membuat AP exp15 tidak menjadi perbandingan terkontrol langsung dengan angka test exp12. Tidak ada syarat bahwa replikasi harus lebih baik agar dinyatakan berhasil.

**10. Tahap implementasi dan kriteria selesai**

| Tahap | Pekerjaan | Kriteria selesai |
| --- | --- | --- |
| P0 — Kunci acuan | Manifest commit/hash, ekstrak parameter notebook, audit default library dan grid undersampling | `REFERENCE_PARITY.md` membedakan parameter terbukti, adaptasi IBM, patch kompatibilitas, dan hal yang belum diketahui |
| P1 — Environment | Environment terpisah, lock dependensi, probe TensorFlow/StellarGraph/XGBoost/MATLAB | `doctor` dan contoh kecil kedua cabang berhasil; versi dan perangkat tercatat |
| P2 — Data dan kalender | Adapter 15 kolom, audit full CSV, aturan anchor, lima fold, encoder | `dataset_audit.json`, `fold_manifest.json`, schema fitur dan ID konsisten; tidak ada fit preprocessing pada holdout |
| P3 — Graf dan sampling | Graf tripartit, constant features, undersampling sebelum graf, perluasan graf holdout | Node/edge serta sampled IDs sesuai acuan; kedua learner melihat struktur yang sama |
| P4 — Learner | HinSAGE dengan parameter referensi; FI-GRL MATLAB dengan mapping dan RNG tersimpan | Uji parity input, komputasi, ekstraksi embedding, dan pemulihan model lulus |
| P5 — Classifier/evaluasi | Lima varian, XGBoost 100 trees, AP/PR/Lift, prediksi dan ringkasan | Metrik dapat dihitung ulang dari artefak dan semua varian memakai fold yang sama |
| P6 — Smoke/pilot | Fixture sintetis, satu window IBM utuh, lalu satu fold semua varian | Pipeline selesai, waktu/memori terukur, tidak ada alignment atau penggunaan label holdout yang salah |
| P7 — Matriks utama | Lima fold tanpa undersampling, lalu sweep rasio yang sumbernya sudah dikunci | Semua sel matriks yang direncanakan memiliki hasil atau alasan tidak layak/gagal; kedua cabang tercakup |
| P8 — Replikasi/laporan | Seed tambahan, audit parity final, dokumentasi run dan batas klaim | README/VALIDATION berisi yang benar-benar dijalankan; hasil tidak tercampur dengan smoke atau varian adaptasi |

Pengujian yang diperlukan berfokus pada risiko metode: parsing nominal/label/ID besar, batas window, kestabilan ID setelah resampling, pemisahan namespace node, constant features, rasio sampling, urutan split internal, konsistensi edge, dan alignment embedding dengan label. Uji penggantian label holdout harus membuktikan bahwa model/prediksi tidak berubah ketika RNG dan input lainnya tetap. Uji ini tidak melarang fitur/edge holdout masuk graf batch inductive yang memang diwajibkan referensi.

Parity HinSAGE membandingkan notebook/helper asli dan wrapper modular pada graph fixture yang sama dengan seed, weights, serta sampel tetangga terkendali. Parity FI-GRL membandingkan edge list, degree, sketch terkendali, faktor SVD, dan proyeksi inductive; toleransi harus memperhitungkan ambiguitas tanda/rotasi basis SVD. Kedekatan metrik saja tidak cukup untuk menyatakan implementasi sama.

Smoke berukuran kecil diberi nama/preset khusus. Pengurangan epoch atau dimensi untuk mengecek wiring, jika diperlukan, tidak menjadi hasil replikasi. Pilot utama menggunakan hyperparameter referensi penuh pada satu window; full-data berarti sumber CSV penuh dipindai dan semua transaksi dalam window yang terkunci dipakai, bukan seluruh 29 tahun dimasukkan sekaligus ke satu graf.

Rencana selesai ketika kontrak di atas dapat dijadikan dasar implementasi. **Eksperimen** baru selesai setelah kedua cabang berjalan, artefak terverifikasi, matriks tercatat, dan setiap perbedaan dari notebook dijelaskan. Hal yang masih harus dikunci saat implementasi adalah anchor kalender hasil audit, dependency lock yang benar-benar kompatibel, backend MATLAB yang berfungsi, serta grid undersampling historis bila sweep lengkap paper akan diklaim.
