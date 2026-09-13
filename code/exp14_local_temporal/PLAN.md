**Rencana EXP14 lokal — tindak lanjut hasil EXP12 n30**

Tanggal audit awal: 13 September 2026; penyesuaian arah penelitian: 14 September 2026. **Perbandingan uniform, topology-aware, dan importance-based neighbor sampling adalah eksperimen utama.** Rolling temporal, recency, dan sampling negatif menjadi analisis tambahan yang tidak boleh menggantikan tiga strategi utama. Audit full-data menemukan hanya satu origin 90 hari yang layak sebelum celah fraud 2017; konfigurasi tambahan temporal mengunci assessment berakhir 1 Januari 2017. Cakupan kode dan hasil pemeriksaan dicatat di [README.md](README.md) dan [VALIDATION.md](VALIDATION.md). Dataset tetap CSV IBM lokal; tidak ada Kaggle.

Tujuan utama EXP14 adalah mengukur pengaruh **strategi neighbor sampling** pada deteksi fraud langka dan waktu inferensi GraphSAGE. Uniform menjadi baseline; topology memakai Jaccard dan local homophily; importance memakai degree centrality dan PPR. Kemampuan mendeteksi fraud periode terbaru, terutama chip, tetap dianalisis tetapi bukan pengganti rumusan masalah sampling. Menambah epoch atau memperbesar model belum menjadi prioritas.

**1. Hasil yang sudah diperiksa**

Empat run berikut memakai full data, uniform, seed 42, N:P 30. N30 berarti 30 root normal per satu root fraud saat training; ini berbeda dari jumlah tetangga GraphSAGE. Semua run selesai dengan early stopping normal.

| Varian | AP selection terbaru | AP test | Precision | Recall | F1 fraud | TP / FP / FN | Epoch terbaik / selesai |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| N30 baseline, bobot kanal power 0,5 | 0,190506 | 0,216910 | 25,06% | 27,50% | 0,262256 | 1.225 / 3.663 / 3.229 | 5 / 25 |
| N30 channel equal, power 0 | 0,174852 | 0,200136 | 24,95% | 24,47% | 0,247109 | 1.090 / 3.278 / 3.364 | 5 / 25 |
| N30 chip weight, power 1, cap 8 | 0,174054 | 0,222251 | 39,79% | 21,17% | 0,276377 | 943 / 1.427 / 3.511 | 12 / 32 |
| N30 self-only | 0,165052 | 0,197936 | 29,51% | 22,36% | 0,254439 | 996 / 2.379 / 3.458 | 12 / 32 |

Precision, recall, F1, dan confusion matrix di atas memakai threshold F2 yang dipilih masing-masing run pada calibration validation. Karena threshold berbeda, perbandingan recall tersebut belum merupakan perbandingan pada biaya false alert yang sama.

Pada run yang sedang dibuka di IDE, `channel_equal_n30`, terdapat 3.658.035 transaksi test dengan 4.454 fraud. Model mendeteksi 1.090 fraud, melewatkan 3.364, dan menghasilkan 3.278 false alert. Sekitar 75,53% fraud terlewat. Accuracy 99,82% tidak mencerminkan keberhasilan deteksi kelas langka ini.

Kolom `auprc` pada kode sebenarnya dihitung dengan `average_precision_score`; dokumen ini menyebutnya AP. AP tidak identik dengan luas PR yang dihitung memakai aturan trapesium. Lihat [definisi resmi scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).

Batas kekuatan bukti:

- Hanya seed 42 yang tersedia untuk keempat varian n30. Nama preset multiseed belum berarti lima seed sudah dijalankan; n30 topology/importance juga belum tersedia.
- Data fingerprint dan environment keempat run sama. Tiga ablasi terbaru memiliki source manifest sama; baseline n30 berasal dari revisi yang berbeda pada beberapa file utilitas/config/sampler. Model, trainer, fitur, dan protokol utamanya memiliki hash sama. Perbandingan ini berguna untuk diagnosis, tetapi EXP14 tetap memerlukan baseline pada revisi sumber yang sama.
- Source lokal cocok dengan tiga ablasi terbaru setelah normalisasi LF menjadi CRLF. Perbedaan semua hash mentah lokal terhadap run tersebut berasal dari line ending, bukan bukti bahwa semua file berubah secara semantik.
- Konsistensi `status.json`, `history.json`, epoch terbaik, jumlah confusion matrix, perhitungan precision/recall/F1, serta penjumlahan kanal/bin waktu sudah diperiksa pada empat run. AP belum dihitung ulang dari prediksi per transaksi: artefak prediksi dan checkpoint full-data n30 tidak tersedia pada direktori lokal yang diperiksa. Path checkpoint dalam JSON menunjuk mesin tempat run dibuat.
- Catatan lama `EXP12_FOLLOWUP.md` menyatakan preset lanjutan belum dijalankan, dan `VALIDATION.md` masih menyatakan full data belum dijalankan. Artefak hasil sekarang lebih baru daripada pernyataan tersebut.

**2. Mengapa hasil belum maksimal**

**A. Pergeseran jenis fraud sangat besar — terbukti dari jumlah kasus.**

| Kanal | Fraud train | Fraud validation penuh | Fraud test |
| --- | ---: | ---: | ---: |
| Chip | 256 | 673 | 3.907 |
| Online | 14.771 | 3.450 | 128 |
| Swipe | 5.859 | 294 | 419 |

Chip hanya 1,23% dari fraud train, tetapi 87,72% dari fraud test. Model belajar terutama dari fraud online dan swipe, kemudian harus mendeteksi pola chip yang jauh lebih dominan pada masa berikutnya. Ini menunjukkan pergeseran komposisi fraud; belum membuktikan penyebab eksternal atau mekanisme perubahan labelnya.

Pada channel-equal, recall chip hanya 20,55%: 803 terdeteksi dan 3.104 terlewat. Chip menyumbang 92,27% dari seluruh false negative. AP online mencapai 0,886545 dan recall 93,75%, tetapi hanya ada 128 fraud online di test, sehingga kemampuan itu tidak cukup mengangkat recall total.

Pembobotan lebih besar tidak menciptakan contoh chip baru. Setelah normalisasi, bobot positif chip baseline adalah 3,3287 dan varian chip-weight 5,2893. Porsi chip dalam total massa bobot contoh positif hanya naik dari sekitar 4,08% menjadi 6,48%; jumlah contoh chip unik tetap 256. Ini menjelaskan mengapa perubahan power/cap saja merupakan intervensi yang terbatas. Angka massa bobot bukan ukuran kontribusi gradient aktual.

**B. Full validation memberi gambaran yang terlalu baik untuk rezim terbaru.**

Pada channel-equal, AP full validation checkpoint adalah 0,842174, AP recent selection 0,174852, dan AP test 0,200136. Jadi gap yang relevan untuk pemilihan checkpoint bukan semata-mata 0,84 versus 0,20. Recent selection justru sudah menunjukkan performa yang rendah sebelum test dibaca.

Kode sudah memilih checkpoint berdasarkan recent AP, bukan full validation AP. Selection mempunyai 210 fraud, 196 di antaranya chip; calibration mempunyai 169 fraud, termasuk 153 chip. Akan tetapi, tiga bin berurutan dalam selection berisi **0, 3, dan 207 fraud**. Minimum 25 pada keseluruhan subset belum menjamin representasi waktu yang merata atau banyak episode fraud independen.

Selection dan calibration masing-masing juga memiliki **nol fraud online**. Karena itu, tuning threshold online secara terpisah pada subset calibration saat ini tidak layak, walaupun false alert online cukup banyak. Channel-equal menghasilkan 1.078 FP online untuk 120 TP online pada test.

**C. Ada indikasi overfitting terhadap distribusi training, bukan kekurangan epoch.**

Pada channel-equal, loss turun dari 0,020129 pada epoch 5 menjadi 0,013316 pada epoch 25, tetapi recent AP turun dari 0,174852 menjadi 0,098510. Epoch 5 adalah maksimum recent AP yang tercatat. Berhenti setelah 20 epoch tanpa perbaikan konsisten dengan konfigurasi; status tersebut bukan crash atau training terputus. Kestabilan kesimpulan ini tetap perlu diperiksa antar-seed dan antarperiode.

**D. Konteks graf tertinggal waktu — mekanismenya terbukti, besarnya dampak masih hipotesis.**

Train berakhir 10 Desember 2015; test berlangsung 27 Januari 2018–28 Februari 2020. `context_policy=frozen_train_history` berarti transaksi setelah cutoff train tidak memperbarui histori tetangga graf. Tetangga uniform diambil dari seluruh histori train tanpa preferensi recency atau cutoff yang berbeda untuk setiap query.

Sebaliknya, 65 fitur perilaku sudah menggunakan observasi terdahulu tanpa label, termasuk observasi held-out yang lebih awal. Jadi informasi perilaku bergerak mengikuti waktu, sedangkan histori graf tetap. Embedding dihitung ulang mengikuti bobot model, tetapi transaksi sumber konteksnya tetap lama. EXP14 perlu mengukur apakah pembaruan histori graf memperbaiki ketidaksesuaian ini.

Self-only AP 0,197936 berada di bawah baseline graf 0,216910, selisih 0,018974. Ini mendukung adanya kontribusi graf pada seed tersebut, tetapi belum membuktikan bahwa graf temporal pasti lebih baik. Mean aggregation atas banyak transaksi normal juga mungkin mengaburkan sinyal lokal; hipotesis ini memerlukan audit umur dan komposisi tetangga.

**E. Threshold mengubah trade-off, bukan memperbaiki ranking.**

| Channel-equal | Threshold | TP | FP | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tetap | 0,5 | 813 | 1.107 | 42,34% | 18,25% |
| F2 dari validation | 0,242638 | 1.090 | 3.278 | 24,95% | 24,47% |

Tambahan 277 fraud terdeteksi memerlukan tambahan 2.171 false alert. Pada chip-weight, threshold F2 justru 0,847763; recall test 21,17%, sementara pada threshold 0,5 recall 31,25% dengan 5.614 FP. Maka turunnya recall varian chip-weight tidak dapat diartikan sebagai hilangnya seluruh sinyal deteksi; skala skor dan operating point juga berubah.

AP tetap sama ketika hanya threshold keputusan diganti. Pemisahan kualitas skor dan kebijakan keputusan ini sesuai [dokumentasi threshold scikit-learn](https://scikit-learn.org/stable/modules/classification_threshold.html).

Pada bin test terakhir, 9 Oktober 2019–28 Februari 2020, channel-equal mempunyai AP 0,045884, 32 TP, 912 FP, recall 17,30%, dan precision 3,39%. Prevalensi bin ini juga turun menjadi 0,0253%; penurunan AP tidak boleh seluruhnya disebut penurunan kemampuan model. Recall dan FP memperlihatkan bahwa kualitas operasional periode terbaru tetap bermasalah.

**F. Sampling normal dan ketepatan waktu perlu diaudit.**

N30 mengambil seluruh 20.886 fraud train serta 626.580 normal per epoch. Normal itu sekitar 3,67% dari 17.049.944 normal train, diacak kembali setiap epoch; seluruh graf train tetap tersedia. Belum diketahui apakah sampling tersebut cukup mencakup normal chip terbaru yang sulit dibedakan dari fraud. Mengubah distribusi negatif lebih terarah merupakan hipotesis lanjutan, bukan perbaikan yang sudah terbukti.

Artefak preprocessing juga mencatat `tie_at_split_boundary.train_val=true` dan `val_test=true`. Backend pandas membagi berdasarkan posisi baris sehingga timestamp yang sama melintasi split. Fitur perilaku sudah mengecualikan peer dengan timestamp sama, tetapi batas split belum memenuhi kontrak waktu yang ketat. Ini harus diperbaiki dan baseline dihitung ulang; belum ada bukti bahwa hal tersebut menjelaskan besarnya penurunan AP.

**3. Keputusan rancangan EXP14**

Jalur penelitian utama `R0` memakai mean GraphSAGE dua layer, hidden 256, ReLU, **BatchNorm**, sigmoid saat prediksi, dan BCE-with-logits saat training. Seluruh strategi memakai 220 fitur hasil encoding/rekayasa dari 15 kolom mentah yang sama, fanout 25, split timestamp yang sama, N:P 30, dan seed yang sama. Ini mempertahankan sumber data 15 atribut tetapi harus dilaporkan sebagai input terenkode 220 dimensi. Bobot positif batch dikoreksi untuk negative subsampling: inverse-frequency populasi train sekitar ratusan kali, sedangkan rasio efektif pada sampel n30 adalah 30; menerapkan bobot populasi langsung pada n30 akan membobot fraud dua kali. Tidak ada bobot kanal tambahan di `R0`. Konsep agregasi mengikuti [GraphSAGE asli](https://arxiv.org/abs/1706.02216).

Eksperimen dibagi menjadi jalur utama dan analisis tambahan yang dilaporkan terpisah:

| Protokol | Tujuan | Aturan |
| --- | --- | --- |
| R0: perbandingan utama | Menguji hipotesis sampling penelitian | Uniform/topology/importance memakai graf training beku, fitur/model/loss/root sampling/split/fanout/seed yang sama; hanya bobot pemilihan tetangga yang berubah |
| A: kontrol statis | Mengukur perubahan terhadap rancangan EXP12 | Split sekitar 70/15/15 dengan seluruh timestamp yang sama berada pada satu split; encoder dan graf beku di cutoff train |
| B: rolling temporal | Mengurangi jarak model/histori terhadap periode prediksi | Training, selection, calibration, dan assessment bergerak maju; akses label dibatasi waktu ketersediaannya |

R0 wajib selesai dan direplikasi sebelum manfaat tambahan A/B ditafsirkan sebagai hasil utama. Bila B membaik, hasilnya harus disebut manfaat protokol rolling, bukan otomatis manfaat algoritma sampling. Bobot R0 dihitung hanya dari transaksi dan label train sebelum cutoff bersama; topology mengecualikan label kandidat dari statistiknya, dan importance memakai degree transaksi terproyeksi serta PPR tiga langkah. Bobot dari graf akhir train tidak dipakai untuk snapshot sebelumnya.

**4. Tahap dan kriteria untuk lanjut**

| Tahap | Pekerjaan yang direncanakan | Keluaran dan kriteria lanjut |
| --- | --- | --- |
| R0a — audit baseline | Pastikan split timestamp, graf train, fitur, BatchNorm, bobot loss, fanout, dan unit waktu sama pada ketiga strategi | Manifest data/source/split serta uji bobot terhadap EXP12; tidak ada label validation/test pada bobot |
| R0b — tiga strategi | Jalankan uniform, topology-aware, importance-based pada seed dan protokol yang sama | Recall, F1, AP/AUPRC, FN, waktu membangun bobot, refresh tabel, dan waktu inferensi per transaksi tersimpan |
| R0c — replikasi | Bandingkan selisih berpasangan pada lima seed, kanal, periode, dan budget alert yang sama | Kesimpulan hipotesis sampling dan trade-off akurasi/waktu; laporkan ketidakpastian serta kegagalan bila selisih tidak konsisten |
| Tambahan A/B | Audit rolling refit, recency, negatif training, threshold menurut rincian di bawah | Hasil dipisahkan dari R0 dan tidak dijadikan bukti efek topology/importance |

Rincian protokol temporal tambahan yang berasal dari rancangan awal:

- Gunakan forward chaining: `train < selection < calibration < assessment`. Assessment fold untuk memilih rancangan EXP14 masih merupakan data pengembangan, bukan test final yang independen.
- Rancangan awal selection, calibration, dan assessment masing-masing 90 hari; origin bergeser 90 hari dan seluruh assessment berakhir sebelum test lama. Training awal memakai expanding history agar 256 contoh chip tidak terbuang oleh window sempit.
- Target tiga origin. Audit jumlah kasus sebelum training; minimum awal 25 fraud total pada selection/calibration dan 25 fraud chip untuk keputusan khusus chip. Jumlah kartu/user/episode juga dilaporkan karena transaksi berdekatan tidak selalu merupakan bukti independen.
- Jika dukungan tidak cukup, nyatakan fold/subgroup tidak memadai. Alternatif window 180 hari boleh ditetapkan sebagai protokol pengganti melalui audit dukungan sebelum training, bukan dipilih karena skornya lebih tinggi. Jika tetap tidak tersedia tiga origin yang layak, laporkan jumlah origin aktual dan keterbatasannya; jangan meminjam test.
- Checkpoint memakai AP selection, threshold memakai calibration, dan rancangan dibandingkan pada assessment fold. Rata-rata AP assessment antar-origin menjadi ukuran ranking utama; laporkan juga tiap origin, prevalensi, AP chip, dan variasinya. Metrik subgroup tanpa fraud dilaporkan NA, bukan diberi nilai nol atau dihilangkan diam-diam. Kandidat tidak dipromosikan hanya karena membaik pada origin yang didominasi online: dukungan dan hasil chip pada origin terbaru harus cukup untuk menilai tujuan EXP14.
- Baseline label delay 0 hanya asumsi eksperimen. Uji sensitivitas 7/30 hari untuk kandidat akhir; training root dan statistik berlabel hanya boleh memakai label yang sudah tersedia. Cutoff selection/calibration juga harus menghormati delay.
- Jangan melakukan refit setelah threshold dikunci lalu memakai threshold lama untuk model baru. Setiap model hasil refit membutuhkan selection/calibration yang sesuai tanpa assessment leakage.

Rincian P2–P3:

- P2 membandingkan expanding refit pada setiap origin dengan kontrol model yang dilatih pada cutoff awal dan kemudian dibekukan, pada periode assessment yang persis sama. Kedua lengan B memakai aturan snapshot kausal yang sama saat training; A0 hanya referensi protokol statis, bukan kontrol tunggal untuk mengisolasi refit. Refit memakai data berlabel yang sah sebelum selection; evaluasi pertama tetap tanpa pembaruan label dari assessment.
- Pada B0 dan kontrol B yang dibekukan, histori untuk query held-out berhenti pada cutoff fit masing-masing. Pada B1 histori tersebut boleh bergerak mengikuti observasi terdahulu tanpa label. Aturan snapshot training B0/B1 sudah identik sejak awal agar perubahan B1 dapat dikaitkan dengan akses histori held-out, bukan sekaligus perubahan kausalitas training.
- P3 mula-mula menguji perubahan akses histori saja: ukuran fanout, model, fitur, training roots, dan seed tetap. Histori unlabeled boleh memasukkan transaksi yang telah terjadi sebelum query, meskipun berasal dari periode validation/assessment; labelnya tidak masuk sampler, fitur, atau optimizer.
- Gunakan snapshot harian dengan cutoff awal hari sebagai rancangan implementasi awal. Seluruh root hari itu hanya melihat transaksi sebelum cutoff. Ini sedikit menunda histori intrahari, tetapi membuat aturan kausal jelas dan memungkinkan cache entity per snapshot.
- Training dan evaluasi varian temporal harus memakai aturan snapshot yang sama. Cache bertanda cutoff; cache seluruh-epoch EXP12 tidak boleh langsung dipakai untuk query dengan waktu berbeda. Uji hasil faktorisasi terhadap ekspansi block pada sampel kecil sebelum memakai backend cepat.
- Setelah akses histori terbukti berguna, lakukan ablasi recency tersendiri: 12 tetangga uniform dari 180 hari terakhir + 13 dari histori eligible lainnya, total maksimum 25 tanpa duplikasi. Kekurangan kandidat diisi dari sisa histori yang sah, tanpa memaksa fanout penuh. Angka 180 dan 12/13 merupakan kandidat awal yang harus dibekukan sebelum evaluasi.
- Catat umur tetangga, jumlah kandidat, proporsi query tanpa histori, waktu refresh, dan latency termasuk persiapan. Perubahan ini terinspirasi kebutuhan representasi waktu pada [Temporal Graph Networks](https://arxiv.org/abs/2006.10637), tetapi EXP14 tetap GraphSAGE; tidak mengklaim mengimplementasikan TGN.

Rincian P4:

- Kandidat pertama: 50% normal dipilih uniform global dan 50% dialokasikan menurut kanal dan kuartal contoh fraud training yang tersedia. Sampling tanpa replacement dalam epoch, tetap 30 normal per fraud, fraud unik tetap sekali per epoch, bobot positif baseline tetap.
- Bila suatu strata kekurangan normal, alihkan kekurangan ke pool uniform dan catat jumlahnya. Proporsi tidak boleh berasal dari label validation/test.
- Ukur apakah FP chip/online dan recall pada budget yang sama membaik. Jangan menyebut kandidat ini hard-negative mining sebelum ada pemilihan berdasarkan skor model.
- Hard-negative mining berbasis skor out-of-fold temporal hanya langkah cadangan jika kandidat sederhana gagal. Jangan memakai FP test atau prediksi model yang dilatih pada baris itu sebagai dasar klaim generalisasi.
- Menyeimbangkan kanal secara agresif dengan mengulang 256 fraud chip bukan prioritas. Mengurangi training ke 12/24 bulan hanya diuji setelah audit memastikan dukungan fraud cukup; uji tersebut terpisah dari perbaikan distribusi negatif.

Rincian P5:

- Pertahankan F2 global sebagai pembanding; kandidat kebijakan utama memaksimalkan recall pada calibration dengan FPR maksimum 0,1%. Laporkan juga FPR 0,05% dan precision/recall pada kuota 0,1% serta 0,2% transaksi. Angka ini adalah budget eksperimen awal, bukan kebijakan operasional yang sudah disepakati.
- Threshold dipilih pada calibration lalu dibekukan pada assessment. FPR aktual assessment bisa melampaui budget calibration akibat drift; pelanggaran tersebut harus dilaporkan.
- Precision@K/recall@K dihitung pada blok assessment dengan K yang ditentukan dari kuota, memakai ranking skor tanpa label untuk menentukan alert. Ini adalah kebijakan batch dengan budget tetap; untuk prediksi real time, gunakan threshold dari masa lalu dan laporkan alert rate aktual.
- Jangan mengklaim threshold meningkatkan AP. Jika nanti menambahkan kalibrasi probabilitas, gunakan subset/cross-fitting temporal yang terpisah dari pemilihan threshold dan evaluasi model.
- Threshold per kanal ditunda sampai calibration kanal memiliki dukungan fraud yang cukup; gunakan global fallback yang eksplisit. Nol fraud online pada calibration EXP12 tidak boleh menghasilkan threshold online yang dianggap tervalidasi.

**5. Matriks eksperimen yang dibatasi**

| ID | Varian | Pembanding langsung / satu perubahan utama |
| --- | --- | --- |
| R0-uniform | Mean GraphSAGE + uniform neighbor sampling | Baseline utama penelitian |
| R0-topology | R0-uniform + Jaccard/local homophily | Efek bobot topology pada graf train yang sama |
| R0-importance | R0-uniform + degree centrality/PPR | Efek bobot importance pada graf train yang sama |
| A0 | N30 default pada source terkunci dan split timestamp yang benar | Kontrol tambahan historis EXP12; catat perubahan jumlah baris dari split EXP12 |
| A1 | A0 self-only | Kontribusi graf pada protokol statis yang sama |
| B0 | N30 expanding refit, histori query held-out beku per-fit | Kontrol B pada cutoff awal, dengan training snapshot kausal yang sama; dampak model/data training lebih baru |
| B1 | B0 + snapshot histori unlabeled yang bergerak untuk query held-out | B0; training snapshot tetap sama, akses histori held-out yang berubah |
| B2 | B1 + campuran tetangga recent/historical | B1; dampak preferensi recency |
| B3 | B0 + sampling normal menurut kanal/waktu | B0; dampak distribusi negatif |
| B4 | Kombinasi komponen yang lolos | Hanya jika ablasi komponennya mendukung; kemudian uji self-only pada protokol B yang sama |

Lakukan pemeriksaan alur dan profiling pada sampel kecil. Ketiga strategi R0 harus dijalankan pada seed 42/43/44/45/46; jangan menyaring topology atau importance berdasarkan seed 42 saja. A/B tambahan boleh disaring bertahap setelah R0.

Perbandingan uniform/topology/importance dilakukan **lebih dahulu** pada lima seed di dalam protokol R0 yang sama. Eligibility dan fanout identik. Bobot topology berlabel dan statistik PPR/degree dihitung pada cutoff train bersama, tanpa label validation/test; label kandidat train dikeluarkan dari statistik homophily. `literal_transaction` mempunyai degree transaksi konstan, sehingga konfigurasi utama importance memakai `projected_transaction`; mode literal dapat menjadi ablation terpisah.

**6. Tolok ukur keberhasilan**

Keberhasilan hipotesis utama dinilai dari selisih `R0-topology` dan `R0-importance` terhadap `R0-uniform` pada seed, split, fitur, arsitektur, loss, fanout, dan threshold policy yang sama; bukan terhadap angka test lama yang dipilih paling tinggi. Ukur waktu membangun bobot, refresh tetangga, dan inferensi termasuk sampling/fitur/model agar biaya komputasi tidak tersembunyi.

Gate pelaporan EXP14 tetap `pending` sampai lima seed 42–46 full-data selesai untuk uniform dan strategi kandidat dengan source/data/protokol identik. Kandidat baru boleh disebut membaik secara **eksploratif** jika rerata AP, Recall, dan F1 melebihi uniform, AP menang minimal pada empat dari lima seed, serta rerata AP melampaui AP historis EXP12 terbaik 0,222251. Perbandingan AP historis ini deskriptif karena model dan split berubah; F1/Recall historis tidak dibandingkan langsung pada threshold yang berbeda. Bila syarat gagal, laporkan hasil negatif dan ubah rancangan hanya berdasarkan validation/development, kemudian ulangi ketiga strategi secara adil. Tidak ada kode yang dapat menjamin peningkatan sebelum eksperimen.

- Laporkan Recall, F1-score fraud, AP/AUPRC, FN, dan waktu inferensi untuk ketiga strategi R0; AP chip serta tiap periode turut dilaporkan. Usulan target praktis awal: kenaikan AP absolut minimal 0,02 dan recall chip minimal 5 poin persentase pada budget alert yang sama. Ini kriteria promosi kandidat, bukan janji hasil.
- Peningkatan muncul pada mayoritas origin dan seed; laporkan mean, simpangan baku, dan selisih berpasangan. Ketidakpastian data sebaiknya memakai bootstrap blok waktu, bukan hanya menganggap jutaan transaksi independen.
- Bandingkan precision, recall, F1 fraud, TP/FP/FN, alert rate dan FPR aktual, termasuk periode terakhir. Kenaikan recall dengan lonjakan alert tidak dianggap perbaikan tanpa menunjukkan trade-off.
- Jika selisih tidak konsisten atau hanya muncul pada satu seed/window, simpulkan belum terbukti. Jangan menaikkan target berdasarkan test yang sudah terlihat.
- Test lama 2018–2020 tetap evaluasi eksploratif. Penggunaan fold hanya sebelum test mengurangi kebocoran keputusan berikutnya, tetapi tidak membuat test lama kembali independen. Klaim final membutuhkan periode/dataset baru yang belum dipakai memilih pendekatan.

**7. Rencana sumber daya lokal**

Artefak n30 berasal dari RTX 5060 Ti 16 GB dan peak VRAM sekitar 1,44 GiB. Mesin aktif saat audit memiliki RTX 3050 Laptop 4 GB dan RAM sekitar 15,68 GiB. Waktu run 142–306 detik pada mesin sumber tidak boleh dianggap estimasi untuk mesin aktif atau untuk protokol temporal baru.

220 fitur float16 saja memerlukan 10.730.236.000 byte, sekitar **9,99 GiB**, belum termasuk CSV/DataFrame, graf, indeks, tensor temporer, dan proses lain. Karena itu, preprocessing pandas full-data pada mesin aktif tidak menjadi asumsi kelayakan.

Rancangan lokal yang perlu disiapkan saat implementasi:

- Simpan fitur/endpoint/label/timestamp dan CSR sebagai memmap serta lakukan chronological external sort dengan batas RAM. DuckDB atau mekanisme disk lokal dapat dipakai tanpa layanan cloud.
- Dukungan 220 fitur contextual harus dipertahankan dan diuji. Backend streaming EXP12 saat ini dibatasi pada encoder `proposal`; mengganti flag ke DuckDB atau menyalin preset Kaggle akan mengubah fitur dan bukan solusi setara.
- Mulai batch training 512/1.024 dan evaluasi 2.048 pada GPU 4 GB; ukur sebelum dinaikkan. Fitur tetap di CPU/disk, hanya batch dan context yang diperlukan masuk GPU. Angka batch adalah rencana profiling, belum tervalidasi.
- Target awal peak RSS di bawah 12 GiB dan VRAM di bawah 3,2 GiB pada mesin aktif. Jalankan satu run setiap saat; cache aktif dibatasi per snapshot/fold.
- Audit ruang disk berdasarkan ukuran array, cache, checkpoint, dan spill sebelum full-data. Pisahkan cold preprocessing, cache load, training, context refresh, serta inference; `preprocessing_seconds=0,278` pada artefak cache lama tidak membuktikan kecepatan preprocessing CSV penuh.
- Smoke dari prefix CSV hanya untuk wiring. Evaluasi arah memakai rentang waktu yang sesuai populasi dan tetap menyertakan histori yang diperlukan.

**8. Daftar perubahan yang direncanakan untuk implementasi nanti**

| Area/file rencana di `code/exp14_local_temporal/` | Tanggung jawab |
| --- | --- |
| `audit_exp12.py` | Audit artefak dan dukungan waktu/kanal/episode, baseline report |
| `exp14/data.py`, `exp14/feature_store.py` | Split timestamp yang ketat; encoder contextual dan memmap lokal |
| `exp14/protocol.py` | Manifest kalender fold, label delay, selection/calibration/assessment |
| `exp14/sampling/temporal.py` | Eligibility snapshot, histori unlabeled, recency, invalidasi cache |
| `exp14/sampling/roots.py` | Kontrol n30 dan distribusi normal menurut kanal/waktu |
| `exp14/trainer.py`, `exp14/evaluation.py` | Training per fold dan penyimpanan skor/threshold/diagnostik |
| `tests/` | Invariansi future/label, equal timestamp, isolasi fold, ekuivalensi backend temporal, kesetaraan fitur memmap |
| `config.*.yaml`, `run.py` | Varian terdaftar dan eksekusi lokal berurutan |
| `VALIDATION.md` | Bukti pemeriksaan implementasi yang benar-benar sudah selesai |

Artefak implementasi kelak: `result/exp14/` dan `model/exp14/`, dengan identitas source/data/protocol/seed, checkpoint portabel, prediksi per transaksi beserta source row/timestamp/channel, metrik per fold/bulan/kanal, dan resource report. Hasil lama EXP12 tetap menjadi referensi audit.

**Sumber lokal audit**

- [Baseline n30](../../result/exp12/exp12_recent_context_c39b0e23c708fe40_uniform_seed42/metrics.json).
- [Channel equal n30](../../result/exp12/exp12_channel_equal_n30_e0fe1853a94ef333_uniform_seed42/metrics.json), beserta `status.json` dan `history.json` pada folder yang sama.
- [Chip weight n30](../../result/exp12/exp12_chip_weight_n30_dd39a2ff0c5542ec_uniform_seed42/metrics.json).
- [Self-only n30](../../result/exp12/exp12_self_only_n30_edc36ce9ebc6f172_uniform_seed42/metrics.json).
- [Metadata preprocessing full data](../../result/exp12/preprocessing_8937270a4eb3b0b6.json).
- [Trainer EXP12](../exp12_recent_context/exp12/trainer.py), [protokol](../exp12_recent_context/exp12/protocol.py), [graf](../exp12_recent_context/exp12/graph.py), [root sampling](../exp12_recent_context/exp12/minibatch.py), dan [metrik](../exp12_recent_context/exp12/metrics.py).

Hasil EXP12/EXP13 tetap sebagai referensi; kode EXP14, preset lokal, pengujian, dan artefak validasinya ditempatkan pada direktori EXP14 sendiri. Matriks full-data dan klaim peningkatan akurasi hanya dapat dinyatakan setelah run yang relevan selesai dan dicatat di `VALIDATION.md`.
