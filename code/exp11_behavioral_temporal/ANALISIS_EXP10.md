# Analisis EXP10 dan arah EXP11

Audit 9 September 2026 atas **full data 24.386.900 transaksi**, comparison ID
`81de71eac888b22b`, seed 42. Sumber: `result/exp10/exp10_temporal_robust_81de71eac888b22b_*_seed42/`
dan [ringkasan audit yang dapat dibuat ulang](validation/exp10_audit.json).
Pilot satu juta baris dan smoke EXP10 tidak dicampurkan dengan hasil ini.

## 1. Berhenti otomatis: early stopping, bukan interrupt

| Strategi | Epoch terbaik | Epoch terakhir | Stale terakhir | Status tersimpan |
|---|---:|---:|---:|---|
| Uniform | 15 | 30 | 15 | complete |
| Topology | 5 | 20 | 15 | complete |
| Importance | 5 | 20 | 15 | complete |

`training.epochs: 100` adalah batas maksimum. `early_stopping_patience: 15`
memutus loop setelah 15 epoch tanpa kenaikan selection score sebesar min_delta.
Kode kemudian memulihkan checkpoint terbaik, mengkalibrasi threshold,
mengevaluasi test, dan menyimpan status `complete`. Ketiga run mengikuti pola itu.
`KeyboardInterrupt` pada kode EXP10 justru akan menyimpan `interrupted`.

Tidak ditemukan bukti crash atau kehabisan VRAM pada ketiga artefak tersebut.
Kesimpulan ini terbatas pada hasil yang tersedia; tidak mencakup pesan console
atau attempt lain yang tidak disertakan. Scheduler menurunkan learning rate;
perintah berhenti datang dari loop early stopping, bukan dari scheduler.
Perilaku scheduler sesuai [dokumentasi PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html).

Memaksa 100 epoch tidak menyelesaikan penyebab utama. EXP11 mempertahankan
early stopping, menambah minimum 20 epoch/patience 20 pada preset utama, dan
mencatat `stop_reason`, epoch terbaik/terakhir, serta stale/patience.

## 2. Ranking dan deteksi memang masih lemah

Metrik menggunakan threshold dari validation EXP10, bukan threshold test.

| Strategi | AP validation | AP test | F1 fraud test | Recall test | TP / 4.454 | FP |
|---|---:|---:|---:|---:|---:|---:|
| Uniform | 0,722792 | 0,035268 | 0,047862 | 6,53% | 291 | 7.415 |
| Topology | 0,698190 | 0,028505 | 0,030011 | 5,14% | 229 | 10.578 |
| Importance | 0,701341 | 0,029190 | 0,031454 | 6,11% | 272 | 12.569 |

Uniform paling baik pada seed ini; satu seed belum cukup untuk klaim keunggulan
strategi. AP adalah average precision yang dilaporkan pipeline sebagai AUPRC.
Accuracy tinggi tidak membuktikan deteksi baik: prevalensi test hanya 0,12176%,
sehingga prediksi seluruh transaksi normal pun mencapai accuracy 99,87824%.

## 3. Pola fraud berubah, meskipun prevalensi global hampir sama

- Validation: **3.450/4.417 fraud online (78,11%)**, fraud chip 673.
- Test: **3.907/4.454 fraud chip (87,72%)**, fraud online hanya 128.
- Uniform test: AP online **0,773267**, AP chip **0,009768**.
- Recall chip uniform hanya **157/3.907 = 4,02%**.

Audit dataset sebelumnya pada cutoff training yang sama menemukan hanya
**256 fraud chip**, dibanding **14.771 online** dan **5.859 swipe** (total
20.886 fraud train, sama dengan laporan EXP10). Audit itu memakai cutoff
timestamp ketat; perbedaan transaksi pada boundary ties dijelaskan di
[audit dataset EXP9](../exp10_temporal_robust/validation/exp9_audit.json).
Jadi subtype yang paling dibutuhkan pada test sangat sedikit contoh latihnya.
Balancing fraud-vs-normal saja tidak menyeimbangkan subtype ini.

Jadi model masih berhasil meranking fraud online, tetapi gagal pada jenis fraud
yang mendominasi periode berikutnya. Ini bukti perubahan komposisi/relasi fitur
dan label; bukan sekadar masalah threshold atau jumlah epoch.

Train berakhir Desember 2015, validation Januari 2018, dan test Februari 2020.
Graf dan agregasi tetangga EXP10 membeku pada histori train. Uniform, topology,
dan importance memakai fitur, model, roots, serta snapshot yang sama; mengganti
distribusi tetangga saja tidak menambahkan sinyal perilaku transaksi terbaru.

## 4. Kelemahan fitur dan preprocessing

Yang sudah benar pada EXP10: urutan kronologis, robust signed-log amount,
one-hot kategori nominal, missing/OOV eksplisit, dan vocabulary/statistik
train-only. Tidak ada alasan menggantinya dengan ordinal ID atau random split.
Pemisahan statistik preprocessing dari held-out data mengikuti
[panduan scikit-learn tentang data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage-during-pre-processing).

Diagnostik full-data juga tidak menunjukkan kegagalan vocabulary/scaling:
OOV `Use Chip`, `MCC`, dan `Errors?` **0%** pada validation/test; amount clip
**0%**; lokasi OOV test hanya **0,0753%**. User tanpa histori train mencapai
**3,8238%** transaksi test dan merchant tanpa histori **0,5396%**. Jadi ada
cold-start, tetapi bukti terkuat tetap kegagalan ranking fraud chip dan
kekurangan fitur perilaku, bukan kategori test yang semuanya tidak dikenal.

Yang belum tersedia:

1. Nominal relatif terhadap kebiasaan user/kartu. Nominal yang biasa untuk
   satu user bisa sangat tidak biasa untuk user lain.
2. Kecepatan transaksi: frekuensi 1 jam/24 jam/7 hari dan jeda transaksi.
3. Merchant/MCC/kanal/lokasi yang baru atau jarang untuk user tersebut.
4. Riwayat kartu individual. EXP10 bahkan tidak membaca kolom `Card`.
5. Pembaruan ringkasan perilaku tanpa label ketika transaksi baru tiba.

EXP11 menambah 24 fitur ini. Angka jumlah, jeda, dan nominal menggunakan
transformasi log, variance floor, dan clip untuk mencegah skala ekstrem.
`User`/`Card` dipakai sebagai kunci histori, bukan bilangan ordinal input model.
Statistik nominal mengabaikan amount non-finite. Kartu kosong memakai histori
user dengan warning dan catatan audit, tidak menggabungkan card ID lintas user.

**Batas metodologi:** fitur perilaku strictly earlier, termasuk menolak semua
transaksi dengan timestamp sama. Agregasi GraphSAGE tetap frozen-train snapshot,
sehingga graf training keseluruhan belum strictly causal per transaksi.
Histori perilaku saat validation/test boleh menerima observasi sebelumnya tanpa
label; perubahan ini memerlukan counter state pada sistem deployment.

## 5. Selection dan threshold belum memecahkan ranking chip

Pada checkpoint uniform, AP-lift tiga blok validation adalah 345,53;
875,33; dan 11,33. Geometric mean **150,75** terlihat besar, tetapi AP blok
terakhir hanya sekitar **0,00352**. Performa periode lama yang mudah masih
dapat menutupi kelemahan periode terbaru.

Threshold uniform 0,754843 memaksimalkan F1 di paruh terakhir validation,
tetapi F1 di segmen kalibrasi itu sendiri hanya **0,02118**, recall **6,07%**.
Median skor fraud test hanya **0,01221**. Menurunkan threshold bisa menambah
recall sekaligus false positive; itu tidak meningkatkan AP.

EXP11 memilih checkpoint lewat AP window validation terbaru sebelum batas 75%.
Awal window default 50%; diperluas mundur jika perlu untuk menampung target
25 fraud, atau seluruh fraud yang tersedia sebelum batas tersebut. Akhirnya
tetap 75%. Sisa 25% khusus kalibrasi. Full validation AP hanya laporan.
Window satu kelas ditolak dengan alasan jelas; tidak mengambil label test.

Default threshold EXP11 memaksimalkan F2 pada bagian kalibrasi agar recall
mendapat bobot lebih besar. Ini **perubahan kebijakan**, bukan jaminan F1 naik.
AP, precision, F1, recall, FP, alert rate, dan metrik @0,5 tetap dilaporkan.

## 6. Balancing dan model

EXP10 memakai 20 negatif/fraud: prevalensi roots sekitar 4,76%, sedangkan
populasi training sekitar 0,122%. Walau `pos_weight=1`, sampling sudah
mengubah biaya efektif fraud sekitar **40,8 kali** terhadap populasi.
Skor sigmoid bukan probabilitas risiko populasi yang terkalibrasi.

EXP11 memakai 50 negatif/fraud, tetap seluruh fraud sekali per epoch,
serta penekanan subtype fraud langka dari label **train saja**. Bobot subtype
menggunakan pangkat 0,5, cap rasio 4, dan normalisasi mean bobot fraud menjadi 1.
Ini tidak mengalikan ulang inverse-frequency fraud global. Semua strategi
mendapat kebijakan identik. LayerNorm dipakai agar normalisasi tidak bergantung
campuran transaksi/entitas dalam batch yang diseimbangkan.

Budget roots/epoch dan batch berubah. Durasi atau epoch yang sama bukan
perbandingan komputasi yang sama. [Preset fitur-only](config.features_only.yaml)
mempertahankan budget/model/selection EXP10 untuk memisahkan pengaruh fitur.

## 7. Cara menyimpulkan hasil EXP11

Lihat [VALIDATION.md](VALIDATION.md) untuk hasil yang benar-benar dijalankan.
Preset utama tetap split 70/15/15. Preset recent-refit 80/5/15 adalah protokol
berbeda, karena memakai label train lebih baru; jangan mencampur klaimnya.
Evaluasi chip, beberapa seed, dan ablation diperlukan sebelum menyatakan
peningkatan full-data. Test yang sudah dianalisis berulang pada EXP9/10/11
merupakan evaluasi eksploratif; klaim final membutuhkan periode/dataset baru
yang belum dipakai mengarahkan desain.
