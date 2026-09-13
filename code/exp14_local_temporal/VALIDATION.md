**Status pemeriksaan EXP14 lokal — 14 September 2026**

Kesimpulan: jalur utama `R0` kini menyediakan uniform, topology-aware, dan importance-based neighbor sampling pada graf training beku yang sama. Implementasi dan tes sintetis sudah berjalan, tetapi ketiga strategi belum dilatih/dinilai pada full data. Karena itu belum ada bukti topology atau importance mengungguli uniform. Semua proses memakai CSV dan cache lokal; tidak memakai Kaggle.

**Yang sudah diverifikasi**

| Pemeriksaan | Bukti |
| --- | --- |
| Preprocessing CSV penuh | Cache `model/exp14/cache/contextual_acf0555ee7cd9c36/complete.json`: 24.386.900 transaksi, 220 fitur, 2.000 user, 100.343 merchant; selesai dalam 1.680,7 detik, ukuran 14 berkas sekitar 11,33 GiB. |
| Split anti-tie | Batas train 17.070.833 dan awal test 20.728.867; seluruh timestamp yang sama berada di satu sisi batas. |
| Rumus fitur | Tes membandingkan 65 fitur contextual dengan EXP12 pada data sintetis; smoke 1.000 baris CSV nyata menunjukkan galat maksimum sekitar 0,00049 akibat penyimpanan float16, tanpa selisih >0,002. Label held-out yang diubah tidak mengubah fitur. |
| Kausalitas graf | Tes mengecek tetangga sebelum awal hari, tidak memakai transaksi sendiri atau masa depan; forward faktorisasi cocok dengan ekspansi block pada output dan gradient sampel kecil. |
| Bobot tiga strategi R0 | Skor topology (Jaccard + local homophily) dan importance (degree centrality + PPR) pada cutoff train yang sama cocok dengan rumus EXP12 dalam tes sintetis. Sampling ketiga strategi berulang dengan seed yang sama dan tidak memilih transaksi di luar cutoff. `R0` memakai BatchNorm dan koreksi bobot kelas untuk n30. |
| Training lokal | Tes dua epoch sintetis memeriksa checkpoint, isolasi selection/calibration/assessment, serta artefak; smoke satu epoch pada prefix 1 juta baris membuktikan jalur A0 dan B1 dapat berjalan, tetapi AP smoke bukan estimasi populasi. |
| Pengujian | `python -B -m unittest discover -s code/exp14_local_temporal/tests -q`: 18 tes lulus, termasuk smoke training ketiga strategi `R0`, ringkasan berpasangan, dan gate yang menolak kontrol berbeda. |
| Smoke IBM nyata | Ketiga strategi `R0` berhasil melewati satu epoch pada prefix 1.000.000 baris di `result/exp14/smoke_research_1m/`, memakai cache lokal 216 fitur. Assessment sengaja tidak dijalankan; AP selection smoke tidak dipakai untuk menyimpulkan strategi terbaik. |

**Audit kalender data penuh**

Konfigurasi awal tiga origin 90 hari yang berakhir sebelum test lama ternyata menghasilkan role tanpa fraud. Pemeriksaan menemukan Januari–Oktober 2017 tidak memiliki fraud berlabel. Ada pula bug validasi batas saat `label_delay_days=0`: cutoff train dan awal selection memang sama, tetapi semula ditolak sebagai batas duplikat. Bug ini sudah diperbaiki dan diberi tes regresi.

Konfigurasi lokal kini mengunci satu origin dengan assessment berakhir 1 Januari 2017, sebelum celah label. Batas ini dipilih dari audit dukungan label sebelum skor model EXP14 tersedia, sesuai aturan cadangan dalam plan. Manifest ada di `result/exp14/fold_manifest.json`; audit user/kartu/episode ada di `result/exp14/temporal_support.json`.

| Role | Tanggal UTC | Fraud | Chip fraud | User fraud | Kartu fraud | Episode fraud 7 hari |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Selection | 6 Apr–4 Jul 2016 | 856 | 84 | 265 | 284 | 292 |
| Calibration | 5 Jul–2 Okt 2016 | 902 | 71 | 254 | 280 | 295 |
| Assessment | 3 Okt–31 Des 2016 | 697 | 49 | 218 | 231 | 239 |

Satu origin ini memenuhi minimum 25 fraud total dan 25 chip per role, tetapi **tidak memenuhi target tiga origin independen**. Jumlah episode mengurangi kesan bahwa ratusan transaksi fraud sepenuhnya independen. Periode assessment 2016 juga mendahului rezim chip yang mendominasi test 2018–2020; hasilnya nanti tidak boleh digeneralisasi ke rezim tersebut tanpa data evaluasi baru.

**Yang belum selesai atau masih bersyarat**

- R0-uniform/topology/importance serta A0/B0 dan ablation B1–B4 belum dijalankan hingga selesai pada 24,39 juta transaksi; belum ada AP assessment full-data, perbandingan berpasangan, atau klaim peningkatan. Artefak `result/exp14/smoke_*` adalah smoke satu epoch/prefix.
- Replikasi lima seed, sensitivitas label delay 7/30 hari, dan bootstrap blok waktu belum dilakukan. Topology/importance kini tersedia untuk protokol utama `R0` dengan graf beku; bobot topology/importance untuk snapshot harian B masih belum tersedia.
- Snapshot awal hari dan cutoff kausal telah diimplementasikan, tetapi cache embedding/entity per snapshot serta profiling runtime full-data varian B belum selesai. Smoke B1 prefix 1 juta baris satu epoch memakan sekitar 84 detik; angka ini tidak boleh diekstrapolasi langsung ke data penuh. Kapasitas lokal untuk matriks run besar belum terbukti.
- P0 baseline EXP12 n30 pada source/protokol EXP14 yang sama belum selesai. A0 nantinya adalah kontrol baru, bukan reproduksi numerik persis artefak EXP12 lama.
- Test lama 2018–2020 sudah diketahui selama perancangan. Evaluasi R0/A0/A1 pada test itu tetap eksploratif; klaim final memerlukan periode/dataset yang belum dipakai memilih pendekatan.
- `summary.json` kini memisahkan gate R0 dari ringkasan smoke. Tanpa lima pasangan run full-data, statusnya `pending_full_data_replicates`; kode tidak mengklaim peningkatan dari satu seed. AP historis EXP12 terbaik 0,222251 dibaca sebagai konteks eksploratif, bukan baseline yang identik protokolnya.

Sebelum menyatakan penelitian selesai secara empiris, jalankan ketiga strategi `R0` pada seed 42–46 dengan konfigurasi yang sama; laporkan Recall, F1, AP, FN, serta waktu bobot/refresh/inferensi dan selisih berpasangan. A0/B0/B1–B4 adalah analisis tambahan. Test lama sudah diketahui selama pengembangan sehingga hasilnya harus diberi batas eksploratif. Jangan mengubah window berdasarkan AP assessment.
