**EXP14: perbandingan neighbor sampling GraphSAGE lokal**

Tujuan utama EXP14 adalah membandingkan **uniform random, topology-aware, dan importance-based neighbor sampling** pada graf User–Transaction–Merchant dengan model, data, split, jumlah tetangga, seed, dan evaluasi yang sama. Jalur utama bernama `R0`. Semua proses memakai CSV IBM dan cache **lokal**, tanpa Kaggle.

`R0` memakai dua lapis mean GraphSAGE, hidden 256, ReLU, BatchNorm, dan classifier linear. Prediksi memakai sigmoid atas logit; training memakai BCE-with-logits yang ekuivalen dan lebih stabil secara numerik. Dataset mentah memiliki 15 kolom; encoder lokal saat ini menghasilkan **220 fitur turunan** dari kolom tersebut. Ini harus dilaporkan sebagai rekayasa fitur, bukan diklaim sebagai input model 15 dimensi. Uniform memilih tetangga merata; topology memakai Jaccard dan local homophily; importance memakai degree centrality proyeksi transaksi dan PPR tiga langkah sesuai formula EXP12. Ketiganya memakai graf beku pada cutoff train yang sama, tanpa label validation/test pada bobot.

Karena training tetap memakai n30 untuk kelayakan lokal, bobot positif pada batch disesuaikan agar rasio kontribusi kelas ekuivalen dengan inverse-frequency BCE pada populasi penuh. Nilai inverse-frequency populasi dicatat di metrik; memakai angka itu langsung lagi pada sampel n30 akan memberi bobot ganda pada fraud. `R0` tidak memakai bobot kanal tambahan. Perbandingan waktu inferensi mencakup sampling tetangga, pengambilan fitur, dan forward model; waktu membangun bobot dan refresh tabel dicatat terpisah.

Input lokal: `dataset/credit_card_transactions-ibm_v2.csv`. Cache: `model/exp14/cache/`. Hasil: `result/exp14/`. Konfigurasi awal di `config.local.yaml` memilih `R0` dan ketiga strategi secara berurutan untuk seed 42. Jalankan seed tambahan dengan `--seed` setelah kontrol pertama diperiksa.

Dari root repository, dengan virtual environment proyek:

```powershell
$python = '.\code\exp6_gpu_tqdm\.venv\Scripts\python.exe'

# Uji fungsi dan kebijakan anti-leakage pada data sintetis.
& $python -B -m unittest discover -s .\code\exp14_local_temporal\tests -v

# Preprocessing penuh di disk lokal; cache digunakan ulang setelah selesai.
& $python -B .\code\exp14_local_temporal\run.py preprocess

# Tampilkan dukungan fraud per origin sebelum memilih fold.
& $python -B .\code\exp14_local_temporal\run.py audit

# Perbandingan utama penelitian: satu strategi per run, split dan model sama.
& $python -B .\code\exp14_local_temporal\run.py run --variant R0 --strategy uniform --seed 42
& $python -B .\code\exp14_local_temporal\run.py run --variant R0 --strategy topology --seed 42
& $python -B .\code\exp14_local_temporal\run.py run --variant R0 --strategy importance --seed 42

# Analisis temporal tambahan; audit full-data menghasilkan satu origin eligible.
& $python -B .\code\exp14_local_temporal\run.py run --variant B0 --fold origin_1 --seed 42

# Setelah B0 berhasil, lanjutkan ablasi satu perubahan per run.
& $python -B .\code\exp14_local_temporal\run.py run --variant B1 --fold origin_1 --seed 42
& $python -B .\code\exp14_local_temporal\run.py run --variant B2 --fold origin_1 --seed 42
& $python -B .\code\exp14_local_temporal\run.py run --variant B3 --fold origin_1 --seed 42

# Ringkasan selisih berpasangan pada fold/seed yang sama.
& $python -B .\code\exp14_local_temporal\run.py summarize
```

`R0` adalah hasil utama penelitian. Varian A0/A1 dan B0–B4 dari rencana perbaikan temporal lama tetap tersedia sebagai **analisis tambahan** dan tidak menggantikan perbandingan tiga strategi. `topology`/`importance` saat ini sengaja dibatasi pada `R0` dengan graf beku; bobot snapshot harian untuk B belum diimplementasikan dan tidak boleh diganti dengan bobot graf akhir train.

`summary.json` mempunyai `research_gate` yang **tetap pending** sampai uniform dan kandidat masing-masing selesai pada full data untuk seed 42–46 dengan kontrol identik. Gate eksploratif mensyaratkan rerata AP, Recall, dan F1 kandidat lebih tinggi dari uniform, AP menang pada sedikitnya empat dari lima seed, serta rerata AP melampaui AP historis EXP12 terbaik yang tersedia (0,222251). Angka EXP12 itu hanya pembanding deskriptif: konfigurasi model, split timestamp, dan kebijakan threshold berbeda. Hasil test lama sudah diketahui dan tidak dapat menjadi bukti final independen. Jika gate gagal, laporkan kegagalan; revisi parameter hanya boleh dipilih dari validation/development lalu seluruh tiga strategi dijalankan ulang secara setara.

Setiap run menyimpan `config.json`, `history.json`, `status.json`, checkpoint, prediksi selection/calibration/assessment, dan metrik per kanal/bulan serta kuota alert. Checkpoint dipilih pada selection; threshold FPR 0,1% dikalibrasi pada calibration; assessment dibaca terakhir. F2 dan FPR 0,05% juga dihitung sebagai operating point pembanding. Fold 90 hari maju berada sebelum test lama. Audit full-data menemukan Januari–Oktober 2017 tanpa fraud berlabel; konfigurasi awal mengunci satu origin yang berakhir 1 Januari 2017. Selection 6 April–4 Juli 2016, calibration 5 Juli–2 Oktober 2016, assessment 3 Oktober–31 Desember 2016. Ini perubahan terukur dari target awal tiga origin; satu fold tidak cukup untuk klaim peningkatan lintas periode. `fold_manifest.json` melaporkan dukungan fraud per role, sedangkan `temporal_support.json` menambahkan jumlah user, kartu, dan episode fraud. Data berlabel baru bisa diikutkan melalui `protocol.label_delay_days` dalam salinan konfigurasi dengan direktori hasil terpisah.

Cache fitur di-fit pada split awal 70% yang timestamp-nya utuh. Encoder tetap sama pada seluruh rolling fold agar perubahan B0 terutama berasal dari data training lebih baru. Semua fitur perilaku berbasis observasi masa lalu tanpa label. Histori graf B memotong transaksi pada awal hari query; training juga memakai aturan itu. Memmap fitur diurutkan menurut user agar preprocessing dapat memakai satu group user sekaligus; map ID kronologis mengembalikan urutan saat training/evaluasi. Lima fitur `merchant_channel` dihitung pada lintasan kedua menurut merchant dan kanal sehingga tetap mencakup transaksi seluruh user.

Prefix dengan `--max-rows` hanya memeriksa wiring; gunakan `--results-dir result/exp14/nama_smoke` agar artefaknya terpisah dan jangan sebut AP prefix sebagai hasil penelitian. Test 2018–2020 telah dilihat selama pengembangan sehingga hasil R0/A0/A1 di test lama tetap eksploratif. Rencana eksperimen dan kriteria ada di [PLAN.md](PLAN.md); hasil pemeriksaan yang benar-benar selesai ada di [VALIDATION.md](VALIDATION.md).
