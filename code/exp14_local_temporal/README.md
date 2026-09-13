**EXP14: GraphSAGE temporal lokal**

EXP14 menjalankan dua-layer mean GraphSAGE dengan 220 fitur contextual pada CSV IBM penuh. Preprocessing memakai DuckDB **lokal** untuk mengurutkan transaksi dan NumPy memmap untuk fitur/indeks. Formula 65 fitur perilaku dan definisi model memakai modul EXP12 yang sudah diuji; backend 15 fitur `proposal` dan deployment Kaggle tidak dipakai.

Input lokal: `dataset/credit_card_transactions-ibm_v2.csv`. Cache: `model/exp14/cache/`. Hasil: `result/exp14/`. Konfigurasi awal ada di `config.local.yaml`; run default hanya A0 seed 42 agar perubahan berikutnya dilakukan menurut urutan ablation pada [PLAN.md](PLAN.md).

Dari root repository, dengan virtual environment proyek:

```powershell
$python = '.\code\exp6_gpu_tqdm\.venv\Scripts\python.exe'

# Uji fungsi dan kebijakan anti-leakage pada data sintetis.
& $python -B -m unittest discover -s .\code\exp14_local_temporal\tests -v

# Preprocessing penuh di disk lokal; cache digunakan ulang setelah selesai.
& $python -B .\code\exp14_local_temporal\run.py preprocess

# Tampilkan dukungan fraud per origin sebelum memilih fold.
& $python -B .\code\exp14_local_temporal\run.py audit

# Kontrol statis A0; evaluasi test lama diberi label eksploratif.
& $python -B .\code\exp14_local_temporal\run.py run --variant A0 --seed 42

# Audit full-data saat ini menghasilkan satu origin yang eligible.
& $python -B .\code\exp14_local_temporal\run.py run --variant B0 --fold origin_1 --seed 42

# Setelah B0 berhasil, lanjutkan ablasi satu perubahan per run.
& $python -B .\code\exp14_local_temporal\run.py run --variant B1 --fold origin_1 --seed 42
& $python -B .\code\exp14_local_temporal\run.py run --variant B2 --fold origin_1 --seed 42
& $python -B .\code\exp14_local_temporal\run.py run --variant B3 --fold origin_1 --seed 42

# Ringkasan selisih berpasangan pada fold/seed yang sama.
& $python -B .\code\exp14_local_temporal\run.py summarize
```

Varian A1 mematikan konteks graf. B0 melatih ulang pada cutoff tiap origin dan membekukan histori query setelah fit. `B_control` memakai checkpoint B0 dari origin layak pertama pada origin berikutnya; jalankan B0 origin pertama terlebih dahulu. B1 memperbarui histori query dari observasi terdahulu tanpa label. B2 menambah prioritas 12 tetangga dari 180 hari terbaru dan 13 dari histori lebih tua. B3 mengubah separuh contoh normal training menjadi sampel yang mengikuti distribusi kanal/kuartal fraud train. B4 menggabungkan B2+B3 dan hanya layak dicoba bila kedua ablasi mendukung.

Setiap run menyimpan `config.json`, `history.json`, `status.json`, checkpoint, prediksi selection/calibration/assessment, dan metrik per kanal/bulan serta kuota alert. Checkpoint dipilih pada selection; threshold FPR 0,1% dikalibrasi pada calibration; assessment dibaca terakhir. F2 dan FPR 0,05% juga dihitung sebagai operating point pembanding. Fold 90 hari maju berada sebelum test lama. Audit full-data menemukan Januari–Oktober 2017 tanpa fraud berlabel; konfigurasi awal mengunci satu origin yang berakhir 1 Januari 2017. Selection 6 April–4 Juli 2016, calibration 5 Juli–2 Oktober 2016, assessment 3 Oktober–31 Desember 2016. Ini perubahan terukur dari target awal tiga origin; satu fold tidak cukup untuk klaim peningkatan lintas periode. `fold_manifest.json` melaporkan dukungan fraud per role, sedangkan `temporal_support.json` menambahkan jumlah user, kartu, dan episode fraud. Data berlabel baru bisa diikutkan melalui `protocol.label_delay_days` dalam salinan konfigurasi dengan direktori hasil terpisah.

Cache fitur di-fit pada split awal 70% yang timestamp-nya utuh. Encoder tetap sama pada seluruh rolling fold agar perubahan B0 terutama berasal dari data training lebih baru. Semua fitur perilaku berbasis observasi masa lalu tanpa label. Histori graf B memotong transaksi pada awal hari query; training juga memakai aturan itu. Memmap fitur diurutkan menurut user agar preprocessing dapat memakai satu group user sekaligus; map ID kronologis mengembalikan urutan saat training/evaluasi. Lima fitur `merchant_channel` dihitung pada lintasan kedua menurut merchant dan kanal sehingga tetap mencakup transaksi seluruh user.

Prefix dengan `--max-rows` hanya memeriksa wiring; ia tidak menggantikan evaluasi full-data. Test 2018–2020 telah dilihat selama pengembangan sehingga hasil A0/A1 tetap eksploratif. Rencana eksperimen dan kriteria promosi ada di [PLAN.md](PLAN.md); hasil validasi yang benar-benar telah dijalankan dicatat di `VALIDATION.md`.
