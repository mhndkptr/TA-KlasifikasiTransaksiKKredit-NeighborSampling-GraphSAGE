# Grafik perbandingan eksperimen

Buka **[perbandingan_experiment.ipynb](perbandingan_experiment.ipynb)** di VS Code/Jupyter,
pilih kernel Python, lalu **Run All**. Notebook membaca hasil EXP1–EXP11 dari `../result/`
secara otomatis, termasuk hasil baru yang ditambahkan kemudian.

Laporan yang sudah dihasilkan ada di **[output/comparison_report.html](output/comparison_report.html)**.
Buka file itu di browser untuk langsung melihat grafik tanpa menjalankan Python.
Laporan bersifat offline dan tidak membutuhkan CDN atau internet.

## Instalasi dan menjalankan ulang

Python 3.10+; perintah dijalankan dari root repository. Tidak perlu GPU, dataset,
PyTorch, atau checkpoint model. Environment notebook terpisah dari training.

```powershell
python -m venv notebooks/.venv
./notebooks/.venv/Scripts/python.exe -m pip install -r notebooks/requirements.txt
./notebooks/.venv/Scripts/python.exe notebooks/compare_results.py
```

Pada mesin ini environment `notebooks/.venv` sudah disiapkan. Untuk notebook di VS Code,
gunakan **Select Kernel → Python Environments → notebooks/.venv/Scripts/python.exe**.
Jika memakai environment sendiri, jalankan `python -m pip install -r notebooks/requirements.txt`,
lalu `python notebooks/compare_results.py`. Di Linux/macOS executable venv ada di
`notebooks/.venv/bin/python`.

Contoh ekspor hanya EXP10–EXP11:

```powershell
./notebooks/.venv/Scripts/python.exe notebooks/compare_results.py --experiments exp10 exp11 --kinds main --output-dir notebooks/output/exp10_exp11
```

`--result-dir` dapat menunjuk folder hasil lain. Semua default path ditentukan dari
lokasi script, sehingga CLI tidak bergantung pada working directory. Output berupa
hasil turunan; JSON asli tidak diubah.

## Grafik yang tersedia

- **Antar eksperimen dan metode:** mean ± SD sampel per konfigurasi/metode, dengan titik setiap seed.
- **Setiap run per eksperimen/metode:** semua seed dan attempt, tanpa dirata-ratakan. Pilih metode tertentu untuk membandingkan run metode tersebut.
- **History:** semua metrik numerik per epoch, satu kurva untuk setiap run, dan tanda best epoch jika tercatat.
- **Metrik utama dan tambahan:** seluruh angka pada `metrics`, threshold alternatif, channel pembayaran, temporal bins, timing, statistik prediksi, dan diagnostik lainnya ditemukan otomatis.
- **Ekspor grafik:** tombol SVG serta PNG pada toolbar Plotly; tabel CSV tersedia untuk pengolahan berikutnya.

Pada HTML, pilih tampilan, eksperimen, metode, konfigurasi, dan metrik. Tombol
**Sebelumnya/Berikutnya** menelusuri semua metrik pada filter tersebut. Centang
**Pilot** atau **Smoke** untuk memasukkannya. Tabel menampilkan angka tepat dari sumber.

Sel konfigurasi notebook menyediakan `EXPERIMENTS`, `METHODS`, `KINDS`,
`COMPARISON_IDS`, `FOCUS_EXPERIMENT`, `RUN_METRIC`, dan `HISTORY_METRICS`.
`METRICS = None` menampilkan semua metrik utama sebagai grafik notebook;
`HISTORY_METRICS = None` menampilkan semua kurva pada eksperimen fokus.
Default menampilkan enam metrik umum agar notebook tidak terlalu panjang.
Semua metrik selalu tersedia dalam laporan HTML, tanpa harus mengubah default ini.

## Pembacaan dan agregasi

- JSON per run adalah sumber angka; `summary.csv` sumber tidak dihitung sebagai run.
- `history.json` dipakai bila tersedia, selain itu history di dalam JSON hasil. Keduanya tidak digabung sehingga epoch tidak terduplikasi.
- Run dengan status selain `complete` dilewati. Format lama tanpa status diterima bila memiliki objek `metrics` dan ditandai `metrics_available`.
- Semua attempt muncul dalam grafik setiap run. Agregasi memakai attempt terbaru per `(experiment, kind, comparison_id, strategy, seed)` berdasarkan mtime `metrics.json`, sama dengan pipeline ringkasan. Penyalinan file yang mengubah mtime dapat memengaruhi pilihan ini; inventaris menunjukkan run terpilih.
- Mean/SD tidak mencampur ID konfigurasi, pilot, dan smoke. `n=1` memiliki SD kosong, bukan nol. Nilai null/tidak dicatat tetap N/A dan dikeluarkan dari hitungan nilai valid.
- String dan boolean bukan skor numerik. Katalog dan path indeks mempertahankan identitas angka nested; `[0]` berarti elemen pertama.
- EXP lama tanpa `comparison_id` menggunakan identitas `legacy-...` dari metadata yang tersedia. Kesetaraan konfigurasi lengkap tidak bisa diverifikasi dari artefak lama.

Ukuran test, fitur, rentang waktu, pemilihan checkpoint, threshold, serta cakupan
latency dapat berbeda antar eksperimen. Perbandingan lintas eksperimen bersifat
deskriptif. Untuk menilai metode pada protokol yang sama, pilih satu `comparison_id`.
Jangan menafsirkan `selection_score` dengan `selection_metric` berbeda pada skala yang
sama. Indeks temporal bin yang sama belum tentu mewakili rentang tanggal yang sama.

## File output

| File | Isi |
|---|---|
| `comparison_report.html` | Grafik interaktif mandiri, semua metrik dan filter |
| `runs.csv` | Inventaris run, metadata protokol, dan pilihan agregasi |
| `metrics_long.csv` | Satu baris per run/metrik beserta metadata sumber |
| `metrics_wide.csv` | Satu baris per run, satu kolom per metrik |
| `history_long.csv` | Metrik tiap epoch setiap run |
| `summary.csv` | Count, mean, SD, minimum, maksimum per konfigurasi/metode/metrik |
| `metric_catalog.csv` | Daftar metrik dan jumlah nilai tersedia |
| `load_issues.csv` | File rusak, run belum selesai, atau history yang dilewati |

Output dan virtual environment diabaikan Git; jalankan ulang untuk membuat hasil
terbaru. Untuk pemeriksaan logika loader/agregasi:

```powershell
./notebooks/.venv/Scripts/python.exe -m unittest discover -s notebooks -p test_compare_results.py -v
```
