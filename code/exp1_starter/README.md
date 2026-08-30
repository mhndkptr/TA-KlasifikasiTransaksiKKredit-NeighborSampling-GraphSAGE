# Experiment 1 - GraphSAGE Neighbor Sampling

Implementasi ini mengikuti Bab III proposal: split temporal 70/15/15, graf
User-Transaction-Merchant, GraphSAGE dua lapis, fan-out 25/10, serta tiga
strategi sampling. `max_rows` sengaja bernilai 200.000 agar instalasi dapat
diuji lebih dulu; ubah menjadi `null` untuk eksperimen penuh.

## Menjalankan

```powershell
cd code/exp1_starter
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
python run.py --config config.yaml
```

Untuk satu strategi/seed saat smoke test:

```powershell
python run.py --config config.yaml --strategy uniform --seed 42 --max-rows 10000
```

Artefak graph cache disimpan di `model/`, checkpoint terbaik per eksperimen
di `model/exp1_<strategy>_seed<seed>.pt`, metrik tiap run dalam JSON di
`result/`, dan ringkasan semua run dalam `result/exp1_summary.csv`.

## Keputusan implementasi

- Tabel 3.1 (70/15/15) dipakai sebagai sumber pembagian data. Algoritma 5
  dalam proposal menyebut 80/10/10; perbedaan ini dicatat di metadata hasil.
- Label validasi/uji tidak pernah dipakai untuk menghitung bobot sampling.
- `topology` menggabungkan Jaccard lingkungan dan homofili lokal berbasis
  label training saja. Jika label target tidak tersedia, komponen homofili
  dinetralkan sehingga tidak terjadi leakage. Karena endpoint langsung pada
  graf bipartit berbeda tipe membuat Jaccard satu-hop selalu nol, reciprocal
  degree dipakai sebagai fallback kekuatan struktural dan dicatat di kode.
- `importance` menggabungkan degree centrality dan personalized PageRank
  lokal terbatas sesuai parameter konfigurasi.
- HeteroData disimpan untuk audit. Model memakai representasi homogen dari
  graf yang sama agar ketiga sampler dapat dibandingkan dengan satu
  arsitektur GraphSAGE identik.

Eksperimen penuh membutuhkan RAM/GPU yang memadai. Mulailah dari smoke test,
kemudian naikkan `max_rows` secara bertahap sebelum memakai `null`.
