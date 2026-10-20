# EXP21D-v5 Sampler Correctness (Eksperimen 2A)

Implementasi koreksi sampler aktif untuk pipeline Hybrid HeteroGraphSAGE + XGBoost dengan konfigurasi terkunci `[15, 10]` dan window temporal `300/30/8/60`.

Folder ini **belum memiliki `kernel-metadata.json` dan tidak dipush ke Kaggle**. Notebook disiapkan untuk digunakan setelah slot session tersedia.

## Kontrak matematis sampler

NeighborLoader PyG mengambil incoming edges. Dari root transaksi:

1. Hop pertama memilih satu user dan satu merchant melalui relasi `user -> transaction` dan `merchant -> transaction`; tidak ada kompetisi, sehingga bobotnya tetap satu.
2. Hop kedua memilih kandidat transaksi yang masuk ke user/merchant:
   - `transaction -> user` menggunakan **degree merchant pasangan transaksi kandidat**;
   - `transaction -> merchant` menggunakan **degree user pasangan transaksi kandidat**.

Untuk kandidat dengan counterpart degree `d`:

- Uniform: `w = 1`;
- Topology: `raw = 1 / log(1 + d)`;
- Importance: `raw = log(1 + d)`;
- stabilisasi: `w = minmax(raw) + 0.1`.

Bobot disimpan sebagai `sampling_weight` dan diteruskan secara eksplisit melalui `NeighborLoader(weight_attr="sampling_weight")`.

## Correctness gate

`validate_sampler.py` memeriksa:

- mapping counterpart degree pada kedua reverse relation;
- arah monotonicity Topology dan Importance;
- Uniform selalu berbobot satu;
- relasi first-hop tetap uniform;
- label fraud tidak memengaruhi bobot;
- transaksi dengan counterpart ganda ditolak;
- loader menerima `weight_attr="sampling_weight"`;
- seluruh bobot finite, positif, dan memiliki local sum positif.

## Menjalankan secara lokal

Validasi tanpa training:

```powershell
python validate_sampler.py
```

Single sampler:

```powershell
python main.py --config config.reference.yaml --sampling topology --seed 42
```

Matriks single-seed tiga sampler:

```powershell
python run_all.py --config config.reference.yaml --seed 42
```

Notebook yang telah disiapkan: `exp21d_v5_sampler_correctness.ipynb`.

## Artefak

Output berada di `result/exp21d_v5_sampler_correctness/seed_42/`:

- `*_sampler_weight_audit.json`: distribusi bobot dan counterpart degree setiap relation;
- `events.jsonl`: log append-only per run dan epoch;
- `*_history.json`: histori training GNN;
- `*_xgboost_history.json`: histori boosting hybrid/tabular/embedding;
- `*_predictions.npz`: probabilitas validation/test ketiga subset fitur;
- `features/<sampler>/*.npy`: matriks tabular + embedding 128-d yang dapat dibuka dengan memory mapping;
- checkpoint GNN serta tiga model XGBoost;
- tabel perbandingan, report Markdown, dan visualisasi 300 DPI.

Eksperimen ini hanya membuktikan dan menjalankan mekanisme sampler yang telah dikoreksi. Audit empiris overlap edge, kuartil degree hasil sampling, dan homofili merupakan cakupan Eksperimen 2B.
