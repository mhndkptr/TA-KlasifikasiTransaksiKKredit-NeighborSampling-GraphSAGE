# EXP21D-v6 Fair Control 1C — Frozen Random GraphSAGE

Implementasi kontrol negatif untuk menguji apakah kenaikan performa berasal dari representasi GraphSAGE yang dipelajari atau sekadar proyeksi acak dan agregasi struktur graf.

Folder ini sengaja **tidak memiliki `kernel-metadata.json` dan belum dipush atau dijalankan di Kaggle**.

## Kontrak eksperimen

- split temporal terkunci `300/30/8/60`;
- fan-out terkunci `[15, 10]`;
- satu checkpoint GraphSAGE random-init per seed digunakan ulang oleh Uniform, Topology, dan Importance;
- parameter encoder dibekukan, mode selalu `eval`, jumlah training step nol, tanpa optimizer dan backward;
- label tidak dimasukkan ke `HeteroData`; label hanya dipakai oleh XGBoost dan evaluasi;
- sampler menggunakan koreksi counterpart degree dari EXP 2A dan `weight_attr=sampling_weight` aktif;
- tiga representasi downstream: tabular 10-d, frozen GraphSAGE 128-d, dan hybrid 138-d.

Identitas checkpoint dibuktikan melalui `checkpoint_sha256` di `encoder_provenance.json`. `run_all.py` menghentikan agregasi bila hash antar-sampler berbeda.

## Menjalankan

Validasi kontrak tanpa training dataset penuh:

```powershell
python validate_frozen_control.py
```

Satu sampler:

```powershell
python main.py --config config.reference.yaml --sampling uniform --seed 42
```

Ketiga sampler secara berurutan:

```powershell
python run_all.py --config config.reference.yaml --seed 42
```

Notebook Kaggle lokal tersedia sebagai `exp21d_v6_fairctrl_1c.ipynb`. Notebook menjalankan sampler secara berurutan dalam satu session agar satu checkpoint encoder dipakai bersama.

## Artefak untuk analisis ulang

Setiap run menyimpan:

- `events.jsonl`, `manifest.json`, `resolved_config.yaml`, snapshot source, dan environment;
- `encoder_provenance.json`, checkpoint hash, serta history eksplisit dengan nol training step;
- audit bobot sampler setiap split;
- matriks `.npy` tabular, frozen embedding, hybrid, label, dan transaction ID;
- probabilitas validation/test di `predictions.npz`;
- tiga model XGBoost JSON dan seluruh threshold curve/metrics;
- indeks artefak lengkap.

Agregasi tiga sampler membuat tabel CSV, source data gambar, dan empat visualisasi 300 DPI dalam PNG dan PDF: perbandingan AUPRC/F1, kurva precision-recall, confusion matrix, dan PCA embedding beku. Data tersebut memungkinkan metrik atau visualisasi tambahan dihitung tanpa mengulang ekstraksi GraphSAGE maupun training XGBoost.
