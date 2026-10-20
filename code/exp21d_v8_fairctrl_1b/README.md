# EXP21D-v8 Fair Control 1B — Equal-Dimension PCA

Implementasi Eksperimen 1B membandingkan tiga input XGBoost dengan protokol evaluasi yang sama:

1. 10-d clean tabular;
2. 10-d PCA dari embedding GraphSAGE 128-d;
3. hybrid 20-d (10-d tabular + 10-d GNN-PCA).

PCA di-fit secara eksklusif pada split train. Validation hanya dipakai untuk memilih threshold, sedangkan test dievaluasi satu kali setelah threshold dikunci. Block permutation importance pada test mengacak seluruh blok tabular atau GNN-PCA secara bersama agar korelasi internal blok tetap terjaga.

## Prasyarat

Eksperimen membaca artefak lengkap EXP21D-v5 Sampler Correctness untuk `uniform`, `topology`, dan `importance`. Setiap sampler harus memiliki matriks hybrid 138-d dan label untuk train, validation, dan test.

Konfigurasi lokal sudah menunjuk ke hasil Kaggle EXP 2A yang telah diunduh:

```powershell
python validate_implementation.py
python main.py --config config.reference.yaml
```

Notebook dapat dibuat ulang dengan:

```powershell
python create_notebook.py
```

## Artefak untuk analisis ulang

Setiap run menyimpan model PCA, komponen dan explained variance PCA, model XGBoost, fitur GNN-PCA 10-d dan hybrid 20-d, label, probabilitas mentah validation/test, threshold terkalibrasi, history XGBoost, repeat-level permutation importance, tabel ringkasan, log append-only, snapshot kode, serta visualisasi PNG/PDF 300 DPI.

Probabilitas dan label mentah memungkinkan metrik atau threshold baru dihitung tanpa training ulang. Fitur hasil PCA dan model tersimpan memungkinkan benchmark classifier tambahan tanpa mem-fit PCA kembali.

Folder ini sengaja tidak memiliki `kernel-metadata.json` dan tidak dipush ke Kaggle.
