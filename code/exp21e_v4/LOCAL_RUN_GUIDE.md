# Panduan Lokal EXP21E-v4

Jalankan perintah dari folder `code/exp21e_v4`. Dataset lokal harus tersedia di
`../../dataset/credit_card_transactions-ibm_v2.csv`, sesuai `config.reference.yaml`.

## Validasi no-future

```powershell
python causal_validation.py
```

## Smoke satu strategi

```powershell
python main.py --config config.reference.yaml --sampling uniform --seed 42 --smoke
```

## Smoke seluruh strategi

```powershell
python run_all.py --config config.reference.yaml --skip-install --smoke
```

## Pilot lengkap tiga strategi

```powershell
python run_all.py --config config.reference.yaml --skip-install
```

Pilot lengkap memakai 20 epoch dan seluruh 30 hari validation serta 60 hari test. V4 tetap
berstatus pilot dan tidak digunakan sebagai tabel hasil final paper.

## Membuat notebook Kaggle

```powershell
python create_nb.py
```

Notebook yang dihasilkan adalah `exp21e_v4_causal_temporal_pilot.ipynb` dan metadata push ada
di `kernel-metadata.json`.
