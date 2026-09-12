# Validasi EXP13

Status: 12 September 2026.

## Selesai

- `kaggle.json` valid dan memiliki `username`/`key`, tanpa mencetak nilainya.
- Credential diabaikan oleh Git pada level root dan folder EXP13.
- Kaggle CLI 2.2.4 terpasang pada virtual environment proyek.
- Autentikasi read-only dan pemeriksaan kuota berhasil; GPU tersisa 30 jam saat
  deployment pertama dilakukan.
- Empat test deployment lulus: konfigurasi engine, exact accelerator ID,
  fail-closed T4 check, dan zero-secret bundle.
- Generated `kernel.py` lolos parse AST; metadata JSON valid.
- Pemindaian memastikan API key tidak berada di `kernel.py` maupun
  `kernel-metadata.json`.
- Kernel privat uniform seed 42 berhasil di-push sebagai version 1 dengan
  `NvidiaTeslaT4` dan mulai berjalan.

## Menunggu hasil remote

- Verifikasi log runtime `EXP13 accelerator verified: Tesla T4`.
- Preprocessing full 24.386.900 baris pada RAM/disk aktual Kaggle.
- Epoch, early stopping, checkpoint, test metrics, dan unduhan output.
- Setelah baseline sehat: topology dan importance, lalu multi-seed.

## Test command

```powershell
& .\code\exp6_gpu_tqdm\.venv\Scripts\python.exe -B -m unittest discover `
  -s .\code\exp13_try_kaggle\tests `
  -t .\code\exp13_try_kaggle -v
```
