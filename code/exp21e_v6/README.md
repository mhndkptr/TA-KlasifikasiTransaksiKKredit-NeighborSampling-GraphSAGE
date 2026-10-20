# EXP21E-v6 — Rolling-Origin Robustness

V6 menguji robustness hasil V5 terhadap perubahan periode waktu tanpa melakukan retuning.

- Sampler: `uniform`, `topology`, dan `importance`.
- Ketiga sampler dipertahankan karena selisih validation V5 kecil dan tidak konsisten antarseed.
- Seed: `42`, `100`, dan `2026`.
- Window per eksekusi: satu dari dua anchor yang telah dipilih deterministik pada
  `primary ± 180 hari`.
- Matriks per eksekusi: `3 sampler × 1 window × 3 seed = 9 GNN runs`.
- Protokol, arsitektur, loss, hyperparameter, causal evaluation, checkpoint selection, dan
  threshold calibration dibekukan dari V5.

`window_selection.py` membuat `fold_manifest.json` sebelum training. Pemilihan window tidak
menggunakan metrik model atau performa test.

Konfigurasi Kaggle aktif sekarang adalah window kedua, `rolling_plus_180d`. Untuk kembali ke
window pertama, ubah hanya:

```yaml
robustness:
  active_window_id: "rolling_minus_180d"
```

Jangan mengubah seed, strategi, hyperparameter, atau field protokol lainnya.
