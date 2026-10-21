# Panduan Eksekusi Lokal EXP26 — Standalone GraphSAGE (Tanpa XGBoost)

Panduan ini menjelaskan langkah-langkah menjalankan **EXP26 (`exp26_standalone_graphsage`)** pada PC lokal Windows 11 (NVIDIA RTX 5060 Ti 16GB VRAM, Anaconda environment `mhp_exp24_gpu`).

---

## 1. Persiapan Environment

Gunakan environment Conda yang sudah terpasang (`mhp_exp24_gpu`):
```cmd
conda activate mhp_exp24_gpu
cd D:\Hendika\TA-KlasifikasiTransaksiKKredit-NeighborSampling-GraphSAGE\code\exp26_standalone_graphsage
```

---

## 2. Langkah 1: Verifikasi 8 Correctness Gates Sampler

Pastikan seluruh pemeriksaan integritas sampler lulus:
```cmd
python validate_sampler.py
```
Output yang diharapkan:
```json
{
  "status": "passed",
  "checks": {
    "uniform_weights_are_strictly_one": true,
    "first_hop_remains_uniform": true,
    "zero_label_leakage_topology": true,
    "zero_label_leakage_importance": true,
    "topology_weights_bounded": true,
    "importance_weights_bounded": true,
    "neighbor_loader_uses_weight_attr": true,
    "topology_audit_passed": true,
    "importance_audit_passed": true
  }
}
[SUKSES] Seluruh 8 Correctness Gates EXP26 LULUS.
```

---

## 3. Langkah 2: Smoke Test (Opsional, ~1 Menit per Seed)

Menguji pipeline end-to-end dengan jendela waktu penuh (`300/30/8/60` hari) pada `180.000` baris pertama:
```cmd
python run_seed.py --config config.smoke.yaml --seeds 42 --device cuda
```
Atau 5 seed smoke test sekaligus:
```cmd
python run_seed.py --config config.smoke.yaml --seeds 42 100 2026 3407 8713 --device cuda
```

---

## 4. Langkah 3: Eksekusi Eksperimen Utama (Full Dataset)

### A. Menjalankan 1 Metode (misal Uniform) dengan 3 Fan-Out Hop ([5, 5], [15, 10], [25, 10]) pada Seed 42:
```cmd
python run_seed.py --methods uniform --fanouts all --seeds 42 --device cuda
```
*(Atau tentukan fan-out spesifik: `--fanouts 5,5 15,10 25,10`)*

### B. Menjalankan Ketiga Metode Sampling dengan 3 Fan-Out Hop pada Seed Tertentu (misal Seed 42):
```cmd
python run_seed.py --methods all --fanouts all --seeds 42 --device cuda
```

### C. Menjalankan Seluruh 5 Seed Berpasangan dengan Default Fan-Out [25, 10] (15 Run Total):
```cmd
python run_seed.py --seeds 42 100 2026 3407 8713 --device cuda
```

### D. Menjalankan Full Multi-Fanout & Multi-Seed Sweep (3 Metode × 3 Fan-Out × 5 Seed = 45 Run):
```cmd
python run_seed.py --fanouts all --device cuda
```

### E. Mengubah Pangkat Bobot Sampling (Power Exponent p)
Secara default $p = 1.0$ (skala linier proposal). Jika ingin menaikkan ketajaman pembobotan (misal $p = 2.0$):
- **Lewat CLI:**
  ```cmd
  python run_seed.py --methods topology importance --fanouts 5,5 --power-exponent 2.0 --seeds 42 --device cuda
  ```
- **Atau lewat `config.reference.yaml`:**
  Ubah baris `power_exponent: 1.0` menjadi `power_exponent: 2.0`.

### F. Menjalankan Agregasi & Visualisasi Mandiri Kapan Saja:
```cmd
python aggregate_exp26.py
```

---

## 5. Lokasi Output & Log

Seluruh artefak EXP26 tersimpan secara terpisah dari EXP24 di dalam:
- **Folder Hasil Utama**: `..\..\result\exp26_standalone_graphsage\`
  - `runs_index.csv`: Tabel indeks ringkasan seluruh run yang pernah dijalankan (mencakup kolom `fanout`, `k1`, `k2`, metrik MAP, NDCG, Accuracy, dll).
  - `logs\<run_id>.log`: Berkas log terminal lengkap untuk setiap run.
  - `runs\<run_id>\`: Direktori terisolasi berisi konfigurasi, CSV history per-epoch, JSON hasil akhir, dan NPZ prediksi untuk masing-masing run.
  - `seed_<seed>\`: Direktori kanonikal per-seed untuk agregasi multi-seed.
  - `exp26_fanout_comparison.csv`: Tabel perbandingan metrik seluruh run berdasarkan variasi fan-out hop.
  - `exp26_fanout_summary.csv`: Ringkasan mean, standar deviasi, dan median untuk setiap kombinasi `(method, fanout)`.
  - `exp26_multiseed_per_run_metrics.csv`: Hasil seluruh run berdampingan.
  - `exp26_multiseed_summary.csv`: Rata-rata, standar deviasi, dan median antar-seed.
  - `exp26_paired_seed_deltas.csv`: Selisih berpasangan (*paired delta*) antar metode sampling.
  - `plots\exp26_fanout_sensitivity.png`: Grafik komparasi sensitivitas fan-out hop ([5, 5], [15, 10], [25, 10]) terhadap AUPRC, F1-Score, dan Latensi.
  - `plots\exp26_paired_auprc_f1.png`: Grafik perbandingan AUPRC dan F1 antar-seed.
- **Folder Model Checkpoint**: `..\..\model\exp26_standalone_graphsage\`
