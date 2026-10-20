# Panduan Eksekusi Lokal EXP25 — Standalone GraphSAGE (Tanpa XGBoost)

Panduan ini menjelaskan langkah-langkah menjalankan **EXP25 (`exp25_standalone_graphsage`)** pada PC lokal Windows 11 (NVIDIA RTX 5060 Ti 16GB VRAM, Anaconda environment `mhp_exp24_gpu`).

---

## 1. Persiapan Environment

Gunakan environment Conda yang sudah terpasang (`mhp_exp24_gpu`):
```cmd
conda activate mhp_exp24_gpu
cd D:\Hendika\TA-KlasifikasiTransaksiKKredit-NeighborSampling-GraphSAGE\code\exp25_standalone_graphsage
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
[SUKSES] Seluruh 8 Correctness Gates EXP25 LULUS.
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

### A. Menjalankan 1 Seed Penuh (Ketiga Metode: Uniform, Topology, Importance)
```cmd
python run_seed.py --config config.reference.yaml --seeds 42 --device cuda
```

### B. Menjalankan Seluruh 5 Seed Berpasangan (15 Run Total)
```cmd
python run_seed.py --config config.reference.yaml --seeds 42 100 2026 3407 8713 --device cuda
```

---

## 5. Lokasi Output & Log

Seluruh artefak EXP25 tersimpan secara terpisah dari EXP24 di dalam:
- **Folder Hasil Utama**: `..\..\result\exp25_standalone_graphsage\`
  - `runs_index.csv`: Tabel indeks ringkasan seluruh run yang pernah dijalankan.
  - `logs\<run_id>.log`: Berkas log terminal lengkap untuk setiap run.
  - `runs\<run_id>\`: Direktori terisolasi berisi konfigurasi, CSV history per-epoch, JSON hasil akhir, dan NPZ prediksi untuk masing-masing run.
  - `seed_<seed>\`: Direktori kanonikal per-seed untuk agregasi multi-seed.
  - `exp25_multiseed_per_run_metrics.csv`: Hasil seluruh run berdampingan.
  - `exp25_multiseed_summary.csv`: Rata-rata, standar deviasi, dan median antar-seed.
  - `exp25_paired_seed_deltas.csv`: Selisih berpasangan (*paired delta*) antar metode sampling.
  - `plots\exp25_paired_auprc_f1.png`: Grafik perbandingan AUPRC dan F1 antar-seed.
- **Folder Model Checkpoint**: `..\..\model\exp25_standalone_graphsage\`
