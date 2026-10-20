# EXP24 — Proposal-Aligned Neighbor Sampling for Heterogeneous GraphSAGE

Eksperimen ini mengimplementasikan strategi *neighbor sampling* yang selaras verbatim dengan **Proposal Tugas Akhir (Bab III Subbab 3.5.2 & 3.5.3, Algoritma 3 & Algoritma 4)** dengan resolusi anti-leakage berbasis **ASA-GNN (Tian et al., IEEE TCSS 2024)** dan **Local Personalized PageRank (PPR)**.

Pipeline ini dibangun di atas fondasi tervalidasi **EXP21D-v10**, mengunci arsitektur Hybrid HeteroGraphSAGE + XGBoost, window temporal `300/30/8/60`, fan-out `[15, 10]`, dan 5 seed berpasangan (`42`, `100`, `2026`, `3407`, `8713`).

---

## 1. Formulasi Sampling EXP24

### A. Uniform Random Sampling (Baseline)
- $w(v, u) = 1.0$ pada seluruh edge.

### B. Topology-Aware Sampling (Algoritma 3 Proposal + ASA-GNN)
- Menggabungkan **Bipartite Jaccard Similarity** $J(v, u)$ dan **Attribute Homophily ASA-GNN** $H_{\text{ASA-GNN}}(v, u)$:
  $$w(v, u) = 0.5 \cdot J(v, u) + 0.5 \cdot H_{\text{ASA-GNN}}(v, u)$$
- $H_{\text{ASA-GNN}}(v, u) = \frac{1 + \cos(x_v, x_{\text{centroid}})}{2} \in [0, 1]$.
- **100% Bebas Label Leakage**: Menggantikan label target $L[v]$ yang belum diketahui saat data uji dengan keselarasan atribut perilaku transaksi.

### C. Importance-Based Sampling (Algoritma 4 Proposal + Local PPR)
- Menggabungkan **Degree Centrality** $C_d(u)$ dan **Personalized PageRank** $C_p(u)$:
  $$w(v, u) = 0.3 \cdot C_d(u) + 0.7 \cdot C_p(u)$$
- $C_p(u)$ dihitung secara lokal/bipartit via power iteration ($r=3, \beta=0.15$) dengan sparse matrix ops di GPU (< 10 ms per batch, bebas OOM).

---

## 2. Telemetri Granular Per-Epoch

Setiap epoch mencatat telemetri komprehensif ke `training_history.csv` dan `history.json`:
- **Metrik Model**: `train_loss`, `train_acc`, `val_loss`, `val_auprc` (early stopping metric), `val_roc_auc`, `val_f1_default`, `val_f1_optimal`, `val_precision_optimal`, `val_recall_optimal`, `val_optimal_threshold`, `val_fp`, `val_fn`.
- **Waktu & Latensi**: `time_train_epoch_sec`, `time_val_epoch_sec`, `time_epoch_total_sec`, `cumulative_runtime_sec`, `val_latency_ms_per_1k_tx`.
- **Sumber Daya Hardware**: `peak_gpu_vram_mb`, `peak_ram_mb`.

---

## 3. Cara Menjalankan

Panduan lengkap instalasi environment dan eksekusi pada PC Windows 11 lokal (RTX 5060 Ti 16GB, Core i7, 32GB RAM) dapat dibaca di:
📄 **[PANDUAN_RUN_LOCAL.md](PANDUAN_RUN_LOCAL.md)**

### A. Verifikasi Correctness Gates (Tanpa Training)
```powershell
python validate_sampler.py
```

### B. Single Run / Smoke Test Lokal
```powershell
python main.py --config config.reference.yaml --sampling topology --seed 42 --nrows 50000
```

### C. Eksekusi Multi-Seed Paired (5 Seed)
```powershell
python run_seed.py --config config.reference.yaml --seeds 42 100 2026
python run_seed.py --config config.reference.yaml --seeds 3407 8713
```

### D. Generate Notebook untuk Kaggle GPU
```powershell
python create_notebook.py
```
Notebook `exp24_proposal_sampler.ipynb` akan dihasilkan dan siap diunggah ke sesi Kaggle GPU.
