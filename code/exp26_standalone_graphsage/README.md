# EXP26 — Standalone GraphSAGE with Proposal-Aligned Neighbor Sampling

Eksperimen ini mengimplementasikan arsitektur **Standalone End-to-End Heterogeneous GraphSAGE (Linear Layer + Sigmoid, TANPA XGBoost)** yang 100% konsisten dengan **Proposal Tugas Akhir (Bab III Subbab 3.4, 3.5, 3.6, dan 3.7)**.

Dengan menghapus downstream classifier XGBoost (yang pada EXP24 mendominasi keputusan prediksi melalui fitur tabular dan menutupi perbedaan antar-sampler), EXP26 mengisolasi pengaruh murni dari ketiga strategi *neighbor sampling*:
1. **Uniform Random Sampling** (Baseline — Algoritma 2 Proposal)
2. **Topology-Aware Sampling** (Jaccard + Attribute Homophily ASA-GNN — Algoritma 3 Proposal)
3. **Importance-Based Sampling** (Degree Centrality + Local PPR — Algoritma 4 Proposal)

---

## 1. Perbedaan Utama EXP26 vs EXP24

| Komponen | EXP24 (`exp24_proposal_sampler`) | EXP26 (`exp26_standalone_graphsage`) |
| :--- | :--- | :--- |
| **Classifier Akhir** | Hybrid (10-d Tabular + 128-d GNN) $\rightarrow$ **XGBoost** | **Standalone GraphSAGE** (128-d GNN $\rightarrow$ Linear Layer $\rightarrow$ Sigmoid) |
| **Kesesuaian Bab 3.4 Proposal** | Modifikasi Hibrida | **100% Sesuai Gambar 3.3 & Persamaan 3.1–3.2 Proposal** |
| **Isolasi Efek Sampler** | Sebagian tertutup (*masked*) oleh pohon keputusan XGBoost | **100% Murni** mencerminkan kualitas agregasi subgraf hasil *sampling* |
| **Waktu Eksekusi per Run** | ~1.500–2.400 detik (karena ekstraksi 1,4 juta embedding train untuk XGBoost) | **~40% Lebih Cepat** (langsung evaluasi GNN pada Val & Test) |

---

## 2. Formulasi Sampling EXP26 (Identik Proposal & EXP24)

### A. Uniform Random Sampling (`uniform`)
- $w(v, u) = 1.0$ pada seluruh edge.

### B. Topology-Aware Sampling (`topology`)
- Menggabungkan **Bipartite Jaccard Similarity** $J(v, u)$ dan **Attribute Homophily ASA-GNN** $H_{\text{ASA-GNN}}(v, u)$:
  $$w_{\text{raw}}(v, u) = 0.5 \cdot J(v, u) + 0.5 \cdot H_{\text{ASA-GNN}}(v, u)$$
- $w(v, u) = \operatorname{MinMax}(w_{\text{raw}}) + 0.1 \in [0.1, 1.1]$.

### C. Importance-Based Sampling (`importance`)
- Menggabungkan **Degree Centrality** $C_d(u)$ dan **Personalized PageRank** $C_p(u)$ ($r=3, \beta=0.15$):
  $$w_{\text{raw}}(v, u) = 0.3 \cdot C_d(u) + 0.7 \cdot C_p(u)$$
- $w(v, u) = \operatorname{MinMax}(w_{\text{raw}}) + 0.1 \in [0.1, 1.1]$.

---

## 3. Cara Menjalankan

### A. Verifikasi Correctness Gates (Tanpa Training)
```cmd
python validate_sampler.py
```

### B. Smoke Test (300d Train / 30d Val / 8d Gap / 60d Test, 180k Baris)
```cmd
python run_seed.py --config config.smoke.yaml --seeds 42 --device cuda
```

### C. Eksekusi Penuh Multi-Fanout & Multi-Seed

1. **Uji 1 Metode (misal Uniform) dengan 3 Fan-Out Hop ([5, 5], [15, 10], [25, 10]) pada Seed 42**:
   ```cmd
   python run_seed.py --methods uniform --fanouts all --seeds 42 --device cuda
   ```

2. **Uji 3 Metode Sampling dengan 3 Fan-Out Hop pada Seed 42**:
   ```cmd
   python run_seed.py --methods all --fanouts all --seeds 42 --device cuda
   ```

3. **Uji 5 Seed Berpasangan dengan Default Fan-Out [25, 10] (15 Run)**:
   ```cmd
   python run_seed.py --seeds 42 100 2026 3407 8713 --device cuda
   ```

4. **Full Multi-Fanout & Multi-Seed Sweep (3 Metode × 3 Fan-Out × 5 Seed = 45 Run)**:
   ```cmd
   python run_seed.py --fanouts all --device cuda
   ```

### D. Single Run Manual
```cmd
python main.py --config config.reference.yaml --sampling topology --seed 42 --num-neighbors 5 5 --device cuda
```
