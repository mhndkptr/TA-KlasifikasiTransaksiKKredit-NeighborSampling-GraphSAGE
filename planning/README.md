# Master Roadmap Eksperimen Tugas Akhir: Optimasi GraphSAGE & Neighbor Sampling

Dokumen ini adalah ringkasan induk (*master index*) dari perencanaan implementasi bertahap untuk merealisasikan seluruh rekomendasi perbaikan berbasis temuan paper **SHINE (IEEE TKDE 2024)** pada deteksi fraud transaksi kartu kredit (dataset IBM Synthetic / FRAUD II).

Seluruh perencanaan dipecah ke dalam **6 dokumen eksperimen terpisah** dengan prinsip *Ablation Study* bertahap agar kausalitas ilmiah setiap komponen dapat dipertanggungjawabkan secara kuat pada laporan Tugas Akhir.

---

## Daftar Berkas Perencanaan Eksperimen

| Urutan | Berkas Perencanaan | Kode ID | Fokus Utama & Kontribusi Ilmiah | Estimasi Waktu Komputasi |
| :---: | :--- | :---: | :--- | :---: |
| 1 | [01_exp20_threshold_calibration.md](file:///d:/Projects/Klasifikasi%20Transasi%20Kartu%20Kredit%20-%20GraphSAGE/tugas-akhir-klasifikasi-transaksi-kartu-kredit/planning/01_exp20_threshold_calibration.md) | `EXP20` | **Audit Metrik & Kalibrasi Threshold Dinamis**: Memvalidasi lonjakan F1-score dengan mencari ambang batas optimal ($\tau^*$) pada Validation Set tanpa melatih ulang GNN. | $\approx 2$ menit (Nol training) |
| 2 | [02_exp21_temporal_window_alignment.md](file:///d:/Projects/Klasifikasi%20Transasi%20Kartu%20Kredit%20-%20GraphSAGE/tugas-akhir-klasifikasi-transaksi-kartu-kredit/planning/02_exp21_temporal_window_alignment.md) | `EXP21` | **Penyelarasan Horizon Waktu (2016–2020)**: Mengeliminasi *30-year concept drift* dan risiko Kaggle *timeout* dengan mengadopsi protokol rolling window 300 hari train & 60 hari test persis seperti paper SHINE. | $\approx 25$ menit |
| 3 | [03_exp22_gnn_aggregator_max_pooling.md](file:///d:/Projects/Klasifikasi%20Transasi%20Kartu%20Kredit%20-%20GraphSAGE/tugas-akhir-klasifikasi-transaksi-kartu-kredit/planning/03_exp22_gnn_aggregator_max_pooling.md) | `EXP22` | **Inovasi Agregator GNN (Max vs Mean)**: Membuktikan bahwa agregator `max` mencegah penenggelaman sinyal anomali minoritas (*anti-dilution*) pada ketimpangan kelas ekstrem. | $\approx 25$ menit |
| 4 | [04_exp23_pure_graphsage_end_to_end.md](file:///d:/Projects/Klasifikasi%20Transasi%20Kartu%20Kredit%20-%20GraphSAGE/tugas-akhir-klasifikasi-transaksi-kartu-kredit/planning/04_exp23_pure_graphsage_end_to_end.md) | `EXP23` | **Pure End-to-End HeteroGraphSAGE (Ablasi Eliminasi XGBoost)**: Mengeliminasi classifier downstream XGBoost dan mengevaluasi klasifikasi sigmoid murni GraphSAGE pada protokol rolling window 2016–2020 untuk mengisolasi kausalitas neighbor sampling. | $\approx 15$ menit |
| 5 | [05_exp24_attribute_aware_sampling.md](file:///d:/Projects/Klasifikasi%20Transasi%20Kartu%20Kredit%20-%20GraphSAGE/tugas-akhir-klasifikasi-transaksi-kartu-kredit/planning/05_exp24_attribute_aware_sampling.md) | `EXP24` | **Penyempurnaan Neighbor Sampling (Core Thesis)**: Memperkaya metode sampling `topology` dan `importance` dengan bobot kedekatan waktu dan signifikansi nominal, serta membandingkannya dengan *Uniform* dan *NN Sampler* SHINE. | $\approx 45$ menit (4 sampler) |
| 6 | [06_exp25_integrated_hybrid_ablation.md](file:///d:/Projects/Klasifikasi%20Transasi%20Kartu%20Kredit%20-%20GraphSAGE/tugas-akhir-klasifikasi-transaksi-kartu-kredit/planning/06_exp25_integrated_hybrid_ablation.md) | `EXP25` | **Integrasi Pipeline Hibrida Final & Studi Ablasi**: Menyusun matriks komparasi master (Tabular vs GNN vs Hybrid) untuk tabel hasil utama Bab 4 Tugas Akhir. | $\approx 60$ menit |
| 7 | [07_eksplorasi_data_splitting_atribut_tuning_exp21.md](file:///d:/Projects/Klasifikasi%20Transasi%20Kartu%20Kredit%20-%20GraphSAGE/tugas-akhir-klasifikasi-transaksi-kartu-kredit/planning/07_eksplorasi_data_splitting_atribut_tuning_exp21.md) | `EXP21-REV` | **Remediasi & Eksplorasi EXP21 (Saran Pembimbing)**: Mengatasi bias *too good to be true* melalui 3 fase eksperimen:<br>• `EXP21A`: Audit & Eliminasi Kebocoran Atribut (`Zip`/`City`).<br>• `EXP21B`: Balanced Hyperparameter Tuning & Tree Regularization.<br>• `EXP21C`: Multi-Window Temporal Splitting & Walk-Forward Validation. | $\approx 45$ menit |
| 8 | [08_exp21d_optimal_synthesis_model.md](file:///d:/Projects/Klasifikasi%20Transasi%20Kartu%20Kredit%20-%20GraphSAGE/tugas-akhir-klasifikasi-transaksi-kartu-kredit/planning/08_exp21d_optimal_synthesis_model.md) | `EXP21D` | **The Golden Master Hybrid Pipeline (Sintesis EXP21A, EXP21B, EXP21C)**: Mengintegrasikan konfigurasi terbaik (clean features tanpa Zip/City, regularized XGBoost SPW=10-27, embargo gap 8 hari pada window 2018 aktif) menjadi satu model acuan utama. | $\approx 35$ menit |
| 9 | [09_exp21e_pure_end_to_end_graphsage.md](file:///d:/Projects/Klasifikasi%20Transasi%20Kartu%20Kredit%20-%20GraphSAGE/tugas-akhir-klasifikasi-transaksi-kartu-kredit/planning/09_exp21e_pure_end_to_end_graphsage.md) | `EXP21E` | **Pure End-to-End HeteroGraphSAGE Architecture (Proposal Alignment)**: Mengimplementasikan model klasifikasi murni GraphSAGE (Linear Head + Sigmoid) tanpa classifier downstream XGBoost, dengan seluruh preprocessing, fitur, window temporal, dan sampling identik dengan EXP21D. | $\approx 20$ menit |

---

## Alur Progresif Antar Eksperimen

```
[Hasil exp18c v2 Tersimpan]
         │
         ▼
┌──────────────────┐
│      EXP20       │ ──> Bukti empiris: F1 collapse semata akibat fixed threshold 0.5
└──────────────────┘
         │
         ▼
┌──────────────────┐
│      EXP21       │ ──> Standarisasi data era 2016-2020 (Hybrid GNN + XGBoost)
└──────────────────┘
         │
         ├───────────────────────────────────────────┐
         │ (Audit & Remediasi Masukan Pembimbing)     │
         ▼                                           ▼
┌──────────────────┐                       ┌──────────────────┐
│  EXP21A, B, C    │                       │      EXP22       │ ──> Validasi Agregator 'max'
└──────────────────┘                       └──────────────────┘
         │                                           │
         ▼                                           ▼
┌──────────────────┐                       ┌──────────────────┐
│      EXP21D      │ (The Golden Master)   │      EXP23       │ ──> Pure End-to-End GraphSAGE
└──────────────────┘                       └──────────────────┘
         │                                           │
         └─────────────────────┬─────────────────────┘
                               │
                               ▼
                     ┌──────────────────┐
                     │      EXP24       │ ──> Komparasi 4 strategi neighbor sampling
                     └──────────────────┘
                               │
                               ▼
                     ┌──────────────────┐
                     │      EXP25       │ ──> Studi ablasi final Bab 4
                     └──────────────────┘
```

