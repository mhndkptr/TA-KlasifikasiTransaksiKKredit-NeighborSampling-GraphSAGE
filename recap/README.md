# Recap EXP1-EXP14

Folder ini berisi rekap metrik dan analisis eksperimen dari EXP1 sampai EXP14.

Dokumen knowledge base lengkap tentang struktur, fitur, alasan perubahan,
hasil, kegagalan, dan keterkaitan seluruh eksperimen:
`knowledge_base_exp1_exp14.md`.

Cara regenerate tabel dan grafik:

```powershell
python .\recap\generate_recap.py
```

Output utama:

- `knowledge_base_exp1_exp14.md`: dokumentasi menyeluruh EXP1-EXP14.
- `experiment_metrics_detail.csv`: semua run/metrik lokal yang berhasil dibaca.
- `experiment_strategy_summary.csv`: agregasi per eksperimen dan strategy.
- `experiment_best_by_experiment.csv`: run lokal terbaik per eksperimen berdasarkan AP/AUPRC.
- `experiment_metric_comparison.md`: tabel metrik dalam format Markdown.
- `experiment_differences_and_failures.md`: penjelasan perbedaan antar eksperimen dan penyebab kegagalan/limitasi.
- `figures/*.png`: grafik siap diunduh/dipakai di laporan.
- `recap_experiments.ipynb`: notebook ringan untuk menjalankan ulang dan menampilkan hasil.

Catatan interpretasi: tidak semua eksperimen memakai protokol yang identik.
Sebagian adalah smoke test, pilot, ablation, atau deployment layer. Untuk klaim
ilmiah, bandingkan strategy yang berada dalam comparison/protokol yang sama.
