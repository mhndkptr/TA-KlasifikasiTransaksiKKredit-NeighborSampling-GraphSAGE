# Review teknis arsitektur proposal

Review ini merujuk Bab III proposal dan implementasi EXP12. Rancangan dasarnya
layak, tetapi definisi sampling perlu dibuat operasional agar eksperimen tidak
menguji tiga nama strategi yang secara matematis identik.

## Keputusan implementasi

| Area | Risiko pada definisi literal | Keputusan implementasi |
|---|---|---|
| Graf | Tiga tipe node lebih tepat disebut graf heterogen tripartit, atau bipartit dengan partisi `Transaction` dan gabungan `User/Merchant`. | `HeteroData` dipertahankan sebagai representasi audit. Runtime memakai CSR ekuivalen agar empat salinan `edge_index` int64 tidak menjadi beban RAM. |
| Fanout 25/10 | Setiap `Transaction` hanya memiliki dua tetangga langsung, sehingga fanout 10 tidak pernah memangkas apa pun. | Fanout 25 efektif saat entity mengambil historical transaction; kedua endpoint transaksi selalu dipakai pada hop berikutnya. Hal ini harus dinyatakan di laporan. |
| API sampler PyG | `NeighborLoader`/`NeighborSampler` standar tidak memberi kontrak eksperimen yang sama dengan tabel weighted-without-replacement yang dibekukan per epoch. | Runtime memakai `NeighborTableSampler` kustom dan `SAGEConv` PyG. Jangan menulis bahwa kelas ini mewarisi `torch_geometric.sampler.NeighborSampler`; sebut sebagai sampler kustom untuk pipeline PyG, atau tambahkan adapter `BaseSampler` sebagai pekerjaan terpisah. |
| Jaccard | `N(entity)` berisi transaksi, sedangkan `N(transaction)` berisi entity; intersection lintas tipe selalu kosong. | Mode `historical` memakai Jaccard antara himpunan transaksi milik kedua endpoint kandidat: `pair_count / (degree_user + degree_merchant - pair_count)`. Mode `literal` tersedia dan secara benar jatuh ke uniform. |
| Local homophily | User dan merchant tidak memiliki label; label root validation/test juga tidak boleh dipakai. | Estimasi risiko fraud train-only pada kedua endpoint, smoothed dan leave-one-candidate-out, digunakan sebagai agreement score. Ini adalah adaptasi dan harus disebut demikian, bukan formula literal Algoritma 3. |
| Degree centrality | Semua node transaksi pada graf asli memiliki degree dua, sehingga term centrality konstan. | `literal_transaction` tersedia untuk reproduksi proposal. Preset Kaggle memakai `projected_transaction`: jumlah unique train transactions yang berbagi user atau merchant, dibagi jumlah train transaction minus satu. |
| PPR | PPR global 24 juta node mahal dan berpotensi membaca struktur masa depan. | PPR tiga langkah dihitung closed-form pada graf train beku. Tidak ada validation/test edge, degree, atau label di dalam statistik. |
| Sigmoid dan BCE | Sigmoid di model lalu BCE mudah mengalami underflow/overflow. | Model mengeluarkan logits; training memakai `binary_cross_entropy_with_logits`; Sigmoid hanya saat inferensi. Secara fungsional output tetap probabilitas fraud. |
| Bobot 833 | Angka 833 berasal dari rasio global perkiraan dan dapat berbeda pada train kronologis. | `pos_weight = N_train_normal / N_train_fraud` dihitung setiap run. Jangan gabungkan dengan balanced root sampling karena akan memberi kompensasi ganda. |
| Early stopping | Memakai validation yang sama untuk checkpoint dan tuning threshold memberi optimisme tambahan. | 75% awal validation memilih checkpoint lewat AP; 25% akhir mengkalibrasi threshold F2; test baru dibaca sesudah keduanya terkunci. |

## Kontrak anti-leakage

1. CSV diurutkan berdasarkan timestamp dan source row dengan external sort.
2. Boundary 70/15/15 dipindahkan ke akhir kelompok timestamp yang sama; satu
   timestamp tidak pernah berada pada dua split.
3. Median/min/max, kategori, dan location frequency hanya di-fit pada train.
4. CSR history hanya memuat transaksi train. Degree embedding, pair count,
   Jaccard, homophily, centrality, dan PPR dibangun dari CSR tersebut.
5. Kandidat topology memakai leave-one-out terhadap label kandidat train.
6. Validation/test labels hanya digunakan pada evaluator. Test tidak memilih
   epoch, threshold, fitur, atau distribusi sampling.
7. Model utama memakai frozen-train graph. Protokol online yang menambahkan
   transaksi validation sebelumnya sebagai history tanpa label harus menjadi
   eksperimen terpisah dan tidak dicampur dengan hasil utama.

## Anggaran memori full data

Dengan 24.386.900 transaksi dan 15 fitur float16, perkiraan payload utama:

| Payload | Perkiraan |
|---|---:|
| Fitur transaksi | 698 MiB |
| Endpoint user/merchant int64 | 372 MiB |
| Label, channel, timestamp, source row | 442 MiB |
| CSR train (`col`, `rowptr`) | sekitar 261 MiB + entity kecil |
| Pair count train | sekitar 65 MiB |
| Total array persisten | sekitar 1,8 GiB |

Peak preprocessing lebih tinggi karena external sort, chunk DataFrame, dan
workspace `np.unique`, tetapi tetap jauh di bawah 30 GB dengan DuckDB dibatasi
8 GB. Simpan fitur pada CPU; hanya root batch, sampled table, parameter, dan
cached entity context berada di GPU. Mulai dengan batch 1.024 pada T4/P100.

## Protokol eksperimen yang disarankan

- Jalankan satu strategy-seed per Kaggle version: 3 strategi x 5 seed = 15 run.
- Gunakan source/config/data fingerprint yang sama untuk semua run.
- Laporkan mean, standard deviation, seluruh seed, AP lift terhadap prevalence,
  recall, precision, F1, FPR/alert rate, latency termasuk dan tidak termasuk
  persiapan bobot.
- Perlakukan test yang sudah dipakai berulang untuk diagnosis sebagai exploratory.
  Klaim final memerlukan holdout periode baru atau dataset eksternal.
- Tambahkan ablation `literal_transaction` vs `projected_transaction`; tanpa ini,
  kontribusi degree centrality tidak dapat dipisahkan dari PPR.
