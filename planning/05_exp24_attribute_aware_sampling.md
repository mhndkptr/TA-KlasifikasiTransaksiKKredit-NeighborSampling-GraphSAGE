# Rencana Eksperimen 24 (EXP24): Penyempurnaan Strategi Neighbor Sampling: Attribute-Aware Topology & Importance

## 1. Identitas Eksperimen
* **ID Eksperimen**: `EXP24`
* **Nama Rencana**: *Attribute-Aware & Temporal-Enriched Neighbor Sampling for GraphSAGE*
* **Tipe Eksperimen**: Eksplorasi Strategi Sampling Ketetanggaan (*Core Thesis Contribution*)
* **Target Utama**: Memutakhirkan strategi sampling `topology` dan `importance` dari aproksimasi derajat topologi murni menjadi pembobotan berbasis atribut transaksi (waktu dan besaran nominal), serta membandingkannya secara langsung dengan baseline `uniform` dan metode *Nearest-Neighbor Sampler* dari paper SHINE.

---

## 2. Latar Belakang & Motivasi
Fokus utama dari Tugas Akhir ini adalah **analisis strategi *neighborhood sampling* pada arsitektur GraphSAGE**.

Pada implementasi saat ini di `samplers.py` (exp16 s/d exp18c):
* `uniform`: Setiap edge memiliki bobot sama ($w = 1.0$).
* `topology`: Menggunakan aproksimasi invers derajat simpul ($w = 1 / (\text{deg} + \epsilon)$).
* `importance`: Menggunakan derajat simpul ($w = \text{deg}$) sebagai proksi untuk *Degree Centrality*.

Kelemahan pendekatan ini adalah **ketiadaan integrasi atribut transaksi ke dalam proses sampling**. Pada jaringan transaksi kartu kredit heterogen, dua transaksi yang terhubung ke satu *merchant* memiliki makna yang sangat berbeda jika satu terjadi 3 tahun lalu dan satu terjadi 2 menit lalu.

Pada paper SHINE (Bagian III-C & Tabel VII):
Peneliti mengusulkan **NN (Nearest-Neighbor) Sampler** berbasis jarak atribut:
$$w_{i,j} = \exp(-\|x_i \ominus x_j\|_2^2)$$
Hasil empiris pada Tabel VII membuktikan:
* Sampling acak (*random*): AUCPR **0.7788**
* Sampling berbasis kemiripan waktu (`timestamp, sub`): AUCPR **0.7913**
* Sampling berbasis kemiripan nominal (`amount, sub`): AUCPR **0.8329** (Peningkatan signifikan $+5.4\%$)

Pada EXP24, kita memutakhirkan strategi sampling agar tidak sekadar menghitung derajat koneksi graf, melainkan mempertimbangkan dimensi temporal (*recency*) dan disparitas nominal (*amount similarity*).

---

## 3. Hipotesis Penelitian
1. Memberikan bobot sampling lebih tinggi pada transaksi tetangga yang terjadi dalam jendela waktu berdekatan (*temporal recency*) akan menangkap pola kejahatan beruntun (*burst attacks*), sehingga menghasilkan representasi graf yang lebih diskriminatif.
2. Memprioritaskan tetangga dengan bobot nominal relevan (*amount-aware*) akan mencegah *noise* dari transaksi bernilai mikro yang tidak relevan.
3. Metode sampling berbasis atribut (*attribute-aware*) akan mengungguli *uniform sampling* dan aproksimasi derajat kasar pada metrik AUCPR dan Lift@1%.

---

## 4. Desain & Formulasi Strategi Sampling yang Diuji

Eksperimen ini membandingkan 4 strategi *neighbor sampling* secara *head-to-head* pada lingkungan data dan model yang sama:

### 1. Baseline: Uniform Random Sampling (`uniform`)
$$w_{i,j} = 1.0$$
Setiap transaksi historis memiliki peluang yang sama persis untuk diambil sampelnya.

### 2. Enhanced Topology-Aware Sampling (`topology_temporal`)
* **Konsep**: Menggabungkan redaman derajat (*inverse degree*) dengan *kedekatan waktu* (*temporal decay*):
  $$w_{i,j} = \frac{1}{\text{deg}(v) + \epsilon} \cdot \exp\left(-\frac{|t_i - t_j|}{\lambda}\right)$$
  di mana $\lambda$ adalah konstanta peluruhan waktu (misal 24 jam atau 7 hari). Transaksi yang terjadi pada rentang waktu berdekatan dengan transaksi target mendapatkan prioritas sampling tertinggi.

### 3. Enhanced Importance-Based Sampling (`importance_financial`)
* **Konsep**: Menilai kepentingan (*importance*) simpul berdasarkan bobot finansial dan frekuensi:
  $$w_{i,j} = \text{deg}(v) \cdot \log(1 + \text{Amount}_j)$$
  Transaksi dengan perputaran dana dan aktivitas tinggi diprioritaskan untuk mengalirkan sinyal ke node perantara (*merchant/user*).

### 4. Attribute-Similarity / NN Sampler (`shine_nn_sampler`)
* **Konsep (Replikasi SHINE)**:
  $$w_{i,j} = \exp\left(-\frac{(Amount_i - Amount_j)^2}{2\sigma^2}\right)$$
  Mengutamakan tetangga yang memiliki nilai transaksi sepadan untuk mendeteksi anomali pada klaster nilai uang tertentu.

---

## 5. Implementasi pada PyTorch Geometric `NeighborLoader`
* Pembobotan dihitung saat inisialisasi graf dan disimpan pada atribut `edge_weight` pada setiap tipe relasi di `HeteroData`:
  ```python
  data['user', 'buys', 'transaction'].edge_weight = weights_user_tx
  data['merchant', 'sells', 'transaction'].edge_weight = weights_merch_tx
  ```
* `NeighborLoader` dipanggil dengan parameter `weight_attr='edge_weight'` untuk melakukan pengambilan sampel proporsional secara stokastik.

---

## 6. Rencana File & Struktur Output
* Folder Skrip: `code/exp24/`
* Modifikasi File:
  * `samplers.py`: Implementasi fungsi pembobotan terpadu (`apply_uniform_sampling`, `apply_temporal_topology_sampling`, `apply_financial_importance_sampling`, `apply_nn_similarity_sampling`).
  * `run_all.py`: Otomasi eksekusi sekuensial untuk ke-4 metode sampling.
* Output Hasil:
  * `result/exp24/exp24_uniform_results.json`
  * `result/exp24/exp24_topology_results.json`
  * `result/exp24/exp24_importance_results.json`
  * `result/exp24/exp24_nn_sampler_results.json`
  * `result/exp24/sampling_comparative_table.md`
  * `result/exp24/README.md`

---

## 7. Kriteria Keberhasilan (Success Criteria)
1. Keempat metode sampling berhasil dieksekusi secara otomatis dan menghasilkan metrik yang valid.
2. Minimal satu dari metode sampling berbasis atribut (`topology_temporal` atau `nn_sampler`) membuktikan keunggulan statistik atas `uniform sampling` pada metrik **AUCPR** dan **Lift@1%**.
3. Efisiensi waktu komputasi sampling tetap terjaga (overhead komputasi bobot edge hanya terjadi 1 kali di awal).

---

## 8. Kontribusi untuk Laporan Tugas Akhir
* Ini adalah **inti kontribusi ilmiah Tugas Akhir**: menjawab secara komprehensif rumusan masalah mengenai *perbandingan strategi neighbor sampling pada GraphSAGE untuk deteksi penipuan transaksi kartu kredit*.
* Memberikan bukti empiris yang mengaitkan temuan paper SHINE dengan kerangka kerja penelitian yang Anda ajukan.
