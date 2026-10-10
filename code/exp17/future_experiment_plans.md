# Rencana Strategis Lanjutan: Eksperimen Klasifikasi GraphSAGE

Dokumen ini memuat peta jalan (*roadmap*) eksperimen lanjutan untuk meningkatkan metrik AUPRC (Area Under Precision-Recall Curve) menuju target ~0.30 untuk deteksi penipuan transaksi kartu kredit menggunakan GraphSAGE dan XGBoost.

---

## 🟢 FASE SAAT INI (SEDANG BERJALAN): Eksperimen 18c (Revisi)

**Status:** Sedang dieksekusi di Kaggle (`muhammadhendikaputra/exp18c-graphsage-training`).

**Deskripsi:** 
Menggabungkan Graph-Level Undersampling moderat (30%) dengan Rich Node Features (rata-rata nilai transaksi dan frekuensi transaksi untuk tiap *Client* dan *Merchant*).

**Apa yang Dimodifikasi:**
1. `graph_builder.py`: Menambahkan `StandardScaler` (Scikit-Learn) untuk menormalkan fitur *Client* dan *Merchant* sebelum dikonversi menjadi *tensor*. Ini untuk mencegah *exploding gradients* akibat fitur `tx_count` yang nilainya tidak terbatas.
2. `config.kaggle.yaml` & `config.reference.yaml`: Mengubah `undersampling_ratio` dari `0.02` (2%) yang terbukti merusak struktur graf, menjadi `0.3` (30%) agar konektivitas jaringan *fraud* dan relasi antar tetangga tetap kuat.

**Hasil yang Diharapkan:**
Metrik `uniform` baseline sudah membuktikan peningkatan AUPRC ke 0.0664. Diharapkan metode `topology` dan `importance` mampu menembus AUPRC > 0.20 karena mendapat pasokan *embeddings* yang jauh lebih kaya.

**Langkah Lanjutan (Jika Hasil Tidak Sesuai Target 0.30):**
Jika AUPRC untuk `topology` masih tertahan di kisaran 0.15 - 0.20, itu menandakan arsitektur GraphSAGE saat ini "kurang bertenaga" untuk mencerna fitur tambahan tersebut. Maka kita akan lanjut ke **Eksperimen 19**.

---

## 🟡 FASE 2: Eksperimen 19 - Ekspansi Kapasitas & Arsitektur GNN (GNN Tuning)

**Deskripsi:**
Meningkatkan kapasitas memori dan kedalaman jaringan GraphSAGE agar mampu merangkum topologi graf dan *Rich Node Features* dengan lebih presisi.

**Apa yang Dimodifikasi:**
1. `config.kaggle.yaml`:
   - Mengembalikan/Meningkatkan `hidden_channels` dari 128 menjadi **256**.
   - Menambah `num_layers` (jumlah hop GraphSAGE) dari 2 menjadi **3** lapisan.
2. `model.py`: 
   - Bereksperimen dengan parameter `aggr='lstm'` atau `aggr='max'` pada modul `SAGEConv` ketimbang sekadar `mean` (rata-rata), agar anomali (nilai transaksi ekstrem) tidak tenggelam saat diagregasi.
   - Menambahkan *Dropout* jika penambahan lapisan memicu *overfitting*.

**Hasil yang Diharapkan:**
GraphSAGE dapat mendeteksi pola berlapis (misal: *User -> Merchant A -> User Fraud -> Merchant B*). AUPRC diharapkan naik karena vektor *embedding* yang diumpankan ke XGBoost menjadi sangat terpisah/dapat dibedakan (separabilitas tinggi).

**Langkah Lanjutan (Jika Hasil Tidak Sesuai):**
Jika penambahan *layer* GNN justru menurunkan performa (*oversmoothing*) atau skor tetap stabil tanpa perbaikan, berarti keterbatasan (*bottleneck*) ada di ujung *pipeline*, yakni pada pengklasifikasi XGBoost. Lanjut ke **Eksperimen 20**.

---

## 🟠 FASE 3: Eksperimen 20 - Optimalisasi XGBoost Classifier (Imbalance Tuning)

**Deskripsi:**
Model XGBoost saat ini dilatih menggunakan parameter dasar (*default*). Karena kasus penipuan sangat *imbalanced* (meski sudah ada *undersampling* di sisi graf, di sisi pelatihan akhir rasio fraud tetap kecil), XGBoost perlu dirancang khusus.

**Apa yang Dimodifikasi:**
1. `main.py`:
   - Modifikasi inisialisasi `XGBClassifier`.
   - Menambahkan parameter `scale_pos_weight` yang dihitung secara dinamis dari perbandingan sampel normal vs *fraud* di *training set* XGBoost.
   - Mengatur `max_depth` menjadi 4 atau 5 (lebih dangkal agar tidak *overfit* ke noise).
   - Menurunkan `learning_rate` ke 0.05 atau 0.01 dikombinasikan dengan iterasi pohon (`n_estimators`) yang lebih besar (misal 500).

**Hasil yang Diharapkan:**
Pola presisi akan meningkat tajam. XGBoost tidak akan mudah mengklasifikasikan transaksi abu-abu sebagai non-fraud karena penalti untuk *false negative* sudah diperbesar (*scale_pos_weight*). Target AUPRC 0.30 tercapai.

**Langkah Lanjutan (Jika Hasil Tidak Sesuai):**
Jika XGBoost pun gagal mendongkrak performa, berarti informasi mentah (data asli + struktur) memang belum memadai untuk mencapai tingkat akurasi paper referensi. Lanjut ke iterasi terakhir: **Eksperimen 21**.

---

## 🔴 FASE 4: Eksperimen 21 - Advanced Fraud-Oriented Feature Engineering

**Deskripsi:**
Meningkatkan kualitas data mentah dengan menambahkan fitur statistik prediktif (anomali perilaku/pergeseran pola waktu) secara manual ke *Node* dan *Edge*.

**Apa yang Dimodifikasi:**
1. `data_preprocessing.py` & `graph_builder.py`:
   - Membuat fitur **Sliding Window / Velocity**: 
     - "Jumlah transaksi dalam 24 jam terakhir untuk setiap Client".
     - "Perubahan/selisih jumlah transaksi saat ini vs rata-rata harian Client".
   - Membuat label "Risiko Merchant" secara historis (apakah merchant ini sering terasosiasi dengan *fraud* di masa lalu).
   - Menginjeksikan fitur ini kembali ke *Rich Node Features* atau *Edge Features*.

**Hasil yang Diharapkan:**
Sinyal penipuan akan menjadi sangat jernih. Gabungan antara fitur kelainan (anomali berbasis waktu) yang sangat presisi dengan pengenalan pola jaringan dari GraphSAGE akan mendorong AUPRC mencapai titik optimal di atas 0.30, jauh mengalahkan semua *baseline* saat ini.
