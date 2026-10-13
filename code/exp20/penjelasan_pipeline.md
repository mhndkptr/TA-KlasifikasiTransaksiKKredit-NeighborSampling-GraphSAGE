# Alur Implementasi Pipeline Eksperimen 20 (exp20)

Eksperimen 20 (exp20) mengimplementasikan arsitektur *two-stage* untuk mendeteksi transaksi penipuan (fraud detection). Pipeline ini menggabungkan **Heterogeneous Graph Neural Network (GraphSAGE)** sebagai pembelajar representasi struktural (*feature extractor*) dan **XGBoost** sebagai pengklasifikasi akhir (*classifier*). 

Berbeda dengan eksperimen sebelumnya, `exp20` mempertahankan keseluruhan rasio kelas (tanpa *undersampling*) pada tahap pembentukan graf untuk menghindari kerusakan informasi topologi jaringan.

---

## 1. Tahapan Alur Pipeline

Pipeline ini berjalan secara sekuensial dari pemrosesan data, pembuatan graf, pelatihan GNN, hingga evaluasi XGBoost. Keseluruhan proses ini dikoordinasi di dalam skrip `main.py`.

### Tahap 1: Data Preprocessing & Temporal Split (`data_preprocessing.py`)
1. **Load & Clean Data:** Dataset transaksi kartu kredit dimuat, dilakukan *parsing* waktu menjadi tipe data `datetime`, dan fitur kategorikal di-*encode* menggunakan label encoding.
2. **Scaling:** Fitur metrik (seperti jumlah `Amount`) di-standarisasi agar konvergensi model GNN lebih optimal.
3. **Temporal Split:** Dataset diurutkan secara kronologis dan dibagi ke dalam subset Train, Validation, dan Test tanpa diacak (*no shuffling*). Pembagian berbasis waktu ini krusial pada deteksi anomali nyata (skenario *inductive*) agar model tidak mempelajari data masa depan.

### Tahap 2: Graph-Level Undersampling (`graph_builder.py`)
- Berdasarkan konfigurasi di `exp20`, proses *undersampling* diatur pada **rasio 1.0 (100%)**, yang artinya **tidak ada transaksi yang dihapus/dikurangi** pada kelas mayoritas (Normal). Hal ini dilakukan untuk secara utuh mempertahankan struktur asli topologi (*mega-hubs* dan *connected components*) di tahap eksperimen ini.

### Tahap 3: Pembentukan Graf Heterogen (`graph_builder.py`)
- Graf diinisialisasi dalam objek `HeteroData` (PyTorch Geometric) yang merupakan graf tripartit (memiliki 3 jenis *node*).
- **Nodes**: `Transaction` (membawa fitur dan label asli), `User`, dan `Merchant`.
- **Edges**: Direpresentasikan secara dua arah (*bidirectional*) yaitu `(User, buys, Transaction)` dan `(Merchant, sells, Transaction)`.

### Tahap 4: Neighbor Sampling (`samplers.py`)
Karena graf berskala besar, pembaruan keseluruhan graf (*full-batch*) sangat memakan memori. Oleh karena itu, tetangga disampel secara terstruktur (menggunakan parameter `num_neighbors`). Metode yang disediakan bisa dieksplorasi (`uniform`, `topology`, `importance`). Pada konfigurasi standar, dipakai metode **Uniform Sampling**, di mana setiap tetangga punya probabilitas yang sama besarnya untuk diagregasi informasinya.

### Tahap 5: Phase 1 - Pelatihan GraphSAGE Encoder (`main.py` & `model.py`)
1. Model `HeteroGraphSAGE` dilatih menggunakan *subset* Train dengan tujuan mereduksi *Binary Cross Entropy* (BCE) dari label `Is Fraud?`.
2. **Class-Weighted BCE Loss:** Karena tidak ada subsampling yang dilakukan (kelas sangat *imbalanced*), fungsi bobot *loss* (`pos_weight`) diberikan denda penalti tinggi secara manual yang proporsional dengan kelas mayoritas terhadap kelas minoritas (*fraud*).
3. Setelah beberapa epoch, model akan mengerti bagaimana membentuk titik *embedding* (*latent features*) dari struktur ketetanggaan.

### Tahap 6: Phase 2 - Ekstraksi Fitur Gabungan 
1. `HeteroGraphSAGE` yang sudah dilatih digunakan (dalam mode eval) untuk memproduksi nilai *embedding* akhir dari tiap *node* transaksi.
2. *Embedding* dari GNN ini lalu **digabung (concatenated)** dengan fitur orisinal bawaannya untuk memperkaya representasi atribut.

### Tahap 7: Phase 3 - Klasifikasi dengan XGBoost
1. Fitur gabungan dari tahap sebelumnya dipasok (di-fit) sebagai *input* pembelajaran bagi model **XGBoost Classifier**. XGBoost sangat tangguh dan cepat dalam mengombinasikan fitur terstruktur dan representasi matematis.
2. Parameter pembobotan kelas juga diberikan ke dalam model pohon ini menggunakan *scale_pos_weight*.

### Tahap 8: Phase 4 - Evaluasi Inductive
Model pada akhirnya di-tes melawan dataset Test (fitur yang belum pernah direpresentasikan ke dalam topologi maupun model saat fase latih). Metrik utamanya adalah F1-Score, Recall, AUPRC (Area Under the Precision-Recall Curve), dan Lift@1% (efisiensi prediksi pada persentase 1% probabilitas *fraud* teratas).

---

## 2. Detail Hyperparameter yang Digunakan

Parameter di bawah ini merujuk pada file konfigurasi `config.reference.yaml` (serta `config.kaggle.yaml`) dan aturan-aturan pembentukan (seperti di `main.py`).

### Data & Split Parameter
- **`train_fraction`**: `0.7` (70% transaksi pertama untuk dilatih)
- **`val_fraction`**: `0.15` (15% transaksi berikutnya untuk validasi)
- *(Implikasi Test Dataset adalah sisa ~15% kronologis terakhir)*

### Graph & Sampling
- **`undersampling_ratio`**: `1.0` (Tanpa *undersampling* untuk kelas normal)
- **`method`**: `"uniform"` (Setiap edge memiliki probabilitas di-*sample* yang setara)

### Model HeteroGraphSAGE (GNN)
- **`hidden_channels`**: `128` (Dimensi representasi dalam tiap lapisan GraphSAGE)
- **`out_channels`**: `1` (Output 1 dimensi probabilitas pada layer prediktor akhir GNN)
- **`num_layers`**: `2` (Jumlah sebaran informasi / kedalaman *hop* topologi)
- **`aggr`**: `"mean"` (Agregasi dengan metode merata-rata informasi tetangga)

### Training (GNN)
- **`batch_size`**: `1024` (Ukuran matriks *batch* dibuat besar agar gradien stabil)
- **`epochs`**: `10`
- **`learning_rate`**: `0.001` (Menggunakan optimizer Adam)
- **`num_neighbors`**: `[15, 10]` 
  - Mengambil acak **15 tetangga** terdekat di *hop* pertama.
  - Mengambil acak **10 tetangga** lagi di *hop* kedua (mengurangi kompleksitas pertumbuhan eksponensial tetangga).

### Model XGBoost Classifier
*(Dikodekan secara manual di dalam file `main.py`)*
- **`n_estimators`**: `500` (Total iterasi pohon keputusan yang dibangun bertahap)
- **`max_depth`**: `5` (Kedalaman setiap pohon, dibatasi di angka 5 agar tidak berujung *overfitting*)
- **`learning_rate`**: `0.05`
- **`eval_metric`**: `'logloss'`
- **`scale_pos_weight`**: Dihitung secara dinamis dari `num_neg / num_pos` di dalam Train Labels untuk menangani jumlah mayoritas penipuan minor secara struktural.
