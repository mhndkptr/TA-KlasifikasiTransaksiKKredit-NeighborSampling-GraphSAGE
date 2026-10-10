# Penjelasan Lengkap Pipeline GraphSAGE + XGBoost (Eksperimen 17)

Dokumen ini menjelaskan alur pipeline, struktur data, hyperparameter, dan rumus-rumus yang digunakan dalam implementasi *Inductive Graph Representation Learning* menggunakan arsitektur **Heterogeneous GraphSAGE** (sebagai feature extractor) dilanjutkan dengan **XGBoost** (sebagai classifier) pada eksperimen 17 (`exp18b`).

Dokumen ini juga menyertakan **sumber referensi dan alasan** dari setiap penentuan nilai, atribut, hiperparameter, maupun rumus yang digunakan. Sebagian besar pengaturan mengacu pada repositori penelitian **"Inductive Graph Representation Learning for Fraud Detection" (oleh Charles Van Damme)** serta praktik umum pada makalah **GraphSAGE (Hamilton et al., 2017)**.

---

## 1. Alur Pipeline (Pipeline Flow)

Pipeline sistem terdiri dari beberapa tahap mulai dari pemrosesan data, pembentukan graf, sampling ketetanggaan, pelatihan model, hingga evaluasi induktif.

### Tahap 1: Data Loading & Preprocessing (`data_preprocessing.py`)
1. **Load Data**: Membaca dataset *IBM Credit Card Transactions*.
2. **Parsing Waktu (Temporal)**: Menggabungkan kolom `Year`, `Month`, `Day`, dan `Time` menjadi tipe data `datetime`. 
3. **Sorting**: Mengurutkan seluruh dataset secara kronologis berdasarkan waktu transaksi.
   - **Alasan & Referensi**: Penting untuk skenario *inductive* dan *temporal split* untuk menghindari *data leakage* (model tidak boleh melihat transaksi di masa depan saat dilatih). Ini adalah standar wajib dalam *Fraud Detection*.
4. **Data Cleaning**: Menghapus simbol `$` dan `,` pada kolom `Amount` lalu mengubahnya menjadi tipe float.
5. **Encoding Target**: Mengubah label kelas `Is Fraud?` ('Yes' -> 1, 'No' -> 0).
6. **Encoding Kategorial**: Menggunakan `LabelEncoder` pada fitur kategorik (`Use Chip`, `Merchant City`, `Merchant State`, `Zip`, `MCC`, `Errors?`).
7. **Scaling**: Melakukan standarisasi pada kolom `Amount` menggunakan `StandardScaler`.
   - **Alasan & Referensi**: Jaringan saraf tiruan (termasuk GNN) sangat sensitif terhadap variasi rentang nilai *input*. Nilai uang (`Amount`) bisa sangat besar sehingga akan mendominasi gradien jika tidak diskalakan menjadi distribusi normal ($\mu=0, \sigma=1$).

### Tahap 2: Temporal Split (`data_preprocessing.py`)
Dataset dibagi ke dalam tiga subset berdasarkan urutan waktu (kronologis):
- **Train**: 70% transaksi pertama.
- **Validation**: 15% transaksi berikutnya.
- **Test**: 15% transaksi terakhir.
- **Alasan & Referensi**: Proporsi **70/15/15** yang dibagi secara **kronologis** ini secara eksplisit mengikuti *setup* pemisahan pada repositori referensi penelitian (Charles Van Damme) untuk mensimulasikan lingkungan deteksi penipuan di dunia nyata secara *inductive*.

### Tahap 3: Graph-Level Undersampling (`graph_builder.py`)
Tahap ini **hanya dilakukan pada data Train** untuk menangani *class imbalance*:
- Mempertahankan **semua** transaksi penipuan (fraud).
- Mengurangi transaksi normal (undersampling) secara acak sesuai dengan rasio `0.02` (2%).
- **Alasan & Referensi**: Diambil dari observasi paper referensi di mana rasio *undersampling* sebesar **2%** terbukti cukup menyeimbangkan distribusi data sekaligus mengurangi beban memori komputasi graf raksasa tanpa menghilangkan terlalu banyak informasi mayoritas.

### Tahap 4: Pembentukan Graf Heterogen (`graph_builder.py`)
Graf dibangun menggunakan format `HeteroData` dari PyTorch Geometric. 
- **Nodes**: `User`, `Merchant`, `Transaction`
- **Edges**: `(User, buys, Transaction)`, `(Merchant, sells, Transaction)` serta *reverse edges* (`rev_buys` dan `rev_sells`).
- **Alasan & Referensi**: GraphSAGE adalah algoritma propagasi pesan dua arah (*bidirectional*). Agar pesan bisa mengalir dari *Transaction* kembali ke *User* atau *Merchant* untuk pembaruan *embedding*, maka setiap edge harus dibuat bolak-balik.

### Tahap 5: Neighbor Sampling (`samplers.py`)
Untuk menangani graf yang besar, digunakan `NeighborLoader` (PyG). Ada 3 metode yang dapat dipilih:
- **Uniform Sampling**: Peluang terpilihnya semua tetangga (edge weight) adalah seragam (1.0). (Baseline GraphSAGE standar).
- **Topology-Aware Sampling**: Probabilitas dimodifikasi berbasis invers derajat (*inverse degree*).
  - **Alasan & Referensi**: Pendekatan ini adalah proksi untuk perhitungan *Homophily* atau *Jaccard Similarity* yang diusulkan referensi. Karena menghitung *Jaccard overlap* secara eksak pada graf heterogen sangat memakan komputasi, bobot didekati menggunakan invers derajat untuk meredam dominasi *node* dengan koneksi super banyak.
- **Importance-Based Sampling**: Menggunakan nilai *Degree Centrality*.
  - **Alasan & Referensi**: Ini adalah proksi untuk pendekatan *Personalized PageRank (PPR)* di paper referensi. *Node* yang memiliki sentralitas tinggi (*Hubs*) diberikan prioritas *sampling* untuk memastikan aliran pesan melewati struktur sentral.

### Tahap 6 & 7: Definisi Model dan Training (`model.py` & `main.py`)
- **Heterogeneous GraphSAGE**: Agregasi `mean` dipakai sebagai fungsi *aggregator* (merujuk pada Hamilton et al. yang menyatakan *mean aggregator* sangat stabil sebagai *baseline*).
- **Optimizer Adam**: Menggunakan *learning rate* 0.001 (Standar default *Deep Learning* yang stabil).
- **Class-Weighted Binary Cross-Entropy (BCE)**:
  - **Alasan & Referensi**: Meski sudah dilakukan *undersampling* 2%, dataset fraud masih tetap berstatus *imbalanced* (kelas fraud jauh lebih sedikit dari kelas normal yang tersisa). Pembobotan pada *loss function* memastikan model lebih ter-penalti jika salah memprediksi kelas Fraud.

### Tahap 8: Inductive Evaluation (`main.py`)
Mengevaluasi model pada data Test dan diukur menggunakan metrik F1-Score, Recall, AUPRC, dan Lift@1%.
- **Alasan & Referensi**: Dalam konteks kartu kredit, *Accuracy* adalah metrik yang buruk. AUPRC (Area Under Precision-Recall Curve) dan Lift@1% adalah standar evaluasi deteksi penipuan (*Fraud Detection*) di ranah industri.

---

## 2. Atribut pada Setiap Node dan Pemilihannya

Atribut yang didistribusikan pada graf *heterogeneous* ini dirancang sesuai dengan fitur dataset IBM.

### Node `Transaction`
- **Atribut/Features (x)**: 
  - `Amount` (Numerik, *Scaled*)
  - `Use Chip`, `Merchant City`, `Merchant State`, `Zip`, `MCC`, `Errors?` (Kategorikal, *Label Encoded*)
- **Label (y)**: `Is Fraud?`
- **Alasan Pemilihan Atribut**: Fitur-fitur ini adalah variabel yang terikat langsung pada aktivitas transaksional (kapan, di mana, bagaimana, dan berapa banyak). Fitur ini memiliki sinyal kuat (*predictive power*) untuk membedakan anomali.

### Node `User` dan Node `Merchant`
- **Atribut**: Tidak memiliki atribut sama sekali (dimensi fitur 0).
- **Inisialisasi**: *Learnable Parameter Embeddings* (vektor random ukuran `hidden_channels`).
- **Alasan & Referensi**: Dalam dataset IBM, tidak ada metrik demografi User yang eksplisit selain identitas (ID), dan tidak ada fitur Merchant selain nama dan lokasi transaksinya. Dalam *Graph Representation Learning*, jika node tidak memiliki atribut, *best practice*-nya adalah menginisialisasi dengan parameter yang bisa dilatih (`torch.randn`) agar model dapat "mempelajari" fitur laten dari user/merchant tersebut berdasarkan riwayat transaksi historis mereka.

---

## 3. Detail Hyperparameter

Berdasarkan pengaturan pada `config.reference.yaml` (yang merujuk ke penelitian referensi):

- `undersampling_ratio`: **0.02**
  - **Referensi**: Secara eksplisit diubah dari 0.1 menjadi 0.02 menyesuaikan *2% undersampling rate* yang digunakan pada eksperimen di makalah referensi (Charles Van Damme).
- `hidden_channels`: **64**
  - **Referensi**: Dimensi keluaran *embedding* dari algoritma GraphSAGE yang digunakan sebagai ukuran dimensi final pada eksperimen referensi.
- `num_layers`: **2**
  - **Referensi**: GraphSAGE pada umumnya bekerja paling optimal pada kedalaman 2 hop. Lebih dari itu biasanya menyebabkan isu *oversmoothing* (semua *node* memiliki representasi yang sama).
- `batch_size`: **50**
  - **Referensi**: Ukuran *batch* disesuaikan dengan arsitektur eksperimen *HinSAGE* di *notebook* penelitian referensi (`Experimental Pipeline.ipynb`).
- `epochs`: **10**
  - **Referensi**: Jumlah *epoch* disesuaikan berdasarkan iterasi yang digunakan di dalam *notebook* eksperimen referensi.
- `num_neighbors`: **[2, 32]**
  - **Referensi**: Mengambil sampel **2 tetangga** pada hop pertama dan **32 tetangga** pada hop kedua. Angka ini murni diadaptasi dari pengaturan paper/eksperimen referensi untuk membatasi *neighborhood explosion* namun tetap mengambil konteks dari struktur yang berdekatan.

---

## 4. Rumus-rumus yang Digunakan beserta Penjelasannya

### 1. Class-Weighted Binary Cross Entropy (BCE) Loss
- **Formula Positif Weight:**
  $$W_{pos} = \frac{N_{neg}}{N_{pos} + 1e^{-7}}$$
- **Formula Weighted BCE:**
  $$Loss = -\frac{1}{N} \sum_{i=1}^{N} \left[ W_{pos} \cdot y_i \cdot \log(p_i + 1e^{-7}) + (1 - y_i) \cdot \log(1 - p_i + 1e^{-7}) \right]$$
- **Alasan & Referensi**: Penggunaan $W_{pos}$ memberikan "bobot denda" yang proporsional dengan jumlah mayoritas. Konstanta $1e^{-7}$ (*epsilon*) digunakan secara matematis untuk mencegah munculnya hasil tak terhingga (*NaN/Infinity*) akibat dari menghitung nilai probabilitas $\log(0)$.

### 2. Standarisasi Fitur Numerik (StandardScaler)
- **Formula**:
  $$z = \frac{x - \mu}{\sigma}$$
- **Alasan & Referensi**: Standardisasi fitur metrik (Z-score normalization). Nilai transaksi uang (`Amount`) diskalakan agar memiliki rata-rata ($\mu$) 0 dan standar deviasi ($\sigma$) 1. Ini mempercepat konvergensi gradien di algoritma *Adam* dan mencegah fitur nilai besar mendominasi.

### 3. Modifikasi Probabilitas Neighbor Sampling
Karena ukuran graf industri terlalu masif, *homophily* dan PPR yang diajukan oleh makalah diaproksimasi menggunakan sentralitas derajat (Degree Centrality).

- **Degree Centrality $C_D(v)$**:
  $$C_D(v) = \text{Total edges pada node } v$$

- **Topology-Aware Weights (Proxy Inverse-Degree)**:
  $$w_v = \frac{1}{C_D(v) + \epsilon}$$
  - **Alasan**: Meredam probabilitas pengambilan sampel pada *node* yang punya terlalu banyak koneksi (*mega-hubs*) karena koneksi berlebih sering kali memuat informasi berisik (*noise*) ketika kita ingin mencari kesamaan struktur topologi (sebagai *proxy* untuk *Jaccard Similarity* di graf heterogen). Epsilon $\epsilon$ ditambahkan mencegah pembagian dengan nol.

- **Importance-Based Weights (Proxy PPR / Hub Prioritization)**:
  $$w_v = C_D(v)$$
  - **Alasan**: Semakin sentral atau penting sebuah *node* (koneksi banyak), semakin tinggi kemungkinan di-*sample*. Ini adalah aproksimasi cepat dari algoritma *PageRank* yang direkomendasikan pada paper referensi.

### 4. Evaluasi Deteksi Penipuan (Lift @ 1%)
- **Formula**:
  $$Lift@1\% = \frac{\% \text{ Fraud di dalam 1\% transaksi dengan skor risiko tertinggi}}{\% \text{ Fraud di keseluruhan dataset}}$$
- **Alasan & Referensi**: Metrik bisnis utama yang diadopsi dari paper referensi dan praktik industri kartu kredit. Daripada memeriksa seluruh transaksi, pihak Bank/Sistem *Fraud* biasanya hanya memiliki sumber daya untuk memeriksa maksimal 1% transaksi paling mencurigakan setiap harinya. Lift@1% mengukur seberapa jauh lebih efisien model kita dibandingkan pemeriksaan acak pada kelompok 1% teratas ini. Menggunakan metrik ini membuktikan nilai bisnis nyata dari penerapan GNN.
