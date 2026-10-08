# Base Knowledge: Inductive Graph Representation Learning for Fraud Detection

Dokumen ini berisi rangkuman dan basis pengetahuan (*knowledge base*) yang diekstrak dari repositori **Inductive-Graph-Representation-Learning-for-Fraud-Detection**. Repositori ini merupakan implementasi eksperimental dari paper *'Inductive Graph Representation Learning for Fraud Detection'*.

## 1. Konsep Utama & Latar Belakang

Repositori ini berfokus pada penggunaan teknik *Graph Representation Learning* (pembelajaran representasi graf) secara induktif untuk mendeteksi penipuan (*fraud detection*), khususnya pada jaringan transaksi kartu kredit.
Dua tantangan utama pada jaringan transaksi ini:
1. **Dinamis:** Jaringan transaksi kartu kredit selalu berkembang dengan munculnya transaksi baru setiap saat.
2. **Sangat Tidak Seimbang (*Highly Imbalanced*):** Hanya sebagian kecil transaksi (sekitar 0.65% dalam dataset yang digunakan) yang merupakan penipuan.

Pendekatan induktif memungkinkan model untuk menghasilkan *embedding* pada node yang baru muncul tanpa harus melakukan pelatihan ulang pada seluruh graf.

## 2. Struktur Repositori

Repositori ini diatur dengan komponen modular yang mencakup tahapan *pipeline* eksperimen:
- **`inductiveGRL/`**: Paket Python yang berisi implementasi modular komponen *pipeline*.
  - `graphconstruction.py`: Membangun graf dari dataset menggunakan `NetworkX` dan `StellarGraph`.
  - `hinsage.py`: Implementasi model HinSAGE (varian heterogen dari GraphSAGE) menggunakan `TensorFlow` / `Keras` dan `StellarGraph`.
  - `timeframes.py`: Fungsi untuk memproses *rolling window setup* (data berbasis waktu).
  - `evaluation.py`: Metrik evaluasi khusus untuk dataset tidak seimbang (*Lift score*, *Precision-Recall Curve*).
- **`Demo/`**: Berisi *Jupyter Notebooks* untuk menjalankan eksperimen dalam bentuk yang disederhanakan dengan dataset sintesis (`demo_ccf.csv`).
  - `Demo GraphSAGE.ipynb`
  - `Demo FI-GRL.ipynb`
  - `FIGRL.m` dan `FIGRL.py`: Implementasi dan antarmuka Python-Matlab untuk metode FI-GRL.

## 3. Komponen Pipeline Eksperimental

Pipeline dibagi menjadi enam komponen utama:

### 3.1 Data Transaksi (Transaction Data)
Menggunakan pendekatan *rolling window* untuk menangani data berbasis waktu (diimplementasikan di `timeframes.py`). Dataset memuat informasi seperti klien, *merchant*, MCC (*Merchant Category Code*), negara, jumlah uang, waktu, dan label *fraud*.

### 3.2 Konstruksi Graf (Graph Construction)
Diimplementasikan dalam `graphconstruction.py`. Graf dibangun sebagai **Graf Tripartit Heterogen** (*heterogeneous tripartite graphs*) yang berisi tiga tipe node:
- **Client**
- **Merchant**
- **Transaction** (satu-satunya node yang dikonfigurasi dengan *node features*)

### 3.3 GraphSAGE / HinSAGE
Diimplementasikan dalam `hinsage.py`. Menggunakan *framework* HinSAGE (Heterogeneous GraphSAGE) untuk mempelajari *embedding* dari node transaksi secara *supervised*. 
- Menggunakan `StellarGraph` untuk sampling (berjalan dengan *random walk*).
- Jaringan syaraf dikembangkan menggunakan Keras/TensorFlow. Loss function yang digunakan adalah `binary_crossentropy` karena output akhirnya berfokus pada klasifikasi biner.

### 3.4 FI-GRL (Fast Inductive Graph Representation Learning)
Digunakan sebagai alternatif pembelajaran *embedding*. Komponen FI-GRL ditulis dalam bahasa **Matlab** (`FIGRL.m`). Untuk menjalankannya di Python, digunakan modul `matlab.engine` yang berfungsi sebagai *bridge*.

### 3.5 Klasifikasi (Classifier)
Setelah *embedding* (representasi numerik dari node transaksi) dihasilkan oleh HinSAGE atau FI-GRL, representasi tersebut dimasukkan ke dalam model klasifikasi standar untuk memprediksi apakah transaksi adalah *fraud* atau *legitimate*. Umumnya, repositori ini merekomendasikan **XGBoost** sebagai classifier karena kinerjanya yang sangat baik pada data terstruktur dan tidak seimbang, walaupun algoritma lain juga bisa digunakan.

### 3.6 Evaluasi (Evaluation)
Mengingat label penipuan sangat minoritas (*imbalanced*), metrik akurasi biasa tidak cocok. File `evaluation.py` menyediakan evaluasi menggunakan:
- **Lift Score & Lift Curve**
- **Precision-Recall (PR) Curve**

## 4. Dependencies dan Teknologi
- **Python Frameworks:** `networkx`, `stellargraph`, `tensorflow`, `keras`, `pandas`
- **Machine Learning:** `xgboost` (direkomendasikan)
- **Matlab:** Diperlukan konfigurasi `matlab.engine` secara spesifik bagi yang ingin menjalankan pipeline FI-GRL.

## 5. Insight Penting untuk Pengembangan Lanjutan
1. **Modularitas:** Desain kode sangat modular. Dataset awal (kartu kredit) bisa diganti dengan dataset lain asalkan dibentuk dalam relasi graf yang relevan.
2. **Graph-Level Undersampling:** Disebutkan dalam paper bahwa melakukan *undersampling* di tingkat graf dapat sangat membantu dalam merepresentasikan jaringan yang tidak seimbang (*imbalanced networks*).
3. **Penyajian Data:** Fitur (*node features*) hanya dilekatkan pada node transaksi. Jika ingin mengembangkan fitur tambahan, fokus perubahannya ada pada konstruksi properti dari node transaksi.
