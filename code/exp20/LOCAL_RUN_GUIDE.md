# Panduan Eksekusi Eksperimen 20 secara Lokal (Local Run)

Dokumen ini berisi panduan langkah-demi-langkah untuk menjalankan **Eksperimen 20 (GraphSAGE + XGBoost Tuning)** di komputer lokal (Windows) milik Anda.

## 1. Persiapan Dataset
Pastikan dataset IBM Credit Card Transactions sudah berada di tempat yang tepat. Konfigurasi bawaan (`config.reference.yaml`) akan mencari dataset di *path* relatif berikut:
`../../dataset/credit_card_transactions-ibm_v2.csv`

Pastikan struktur folder Anda seperti ini:
```text
D:\Projects\Klasifikasi Transasi Kartu Kredit - GraphSAGE\tugas-akhir-klasifikasi-transaksi-kartu-kredit\
├── dataset\
│   └── credit_card_transactions-ibm_v2.csv
└── code\
    └── exp20\
        ├── main.py
        ├── config.reference.yaml
        └── ...
```

## 2. Membuat dan Mengaktifkan Virtual Environment (venv)
Buka terminal (Command Prompt atau PowerShell), lalu arahkan ke dalam folder `exp20`:

```powershell
# 1. Masuk ke direktori utama code
cd "D:\Projects\Klasifikasi Transasi Kartu Kredit - GraphSAGE\tugas-akhir-klasifikasi-transaksi-kartu-kredit\code"

# 2. Buat Virtual Environment bernama .venv
python -m venv .venv

# 3. Aktifkan Virtual Environment (Untuk Windows PowerShell)
.\.venv\Scripts\activate

# 4. (Opsional) Verifikasi venv mana yang sedang aktif
# Perintah ini akan menampilkan path 'python.exe' yang harus berada di dalam folder '.venv\Scripts\' Anda
Get-Command python

# 5. Masuk ke direktori exp20 untuk menjalankan eksperimen
cd exp20
```
*(Selain itu, ciri venv sudah aktif adalah munculnya awalan `(.venv)` pada baris perintah terminal PowerShell Anda).*

## 3. Instalasi Dependencies
Setelah `venv` aktif, kita perlu meng-install *library* Python yang dibutuhkan. Jalankan perintah berikut:

```powershell
# Update pip terlebih dahulu
python -m pip install --upgrade pip

# Install PyTorch versi CUDA 12.1 (Optimal untuk RTX series seperti RTX 5060 Ti)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install PyTorch Geometric dan dependensinya
pip install torch_geometric

# Install library pendukung lainnya (Pandas, Scikit-Learn, XGBoost, PyYAML)
pip install pandas numpy scikit-learn xgboost pyyaml
```
*(Catatan: PyTorch versi CUDA 12.1 sangat cocok dan mendukung komputasi paralel dari arsitektur GPU RTX terkini sehingga waktu komputasi GraphSAGE dapat dipangkas secara drastis).*

## 4. Menjalankan Eksperimen
Eksperimen dapat dijalankan secara terpisah berdasarkan metode *sampling*, atau dieksekusi semuanya secara berurutan.

### Opsi A: Menjalankan Eksperimen Spesifik (Direkomendasikan untuk uji coba awal)
Anda bisa menentukan metode spesifik, misalnya metode **uniform**:
```powershell
python main.py --config config.reference.yaml --sampling uniform
```

Untuk menguji dengan dataset kecil (agar cepat selesai tanpa harus menunggu jutaan baris), Anda bisa menambahkan argumen `--nrows`:
```powershell
python main.py --config config.reference.yaml --sampling uniform --nrows 100000
```

### Opsi B: Menjalankan Seluruh Metode Sekaligus
Jika Anda ingin komputer Anda menjalankan ketiga metode (`uniform`, `topology`, `importance`) secara otomatis satu per satu dari awal sampai akhir, Anda bisa menggunakan:
```powershell
python run_all.py
```
*(Perlu diperhatikan: Opsi ini mungkin membutuhkan waktu cukup lama dan memakan RAM yang cukup besar, bergantung pada spesifikasi komputer Anda).*

## 5. Melihat Hasil
Hasil akurasi (JSON) dan *weights* model yang telah di-*training* akan otomatis tersimpan di:
* **Model:** `../../model/exp20/` (disimpan dalam bentuk `.pt` untuk GNN dan `.json` untuk XGBoost)
* **Result (Metrik Evaluasi):** `../../result/exp20/` (disimpan dalam bentuk `_results.json`)

## 6. Menonaktifkan Virtual Environment
Jika Anda sudah selesai melakukan eksperimen dan ingin keluar dari *Virtual Environment*, cukup ketikkan:
```powershell
deactivate
```

## 7. Membersihkan (Menghapus/Reset) Virtual Environment
Jika sewaktu-waktu Anda mengalami *error library* yang bertabrakan atau ingin menghapus *Virtual Environment* secara keseluruhan (mengingat instalasi PyTorch + CUDA memakan *storage* yang cukup besar), Anda bisa menghapusnya dengan langkah berikut:

```powershell
# 1. Pastikan Anda keluar/deaktivasi venv terlebih dahulu
deactivate

# 2. Kembali ke direktori utama 'code'
cd "D:\Projects\Klasifikasi Transasi Kartu Kredit - GraphSAGE\tugas-akhir-klasifikasi-transaksi-kartu-kredit\code"

# 3. Hapus folder .venv beserta seluruh isinya secara paksa (Untuk Windows PowerShell)
Remove-Item -Path .venv -Recurse -Force
```
