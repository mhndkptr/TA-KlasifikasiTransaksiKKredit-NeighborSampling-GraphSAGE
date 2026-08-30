# Experiment 2 - GPU-Bound GraphSAGE

Experiment 2 memindahkan bagian utama training ke GPU: fitur node, label,
struktur adjacency CSR, neighbor sampling, pembentukan subgraf, forward, dan
backward. Preprocessing CSV dengan pandas tetap berjalan di CPU sebelum
training dimulai.

## Menjalankan

```powershell
cd code/exp2_gpu_bound
python run.py --config config.yaml --strategy uniform --seed 42 --max-rows 10000
```

Setelah smoke test berhasil, naikkan `max_rows` dan `batch_size` secara
bertahap. Pantau GPU menggunakan `nvidia-smi -l 1`.

## Perbedaan dari Experiment 1

- Seluruh graph tensor yang diperlukan training disimpan di VRAM.
- Sampling memakai CSR dan operasi PyTorch CUDA, bukan loop NumPy per node.
- Automatic mixed precision (AMP) aktif secara default.
- Batch default dinaikkan menjadi 4.096 root transaction.
- Log mencatat penggunaan VRAM allocated/reserved setiap epoch.
- Nama checkpoint dan hasil memakai awalan `exp2_`, sehingga tidak menimpa
  artefak Experiment 1.

## Trade-off

GPU-bound bukan selalu berarti lebih baik untuk semua ukuran data:

- kebutuhan VRAM meningkat karena fitur, CSR, bobot edge, dan sampled subgraph
  berada di GPU;
- seluruh 24,3 juta transaksi mungkin tidak muat pada GPU konsumen;
- ketiga strategi menggunakan candidate oversampling dengan Gumbel top-k agar
  jalur komputasinya setara; mekanisme ini merupakan aproksimasi sampling
  tanpa penggantian, termasuk untuk baseline uniform pada node berderajat
  lebih besar daripada fan-out;
- candidate yang sama dapat terambil lebih dari sekali sebelum edge akhir
  dideduplikasi;
- preprocessing dan perhitungan metrik scikit-learn masih memakai CPU;
- batch yang terlalu besar dapat menyebabkan CUDA out-of-memory;
- hasil Experiment 2 tidak boleh digabung langsung dengan Experiment 1 tanpa
  melaporkan perbedaan implementasi sampler.

Parameter `proposal_factor` mengatur kualitas aproksimasi. Nilai lebih besar
memberi lebih banyak kandidat kepada sampler, tetapi memakai lebih banyak VRAM
dan komputasi. Nilai awal `4` merupakan kompromi praktis.

## Saran tuning

1. Mulai dengan `--max-rows 10000`.
2. Naikkan ke 200.000 dan periksa VRAM.
3. Naikkan `batch_size` sampai GPU sibuk tanpa OOM.
4. Jika OOM, turunkan `batch_size`, `proposal_factor`, atau `max_rows`.
5. Gunakan konfigurasi identik untuk semua strategi dan seed.
