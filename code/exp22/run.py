import os
import sys

print("="*75)
print("  MENJALANKAN EXP22: PROPOSED MAX POOLING AGGREGATOR (SAGEConv aggr='max')")
print("="*75)

# Pastikan PyG terinstall jika di cloud
try:
    import torch_geometric
    print(f"PyTorch Geometric version: {torch_geometric.__version__}")
except ImportError:
    print("Menginstall torch_geometric...")
    os.system("pip install torch_geometric")

config_file = "config.kaggle.yaml" if os.path.exists("/kaggle/input") else ("config.colab.yaml" if os.path.exists("/content") else "config.reference.yaml")
cmd = f"{sys.executable} main.py --config {config_file} --aggr max"
print(f"Menjalankan perintah: {cmd}")
ret = os.system(cmd)
if ret != 0:
    print(f"[ERROR] Eksekusi gagal dengan kode {ret}")
    sys.exit(ret)
else:
    print("\n[SUCCESS] Eksperimen proposed max pooling EXP22 selesai dieksekusi!")
