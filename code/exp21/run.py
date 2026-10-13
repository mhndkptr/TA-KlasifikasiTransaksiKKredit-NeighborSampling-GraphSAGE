import os
import sys

print("="*70)
print("  MENJALANKAN EXP21: BASELINE UNIFORM SAMPLING (SHINE FRAUD II PROTOCOL)")
print("="*70)

# Pastikan PyG terinstall
try:
    import torch_geometric
    print(f"PyTorch Geometric version: {torch_geometric.__version__}")
except ImportError:
    print("Menginstall torch_geometric...")
    os.system("pip install torch_geometric")

config_file = "config.kaggle.yaml" if os.path.exists("/kaggle/input") else "config.reference.yaml"
cmd = f"{sys.executable} main.py --config {config_file} --sampling uniform"
print(f"Menjalankan perintah: {cmd}")
ret = os.system(cmd)
if ret != 0:
    print(f"[ERROR] Eksekusi gagal dengan kode {ret}")
    sys.exit(ret)
else:
    print("\n[SUCCESS] Eksperimen uniform exp21 selesai dieksekusi!")
