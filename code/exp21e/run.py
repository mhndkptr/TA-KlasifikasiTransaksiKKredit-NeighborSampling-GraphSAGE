import subprocess
import sys

if __name__ == "__main__":
    cmd = [sys.executable, "main.py", "--config", "config.reference.yaml", "--sampling", "uniform"]
    print(f"Menjalankan command: {' '.join(cmd)}")
    subprocess.run(cmd)
