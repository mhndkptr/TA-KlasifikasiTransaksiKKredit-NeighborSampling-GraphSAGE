import os
import sys
import subprocess

if __name__ == "__main__":
    print("="*70)
    print("  MEMULAI RUN CEPAT EXP23 (PURE END-TO-END HETEROGRAPHSAGE - UNIFORM)")
    print("="*70)
    
    config_file = "config.reference.yaml"
    if os.path.exists("/kaggle/input"):
        config_file = "config.kaggle.yaml"
    elif os.path.exists("/content"):
        config_file = "config.colab.yaml"
        
    cmd = [sys.executable, "main.py", "--config", config_file, "--sampling", "uniform"]
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])
        
    print(f"Menjalankan perintah: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)
