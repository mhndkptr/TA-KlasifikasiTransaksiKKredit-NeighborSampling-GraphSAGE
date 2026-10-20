import os
import sys
import subprocess

if __name__ == "__main__":
    if os.path.exists("/kaggle/input"):
        config = "config.kaggle.yaml"
    elif os.path.exists("/content"):
        config = "config.colab.yaml"
    else:
        config = "config.reference.yaml"

    cmd = [sys.executable, "main.py", "--config", config] + sys.argv[1:]
    sys.exit(subprocess.run(cmd).returncode)
