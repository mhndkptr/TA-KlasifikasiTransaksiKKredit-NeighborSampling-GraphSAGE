import sys
import subprocess
from pathlib import Path

def main():
    script_dir = Path(__file__).parent.resolve()
    main_script = script_dir / "main.py"
    
    cmd = [sys.executable, str(main_script)] + sys.argv[1:]
    print(f"Menjalankan perintah: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
