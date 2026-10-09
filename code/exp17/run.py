import os

print("Files in /kaggle/input:")
os.system("ls -R /kaggle/input/")

print("Installing torch_geometric...")
os.system("pip install torch_geometric")

print("Starting training exp17 sequentially for all sampling methods...")
methods = ["uniform", "topology", "importance"]
for method in methods:
    print(f"\\n--- Running sampling method: {method} ---")
    os.system(f"python /kaggle/input/exp17-graphsage-code/main.py --config /kaggle/input/exp17-graphsage-code/config.kaggle.yaml --sampling {method}")

print("Training finished! Check the output files.")