import os
import requests

CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_URL = "https://huggingface.co/valeoai/Franca/resolve/main/franca_vit_small.pth"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "franca_vit_small.pth")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

if not os.path.exists(CHECKPOINT_PATH):
    print(f"Downloading Franca checkpoint to {CHECKPOINT_PATH}...")
    r = requests.get(CHECKPOINT_URL, stream=True)
    r.raise_for_status()
    with open(CHECKPOINT_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Download complete.")
else:
    print("Checkpoint already exists.")

