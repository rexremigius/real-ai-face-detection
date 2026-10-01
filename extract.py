# extract_images.py: unpack test_images.npz into test_images/ai and test_images/real
from pathlib import Path
import numpy as np
from PIL import Image

data = np.load("test_images.npz")
images, labels = data["images"], data["labels"].astype(int)

out = Path("test_images")
folders = {0: out / "ai", 1: out / "real"}       # 0 = AI, 1 = Real
for f in folders.values():
    f.mkdir(parents=True, exist_ok=True)

for i, (img, label) in enumerate(zip(images, labels)):
    Image.fromarray(img).save(folders[label] / f"{i:04d}.png")

print(f"saved {len(images)} images | ai: {(labels == 0).sum()} | real: {(labels == 1).sum()}")