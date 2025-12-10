# src/utils/viz.py

from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

def show_image_with_text(img_path, text_pred=None, text_gt=None):
    img_path = Path(img_path)
    img = Image.open(img_path).convert("RGB")
    plt.figure(figsize=(8, 3))
    plt.imshow(img)
    plt.axis("off")

    title_lines = []
    if text_gt is not None:
        title_lines.append(f"GT: {text_gt}")
    if text_pred is not None:
        title_lines.append(f"Pred: {text_pred}")

    if title_lines:
        plt.title("\n".join(title_lines), fontsize=10)

    plt.tight_layout()
    plt.show()
