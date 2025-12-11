# src/data_prep/prepare_uit_hwdb_line.py

import json
import random
import re
import unicodedata
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps
from tqdm import tqdm  # Thêm tqdm để thấy tiến độ

# =========================
# CONFIG
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data/UIT_HWDB_line"
TRAIN_DIR = DATA_ROOT / "train_data"
TEST_DIR = DATA_ROOT / "test_data"
MANIFEST_DIR = DATA_ROOT / "manifests"
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

# Yêu cầu 2.1: Subset sampling để phù hợp tài nguyên
MAX_TRAIN_SAMPLES = 5000  # Đặt hợp lý cho Colab T4 (Khoảng 3k-5k là tốt)
VAL_RATIO = 0.1
random.seed(42)


def normalize_text(s: str) -> str:
    """Yêu cầu 2.1: Text normalization (NFC)"""
    if s is None:
        return ""
    s = unicodedata.normalize("NFC", s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def crop_to_content(img: Image.Image, margin: int = 4) -> Image.Image:
    """Yêu cầu 2.1: Preprocessing (Cropping)"""
    arr = np.array(img)
    mask = arr < 240  # Ngưỡng tối để bắt chữ
    coords = np.argwhere(mask)
    if coords.size == 0:
        return img

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    y_min = max(int(y_min) - margin, 0)
    x_min = max(int(x_min) - margin, 0)
    y_max = min(int(y_max) + margin, arr.shape[0] - 1)
    x_max = min(int(x_max) + margin, arr.shape[1] - 1)
    return img.crop((x_min, y_min, x_max + 1, y_max + 1))


def load_samples_from_split(split_dir: Path, desc="Loading"):
    all_samples = []
    if not split_dir.exists():
        print(f"[WARN] Not found: {split_dir}")
        return all_samples

    # Duyệt qua các thư mục con (UIT-HWDB thường chia theo folder người viết)
    subdirs = [d for d in split_dir.iterdir() if d.is_dir()]

    for subdir in tqdm(subdirs, desc=desc):
        label_file = subdir / "label.json"
        if not label_file.is_file():
            continue

        try:
            with label_file.open("r", encoding="utf-8") as f:
                labels = json.load(f)
        except:
            continue

        for filename, text in labels.items():
            img_path = subdir / filename
            if not img_path.is_file():
                continue

            text = normalize_text(str(text))
            if not text:
                continue

            # Lưu đường dẫn tương đối để file jsonl gọn nhẹ
            rel_path = img_path.relative_to(PROJECT_ROOT)
            all_samples.append(
                {
                    "image": str(rel_path),  # Path tương đối từ Project Root
                    "text": text,
                }
            )
    return all_samples


def save_jsonl(samples, out_path: Path):
    with out_path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"[INFO] Saved {len(samples)} samples to {out_path.name}")


def main():
    print("=== DATA PREPARATION (Assignment Req 2.1) ===")

    # 1. Load Data
    train_samples = load_samples_from_split(TRAIN_DIR, "Processing Train Data")
    test_samples = load_samples_from_split(TEST_DIR, "Processing Test Data")

    # 2. Shuffle & Subset (Important for Resources)
    random.shuffle(train_samples)
    if len(train_samples) > MAX_TRAIN_SAMPLES:
        print(
            f"[INFO] Subsetting train data: {len(train_samples)} -> {MAX_TRAIN_SAMPLES}"
        )
        train_samples = train_samples[:MAX_TRAIN_SAMPLES]

    # 3. Split Train/Val
    val_size = int(len(train_samples) * VAL_RATIO)
    val_samples = train_samples[:val_size]
    real_train_samples = train_samples[val_size:]

    # 4. Save
    save_jsonl(real_train_samples, MANIFEST_DIR / "train.jsonl")
    save_jsonl(val_samples, MANIFEST_DIR / "val.jsonl")
    save_jsonl(test_samples, MANIFEST_DIR / "test.jsonl")


if __name__ == "__main__":
    main()
