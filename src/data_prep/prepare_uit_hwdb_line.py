# src/data_prep/prepare_uit_hwdb_line.py
#
# Mục tiêu: THỰC HIỆN ĐÚNG 2.1 CHO UIT_HWDB_line
#
# - Đọc dataset UIT_HWDB_line (train_data, test_data, label.json)
# - Chuẩn hoá text tiếng Việt (Unicode NFC, whitespace)
# - Định nghĩa pipeline tiền xử lý ảnh (cropping, quality enhancement, size normalization)
# - Tổ chức lại dữ liệu -> tạo manifest JSONL: train / val / test
# - Chia train/val + chọn subset train phù hợp tài nguyên
#
# -> Sau file này, bạn đã hoàn thành 2.1 về mặt code.

import json
import random
import unicodedata
import re
from pathlib import Path

from PIL import Image, ImageOps, ImageFilter
import numpy as np

# =========================
# 0. CẤU HÌNH ĐƯỜNG DẪN
# =========================

# PROJECT_ROOT = thư mục chứa "src" và "UIT_HWDB_line"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data/UIT_HWDB_line"

# Nếu dataset của bạn có cấu trúc khác (vd Line-level/train)
# thì sửa lại 2 dòng dưới đây cho khớp.
TRAIN_DIR = DATA_ROOT / "train_data"
TEST_DIR  = DATA_ROOT / "test_data"

MANIFEST_DIR = DATA_ROOT / "manifests"
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

MAX_TRAIN_SAMPLES = 8000    # giới hạn số mẫu train (subset)
VAL_RATIO = 0.15            # 15% train_data -> validation

random.seed(42)

# =========================
# 1. TEXT NORMALIZATION
# =========================

def normalize_text(s: str) -> str:
    """
    Chuẩn hoá văn bản tiếng Việt:
      - Đưa về Unicode NFC
      - Trim hai đầu
      - Gộp nhiều khoảng trắng liên tiếp
    """
    if s is None:
        return ""
    s = unicodedata.normalize("NFC", s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

# =========================
# 2. PIPELINE TIỀN XỬ LÝ ẢNH
# =========================

def crop_to_content(img: Image.Image, margin: int = 4) -> Image.Image:
    """
    Cropping:
      - Tìm vùng pixel tối (chữ) so với nền trắng
      - Lấy bounding box nhỏ nhất + margin
    """
    arr = np.array(img)
    mask = arr < 240
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


def preprocess_image_for_ocr(
    img_path: Path,
    target_height: int = 64,
    apply_denoise: bool = False,
    margin: int = 4,
) -> Image.Image:
    """
    Pipeline ảnh (phục vụ minh hoạ 2.1 & dùng lại ở bước 2.2):
      1. Grayscale
      2. Cropping (crop_to_content)
      3. Autocontrast (+ blur nhẹ nếu apply_denoise=True)
      4. Resize về chiều cao cố định target_height (giữ tỉ lệ)
    """
    img = Image.open(img_path).convert("L")  # grayscale

    # 2. Cropping
    img = crop_to_content(img, margin=margin)

    # 3. Quality enhancement
    img = ImageOps.autocontrast(img)
    if apply_denoise:
        img = img.filter(ImageFilter.GaussianBlur(0.3))

    # 4. Size normalization
    w, h = img.size
    if h <= 0:
        raise ValueError(f"Invalid image height for {img_path}")

    new_w = int(w * (target_height / h))
    new_w = max(new_w, 1)
    img = img.resize((new_w, target_height), Image.BILINEAR)

    return img

# =========================
# 3. LOAD DỮ LIỆU TỪ train_data / test_data
# =========================

def load_samples_from_split(split_dir: Path):
    """
    Duyệt các thư mục con trong split_dir (vd: train_data/1, train_data/2, ...),
    đọc label.json, normalize text, ghép với đường dẫn ảnh.
    """
    all_samples = []

    if not split_dir.exists():
        print(f"[WARN] Split directory not found: {split_dir}")
        return all_samples

    for subdir in sorted(split_dir.iterdir()):
        if not subdir.is_dir():
            continue

        label_file = subdir / "label.json"
        if not label_file.is_file():
            continue

        try:
            with label_file.open("r", encoding="utf-8") as f:
                labels = json.load(f)  # {"1.jpg": "text", ...}
        except json.JSONDecodeError as e:
            print(f"[ERROR] JSON error in {label_file}: {e}")
            continue

        for filename, text in labels.items():
            img_path = subdir / filename
            if not img_path.is_file():
                continue

            text = normalize_text(str(text))
            if not text:
                continue

            rel_path = img_path.relative_to(DATA_ROOT)
            all_samples.append({
                "image_path": str(img_path),
                "rel_image_path": str(rel_path),
                "text": text,
            })

    print(f"[INFO] Loaded {len(all_samples)} samples from {split_dir.name}")
    return all_samples

# =========================
# 4. GHI JSONL
# =========================

def save_jsonl(samples, out_path: Path):
    with out_path.open("w", encoding="utf-8") as f:
        for s in samples:
            obj = {
                "image": s["rel_image_path"],
                "text": s["text"],
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"[INFO] Wrote {len(samples)} lines to {out_path}")

# =========================
# 5. MAIN (CHẠY TOÀN BỘ 2.1)
# =========================

def main():
    print("=== 2.1 – Data preparation for UIT_HWDB_line ===")
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("DATA_ROOT   :", DATA_ROOT)

    # 5.1. Load train_data
    train_samples = load_samples_from_split(TRAIN_DIR)

    # 5.2. Subset train
    random.shuffle(train_samples)
    if MAX_TRAIN_SAMPLES is not None and len(train_samples) > MAX_TRAIN_SAMPLES:
        train_samples = train_samples[:MAX_TRAIN_SAMPLES]
        print(f"[INFO] Use subset of train_data: {len(train_samples)} samples")
    else:
        print(f"[INFO] Use all train_data: {len(train_samples)} samples")

    # 5.3. Split train -> train/val
    n = len(train_samples)
    val_size = int(n * VAL_RATIO)
    val_samples = train_samples[:val_size]
    real_train_samples = train_samples[val_size:]

    print(f"[INFO] Split train_data -> {len(real_train_samples)} train / {len(val_samples)} val")

    # 5.4. Load test_data
    test_samples = load_samples_from_split(TEST_DIR)
    if len(test_samples) == 0:
        print("[INFO] No labeled test_data found.")
    else:
        print(f"[INFO] Using {len(test_samples)} samples as test set.")

    # 5.5. Save manifests
    save_jsonl(real_train_samples, MANIFEST_DIR / "train.jsonl")
    save_jsonl(val_samples,         MANIFEST_DIR / "val.jsonl")
    if test_samples:
        save_jsonl(test_samples,    MANIFEST_DIR / "test.jsonl")

    
if __name__ == "__main__":
    main()
