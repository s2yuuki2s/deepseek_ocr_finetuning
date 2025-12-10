# src/datasets/uit_dataset.py

from torch.utils.data import Dataset
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import torch
import json
from pathlib import Path

class UITHandwritingDataset(Dataset):
    """
    Dataset cho UIT_HWDB_line, đọc từ manifest JSONL (train/val/test).

    Preprocessing ảnh:
      - Grayscale
      - Cropping theo nội dung (ink pixels)
      - Autocontrast + optional Gaussian blur
      - Resize về chiều cao cố định (target_height), giữ tỉ lệ
      - Chuẩn hoá [0,1], output tensor (1, H, W)
    """

    def __init__(
        self,
        manifest_path,
        data_root=None,
        target_height=64,
        apply_denoise=False,
        crop_margin=4,
    ):
        manifest_path = Path(manifest_path).resolve()
        if data_root is None:
            # mặc định: data_root = cha của manifests (UIT_HWDB_line)
            data_root = manifest_path.parent.parent
        self.data_root = Path(data_root)

        self.samples = []
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.samples.append(obj)

        self.target_height = target_height
        self.apply_denoise = apply_denoise
        self.crop_margin = crop_margin

    def __len__(self):
        return len(self.samples)

    def _crop_to_content(self, img: Image.Image) -> Image.Image:
        arr = np.array(img)  # (H, W)
        mask = arr < 240
        coords = np.argwhere(mask)

        if coords.size == 0:
            return img

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        y_min = max(int(y_min) - self.crop_margin, 0)
        x_min = max(int(x_min) - self.crop_margin, 0)
        y_max = min(int(y_max) + self.crop_margin, arr.shape[0] - 1)
        x_max = min(int(x_max) + self.crop_margin, arr.shape[1] - 1)

        return img.crop((x_min, y_min, x_max + 1, y_max + 1))

    def _preprocess_image(self, rel_path: str) -> torch.Tensor:
        img_path = self.data_root / rel_path
        img = Image.open(img_path).convert("L")  # grayscale

        # 1) Crop theo nội dung
        img = self._crop_to_content(img)

        # 2) Autocontrast
        img = ImageOps.autocontrast(img)

        # 3) Optional blur
        if self.apply_denoise:
            img = img.filter(ImageFilter.GaussianBlur(0.3))

        # 4) Resize theo target_height
        w, h = img.size
        if h <= 0:
            raise ValueError(f"Invalid image height for {img_path}")
        new_w = int(w * (self.target_height / h))
        new_w = max(new_w, 1)
        img = img.resize((new_w, self.target_height), Image.BILINEAR)

        # 5) [0,1] tensor (1, H, W)
        arr = np.array(img).astype("float32") / 255.0
        arr = np.expand_dims(arr, axis=0)
        return torch.from_numpy(arr)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_tensor = self._preprocess_image(sample["image"])
        text = sample["text"]
        return img_tensor, text
