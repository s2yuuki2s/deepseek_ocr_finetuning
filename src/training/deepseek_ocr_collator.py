# src/training/deepseek_ocr_collator.py

import io
import math
from dataclasses import dataclass
from typing import Any, List, Dict, Tuple

import torch
from PIL import Image, ImageOps
from torch.nn.utils.rnn import pad_sequence

try:
    # Trường hợp bạn cài sẵn package deepseek_ocr (ít khả năng)
    from deepseek_ocr.modeling_deepseekocr import (
        text_encode,
        BasicImageTransform,
        dynamic_preprocess,
    )
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    from huggingface_hub import snapshot_download

    project_root = Path(__file__).resolve().parents[2]
    pkg_dir = project_root / "data" / "deepseek_ocr_model"
    pkg_dir.mkdir(parents=True, exist_ok=True)

    # Chỉ tải nếu chưa có file modeling_deepseekocr.py
    if not (pkg_dir / "modeling_deepseekocr.py").exists():
        print("[INFO] deepseek_ocr package not found. Downloading unsloth/DeepSeek-OCR code to", pkg_dir)
        snapshot_download(
            "unsloth/DeepSeek-OCR",
            local_dir=str(pkg_dir),
        )

    # Biến thư mục này thành package Python
    init_file = pkg_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text("# package for DeepSeek-OCR model code\n", encoding="utf-8")

    # Thêm parent của package vào sys.path
    parent = pkg_dir.parent  # = data/
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))

    # Bây giờ import lại nhưng với tên package đầy đủ
    from deepseek_ocr_model.modeling_deepseekocr import (
        text_encode,
        BasicImageTransform,
        dynamic_preprocess,
    )

@dataclass
class DeepSeekOCRDataCollator:
    """
    Data collator dành cho DeepSeek-OCR.

    features: list các sample dạng {"messages": [...]}
      - messages là list các dict:
          {"role": "<|User|>", "content": "...", "images": [PIL_Image,...]}
          {"role": "<|Assistant|>", "content": "..."}

    Trả về:
      - input_ids, attention_mask, labels
      - images: list[(images_crop, images_ori)]
      - images_seq_mask: (B, L) bool
      - images_spatial_crop: (sum_images, 2) long
    """
    tokenizer: Any
    model: Any
    image_size: int = 640
    base_size: int = 1024
    crop_mode: bool = True
    image_token_id: int = 128815
    train_on_responses_only: bool = True

    def __init__(
        self,
        tokenizer,
        model,
        image_size: int = 640,
        base_size: int = 1024,
        crop_mode: bool = True,
        train_on_responses_only: bool = True,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.image_size = image_size
        self.base_size = base_size
        self.crop_mode = crop_mode
        self.image_token_id = 128815
        self.dtype = model.dtype
        self.train_on_responses_only = train_on_responses_only

        self.image_transform = BasicImageTransform(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            normalize=True,
        )
        self.patch_size = 16
        self.downsample_ratio = 4

        if getattr(tokenizer, "bos_token_id", None) is not None:
            self.bos_id = tokenizer.bos_token_id
        else:
            self.bos_id = 0
            print(f"[WARN] tokenizer has no bos_token_id, using {self.bos_id}")

    # --------- helpers ---------

    def deserialize_image(self, image_data) -> Image.Image:
        if isinstance(image_data, Image.Image):
            return image_data.convert("RGB")
        elif isinstance(image_data, dict) and "bytes" in image_data:
            image_bytes = image_data["bytes"]
            image = Image.open(io.BytesIO(image_bytes))
            return image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image format: {type(image_data)}")

    def process_image(
        self, image: Image.Image
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[int]], List[int], Tuple[int, int]]:
        images_list: List[torch.Tensor] = []
        images_crop_list: List[torch.Tensor] = []
        images_spatial_crop: List[List[int]] = []

        if self.crop_mode:
            if image.size[0] <= 640 and image.size[1] <= 640:
                crop_ratio = (1, 1)
                images_crop_raw = []
            else:
                images_crop_raw, crop_ratio = dynamic_preprocess(
                    image,
                    min_num=2,
                    max_num=9,
                    image_size=self.image_size,
                    use_thumbnail=False,
                )

            # Global view
            global_view = ImageOps.pad(
                image,
                (self.base_size, self.base_size),
                color=tuple(int(x * 255) for x in self.image_transform.mean),
            )
            images_list.append(self.image_transform(global_view).to(self.dtype))

            width_crop_num, height_crop_num = crop_ratio
            images_spatial_crop.append([width_crop_num, height_crop_num])

            # Local crops nếu có
            if width_crop_num > 1 or height_crop_num > 1:
                for crop_img in images_crop_raw:
                    images_crop_list.append(
                        self.image_transform(crop_img).to(self.dtype)
                    )

            num_queries = math.ceil(
                (self.image_size // self.patch_size) / self.downsample_ratio
            )
            num_queries_base = math.ceil(
                (self.base_size // self.patch_size) / self.downsample_ratio
            )

            tokenized_image = (
                [self.image_token_id] * num_queries_base + [self.image_token_id]
            ) * num_queries_base
            tokenized_image += [self.image_token_id]

            if width_crop_num > 1 or height_crop_num > 1:
                tokenized_image += (
                    [self.image_token_id] * (num_queries * width_crop_num)
                    + [self.image_token_id]
                ) * (num_queries * height_crop_num)
        else:
            crop_ratio = (1, 1)
            images_spatial_crop.append([1, 1])

            if self.base_size <= 640:
                resized = image.resize((self.base_size, self.base_size), Image.LANCZOS)
                images_list.append(self.image_transform(resized).to(self.dtype))
            else:
                global_view = ImageOps.pad(
                    image,
                    (self.base_size, self.base_size),
                    color=tuple(int(x * 255) for x in self.image_transform.mean),
                )
                images_list.append(self.image_transform(global_view).to(self.dtype))

            num_queries = math.ceil(
                (self.base_size // self.patch_size) / self.downsample_ratio
            )
            tokenized_image = (
                [self.image_token_id] * num_queries + [self.image_token_id]
            ) * num_queries
            tokenized_image += [self.image_token_id]

        return images_list, images_crop_list, images_spatial_crop, tokenized_image, crop_ratio

    def process_single_sample(self, messages: List[Dict]) -> Dict[str, Any]:
        """
        Xử lý 1 sample (1 cuộc hội thoại) thành input cho model.
        """

        # --- 1. Gom toàn bộ ảnh từ messages ---
        images: List[Image.Image] = []
        for msg in messages:
            if "images" in msg and msg["images"]:
                for img_data in msg["images"]:
                    if img_data is not None:
                        pil_image = self.deserialize_image(img_data)
                        images.append(pil_image)

        if not images:
            raise ValueError("No images found in sample. Each sample must contain images.")

        tokenized_str: List[int] = []
        images_seq_mask: List[bool] = []
        images_list: List[torch.Tensor] = []
        images_crop_list: List[torch.Tensor] = []
        images_spatial_crop: List[List[int]] = []

        prompt_token_count = -1
        assistant_started = False
        image_idx = 0

        # BOS ở đầu
        tokenized_str.append(self.bos_id)
        images_seq_mask.append(False)

        # --- 2. Duyệt từng message ---
        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "<|Assistant|>":
                if not assistant_started:
                    prompt_token_count = len(tokenized_str)
                    assistant_started = True

                content = f"{content.strip()} {self.tokenizer.eos_token}"

            text_splits = content.split("<image>")

            for i, text_sep in enumerate(text_splits):
                # text
                tokenized_sep = text_encode(
                    self.tokenizer, text_sep, bos=False, eos=False
                )
                tokenized_str.extend(tokenized_sep)
                images_seq_mask.extend([False] * len(tokenized_sep))

                # đến chỗ có <image>
                if i < len(text_splits) - 1:
                    if image_idx >= len(images):
                        raise ValueError(
                            "Data mismatch: found '<image>' but no corresponding image."
                        )

                    image = images[image_idx]
                    (
                        img_list,
                        crop_list,
                        spatial_crop,
                        tok_img,
                        _,
                    ) = self.process_image(image)

                    images_list.extend(img_list)
                    images_crop_list.extend(crop_list)
                    images_spatial_crop.extend(spatial_crop)

                    tokenized_str.extend(tok_img)
                    images_seq_mask.extend([True] * len(tok_img))

                    image_idx += 1

        if image_idx != len(images):
            raise ValueError(
                f"Data mismatch: {len(images)} images but only {image_idx} '<image>' used."
            )

        if not assistant_started:
            # không có assistant → mask hết (coi như toàn bộ là prompt)
            prompt_token_count = len(tokenized_str)

        images_ori = torch.stack(images_list, dim=0)
        images_spatial_crop_tensor = torch.tensor(images_spatial_crop, dtype=torch.long)

        if images_crop_list:
            images_crop = torch.stack(images_crop_list, dim=0)
        else:
            images_crop = torch.zeros(
                (1, 3, self.base_size, self.base_size), dtype=self.dtype
            )

        return {
            "input_ids": torch.tensor(tokenized_str, dtype=torch.long),
            "images_seq_mask": torch.tensor(images_seq_mask, dtype=torch.bool),
            "images_ori": images_ori,
            "images_crop": images_crop,
            "images_spatial_crop": images_spatial_crop_tensor,
            "prompt_token_count": prompt_token_count,
        }

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_data: List[Dict[str, Any]] = []

        for feat in features:
            try:
                processed = self.process_single_sample(feat["messages"])
                batch_data.append(processed)
            except Exception as e:
                print(f"[DataCollator] Error processing sample: {e}")
                continue

        if not batch_data:
            raise ValueError("No valid samples in batch")

        input_ids_list = [item["input_ids"] for item in batch_data]
        images_seq_mask_list = [item["images_seq_mask"] for item in batch_data]
        prompt_token_counts = [item["prompt_token_count"] for item in batch_data]

        input_ids = pad_sequence(
            input_ids_list,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        images_seq_mask = pad_sequence(
            images_seq_mask_list,
            batch_first=True,
            padding_value=False,
        )

        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        labels[images_seq_mask] = -100

        if self.train_on_responses_only:
            for idx, prompt_cnt in enumerate(prompt_token_counts):
                if prompt_cnt > 0:
                    labels[idx, :prompt_cnt] = -100

        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        images_batch = [
            (item["images_crop"], item["images_ori"]) for item in batch_data
        ]
        images_spatial_crop = torch.cat(
            [item["images_spatial_crop"] for item in batch_data],
            dim=0,
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "images": images_batch,
            "images_seq_mask": images_seq_mask,
            "images_spatial_crop": images_spatial_crop,
        }
