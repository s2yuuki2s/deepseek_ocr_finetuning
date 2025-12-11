# src/inference/deepseek_ocr_inference.py

import os
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from transformers import AutoModel
from unsloth import FastVisionModel, FastVisionModel as FVM  # FVM chỉ để ngắn

# Dùng đúng prompt như khi training
DEFAULT_PROMPT = "<image>\nFree OCR. "


class DeepSeekOCRInference:
    """
    Wrapper đơn giản cho DeepSeek-OCR (base hoặc LoRA fine-tuned) dùng Unsloth.

    - model_path:
        + "unsloth/DeepSeek-OCR"  -> model gốc trên Hugging Face
        + "outputs/deepseek_ocr_uit_hwdb" -> checkpoint LoRA sau khi fine-tune
    """

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        base_size: int = 1024,
        image_size: int = 640,
        crop_mode: bool = True,
    ):
        os.environ["UNSLOTH_WARN_UNINITIALIZED"] = "0"

        self.model_path = model_path
        self.base_size = base_size
        self.image_size = image_size
        self.crop_mode = crop_mode

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Unsloth tự nhận: nếu model_path là LoRA checkpoint nó sẽ load base + adapter
        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_path,
            load_in_4bit=False,  # inference 16-bit cho ổn định
            auto_model=AutoModel,
            trust_remote_code=True,
            unsloth_force_compile=True,  # giống setting trong docs DeepSeek-OCR :contentReference[oaicite:0]{index=0}
            use_gradient_checkpointing="unsloth",
        )

        # Bật chế độ inference (tắt một số thứ của training, bật use_cache, v.v.) :contentReference[oaicite:1]{index=1}
        FVM.for_inference(self.model)
        self.model.to(self.device)

    @torch.no_grad()
    def predict(self, image_path: str, prompt: str = DEFAULT_PROMPT) -> str:
        """
        Chạy OCR cho 1 ảnh, trả về chuỗi text dự đoán.
        """
        image_path = str(Path(image_path))

        # DeepSeek-OCR yêu cầu output_path != '' để lưu file; ta cho 1 thư mục tạm
        tmp_out = Path("outputs/_tmp_infer")
        tmp_out.mkdir(parents=True, exist_ok=True)

        res = self.model.infer(
            self.tokenizer,
            prompt=prompt,
            image_file=image_path,
            output_path=str(tmp_out),
            base_size=self.base_size,
            image_size=self.image_size,
            crop_mode=self.crop_mode,
            save_results=False,  # không cần lưu result.mmd / ảnh bbox
            test_compress=False,
        )

        # Cố gắng lấy text từ kết quả trả về
        if isinstance(res, str):
            return res.strip()

        if isinstance(res, dict):
            for key in ["text", "output_text", "prediction", "result"]:
                value = res.get(key)
                if isinstance(value, str):
                    return value.strip()

        # Fallback cuối cùng: đọc file result.mmd nếu có
        result_mmd = tmp_out / "result.mmd"
        if result_mmd.is_file():
            try:
                txt = result_mmd.read_text(encoding="utf-8", errors="ignore")
                return txt.strip()
            except Exception:
                pass

        return ""

    # Cho tiện dùng trực tiếp từ CLI
    def __call__(self, image_path: str, prompt: str = DEFAULT_PROMPT) -> str:
        return self.predict(image_path, prompt)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run DeepSeek-OCR inference on one image."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="unsloth/DeepSeek-OCR",
        help="Model ID hoặc đường dẫn checkpoint (vd outputs/deepseek_ocr_uit_hwdb).",
    )
    parser.add_argument(
        "--image", type=str, required=True, help="Đường dẫn tới ảnh cần OCR."
    )
    args = parser.parse_args()

    infer = DeepSeekOCRInference(args.model_path)
    text = infer.predict(args.image)
    print("===== OCR RESULT =====")
    print(text)
