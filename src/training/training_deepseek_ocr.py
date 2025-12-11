# src/training/train_deepseek_ocr.py

import os
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoModel, Trainer, TrainingArguments, set_seed
from unsloth import FastVisionModel, is_bf16_supported

from datasets import load_dataset

# Import collator của bạn
from .deepseek_ocr_collator import DeepSeekOCRDataCollator

# ==========================================
# HYPERPARAMETERS (Yêu cầu 2.2 & 2.4.2)
# ==========================================
CONFIG = {
    "MAX_STEPS": 300,  # Số step train (có thể tăng/giảm tùy GPU / thời gian)
    "BATCH_SIZE": 2,  # Batch size per device
    "GRAD_ACCUM": 4,  # Gradient accumulation
    "LEARNING_RATE": 2e-4,
    "LORA_RANK": 16,
    "LORA_ALPHA": 16,
    "SAVE_STEPS": 100,  # Mỗi 100 step đánh giá + save 1 lần
    "LOGGING_STEPS": 10,
}


def load_deepseek_ocr_model(model_dir: str):
    """Load DeepSeek-OCR + gắn LoRA (dùng Unsloth)."""
    os.environ["UNSLOTH_WARN_UNINITIALIZED"] = "0"
    model_dir = Path(model_dir)

    if not model_dir.exists():
        print(f"[INFO] Downloading unsloth/DeepSeek-OCR → {model_dir}")
        snapshot_download("unsloth/DeepSeek-OCR", local_dir=str(model_dir))
    else:
        print(f"[INFO] Found existing model dir:", model_dir)

    # Load 4-bit để tiết kiệm VRAM
    model, tokenizer = FastVisionModel.from_pretrained(
        str(model_dir),
        load_in_4bit=True,
        auto_model=AutoModel,
        trust_remote_code=True,
        unsloth_force_compile=True,
        use_gradient_checkpointing="unsloth",
    )

    # Gắn LoRA
    print("[INFO] Adding LoRA adapters...")
    model = FastVisionModel.get_peft_model(
        model,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        r=CONFIG["LORA_RANK"],
        lora_alpha=CONFIG["LORA_ALPHA"],
        lora_dropout=0.0,
        bias="none",
        use_rslora=False,
    )
    FastVisionModel.for_training(model)

    # Tokenizer settings cho Trainer
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer


def format_sample(sample):
    """
    Hàm format 1 sample → dạng {messages: [...]}
    Chỉ mở ảnh khi map (lazy loading).
    """
    project_root = Path(__file__).resolve().parents[2]

    img_path = project_root / sample["image"]
    text = sample["text"]
    instruction = "<image>\nFree OCR."  # Prompt chuẩn cho DeepSeek-OCR

    # Mở ảnh
    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"[WARN] Error loading {img_path}: {e}")
        image = Image.new("RGB", (100, 100), color="black")

    return {
        "messages": [
            {
                "role": "<|User|>",
                "content": instruction,
                "images": [image],
            },
            {
                "role": "<|Assistant|>",
                "content": text,
            },
        ]
    }


def train(train_jsonl, val_jsonl, output_dir):
    set_seed(42)
    project_root = Path(__file__).resolve().parents[2]
    model_dir = project_root / "data" / "deepseek_ocr_model"
    output_dir = Path(output_dir)

    # 1. Load Model
    print("[INFO] Loading model...")
    model, tokenizer = load_deepseek_ocr_model(str(model_dir))

    # 2. Load Dataset bằng datasets (đỡ tốn RAM)
    print("[INFO] Loading datasets...")
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(train_jsonl),
            "val": str(val_jsonl),
        },
    )

    # Map format → messages (chưa load ảnh vào RAM hết)
    print("[INFO] Formatting datasets...")
    train_ds = dataset["train"].map(
        format_sample,
        remove_columns=["image", "text"],
        desc="Formatting Train",
    )
    val_ds = dataset["val"].map(
        format_sample,
        remove_columns=["image", "text"],
        desc="Formatting Val",
    )

    print("Train samples:", len(train_ds))
    print("Val samples  :", len(val_ds))

    # 3. Collator (xử lý images + tokenize messages)
    print("[INFO] Building collator...")
    data_collator = DeepSeekOCRDataCollator(
        tokenizer=tokenizer,
        model=model,
        image_size=640,
        base_size=1024,
        crop_mode=True,
        train_on_responses_only=True,  # chỉ train trên câu trả lời
    )

    # 4. TrainingArguments
    # LƯU Ý: TẮT bf16/fp16 của Trainer để tránh lỗi "Attempting to unscale FP16 gradients".
    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=CONFIG["BATCH_SIZE"],
        per_device_eval_batch_size=CONFIG["BATCH_SIZE"],
        gradient_accumulation_steps=CONFIG["GRAD_ACCUM"],
        learning_rate=CONFIG["LEARNING_RATE"],
        warmup_steps=10,
        max_steps=CONFIG["MAX_STEPS"],
        logging_steps=CONFIG["LOGGING_STEPS"],
        eval_strategy="steps",
        eval_steps=CONFIG["SAVE_STEPS"],
        save_strategy="steps",
        save_steps=CONFIG["SAVE_STEPS"],
        bf16=False,
        fp16=False,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=2,
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        args=args,
    )

    print("[INFO] Start Training...")
    trainer.train()

    print("[INFO] Saving Model...")
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print("[INFO] Training completed!")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    manifests = root / "data" / "UIT_HWDB_line" / "manifests"
    output = root / "outputs" / "deepseek_ocr_uit_hwdb"

    train(
        manifests / "train.jsonl",
        manifests / "val.jsonl",
        output,
    )
