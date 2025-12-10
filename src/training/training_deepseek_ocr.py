# src/training/train_deepseek_ocr.py
from unsloth import FastVisionModel, is_bf16_supported

import os
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel, Trainer, TrainingArguments, set_seed
from datasets import Dataset

from .deepseek_ocr_collator import DeepSeekOCRDataCollator


# -------------------------------------------------------------
# 1. LOAD MODEL DEEPSEEK-OCR
# -------------------------------------------------------------
def load_deepseek_ocr_model(
    model_dir: str,
    load_in_4bit: bool = False,
):
    """
    Tải DeepSeek-OCR dùng Unsloth.
    - Nếu model_dir chưa tồn tại: snapshot_download từ "unsloth/DeepSeek-OCR".
    - Thêm LoRA adapters để fine-tune.
    """
    os.environ["UNSLOTH_WARN_UNINITIALIZED"] = "0"
    model_dir = Path(model_dir)

    if not model_dir.exists():
        print(f"[INFO] Downloading unsloth/DeepSeek-OCR → {model_dir}")
        snapshot_download(
            "unsloth/DeepSeek-OCR",
            local_dir=str(model_dir),
        )
    else:
        print(f"[INFO] Found existing model dir: {model_dir}")

    print("[INFO] Loading DeepSeek-OCR model...")
    model, tokenizer = FastVisionModel.from_pretrained(
        str(model_dir),
        load_in_4bit=load_in_4bit,
        auto_model=AutoModel,
        trust_remote_code=True,
        unsloth_force_compile=True,
        use_gradient_checkpointing="unsloth",
    )

    print("[INFO] Adding LoRA adapters...")
    model = FastVisionModel.get_peft_model(
        model,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        r=16,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    # Bật chế độ training
    FastVisionModel.for_training(model)

    # Required by Trainer
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Kiểu dữ liệu
    if is_bf16_supported():
        model = model.to(dtype=torch.bfloat16)
    else:
        model = model.to(dtype=torch.float16)

    # Đưa lên GPU nếu có
    if torch.cuda.is_available():
        model = model.cuda()

    return model, tokenizer


# -------------------------------------------------------------
# 2. BUILD CONVERSATION DATASET TỪ JSONL
# -------------------------------------------------------------
def build_conversation_dataset_from_jsonl(jsonl_path: str, data_root: str):
    """
    Convert train.jsonl / val.jsonl thành HuggingFace Dataset dạng:
    {"messages": [ {role, content, images?}, ... ]}
    """
    import json
    from PIL import Image

    data_root = Path(data_root)
    samples = []

    # Prompt dùng thống nhất cho tất cả mẫu
    instruction = "<image>\nFree OCR."

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            image_rel = obj["image"]          # ví dụ: "train_data/73/42.jpg"
            text = obj["text"]

            img_path = data_root / image_rel
            image = Image.open(img_path).convert("RGB")

            messages = [
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

            samples.append({"messages": messages})

    return Dataset.from_list(samples)


# -------------------------------------------------------------
# 3. HÀM TRAIN CHÍNH
# -------------------------------------------------------------
def train_deepseek_ocr_on_uit_hwdb(
    train_jsonl: str,
    val_jsonl: str,
    data_root: str,
    output_dir: str,
):
    """
    Fine-tune DeepSeek-OCR trên dataset UIT_HWDB_line.

    - train_jsonl, val_jsonl: đường dẫn tới manifest .jsonl
    - data_root: thư mục gốc chứa train_data/, test_data/
    - output_dir: nơi lưu checkpoint sau fine-tune
    """
    set_seed(42)

    data_root_path = Path(data_root)
    output_dir = Path(output_dir)

    # Model được lưu trong: <data_root.parent>/deepseek_ocr  (tức data/deepseek_ocr)
    model_dir = data_root_path.parent / "deepseek_ocr_model"

    print("[INFO] Model dir       :", model_dir)
    print("[INFO] Train manifest  :", train_jsonl)
    print("[INFO] Val manifest    :", val_jsonl)
    print("[INFO] Output dir      :", output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading model...")
    model, tokenizer = load_deepseek_ocr_model(str(model_dir))

    print("[INFO] Building datasets...")
    train_ds = build_conversation_dataset_from_jsonl(train_jsonl, str(data_root_path))
    val_ds   = build_conversation_dataset_from_jsonl(val_jsonl, str(data_root_path))

    print("Train samples:", len(train_ds))
    print("Val samples  :", len(val_ds))

    print("[INFO] Building collator...")
    data_collator = DeepSeekOCRDataCollator(
        tokenizer=tokenizer,
        model=model,
        image_size=640,
        base_size=1024,
        crop_mode=True,
        train_on_responses_only=True,
    )

    args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_steps=5,
    max_steps=60,
    logging_steps=10,
    
    # dùng tên mới:
    eval_strategy="steps",
    eval_steps=60,
    
    # khai báo rõ luôn chiến lược save:
    save_strategy="steps",
    save_steps=60,

    bf16=is_bf16_supported(),
    fp16=not is_bf16_supported(),
    remove_unused_columns=False,
    report_to="none",
    )


    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        args=args,
    )

    print("[INFO] Start training...")
    trainer.train()

    print("[INFO] Saving checkpoint...")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print("[INFO] Training completed!")


# -------------------------------------------------------------
# 4. ENTRYPOINT: CHẠY TRỰC TIẾP MODULE NÀY
# -------------------------------------------------------------
if __name__ == "__main__":
    # project_root: thư mục gốc chứa src/, data/, README.md
    project_root = Path(__file__).resolve().parents[2]

    data_root = project_root / "data" / "UIT_HWDB_line"
    manifest_dir = data_root / "manifests"

    train_jsonl = manifest_dir / "train.jsonl"
    val_jsonl   = manifest_dir / "val.jsonl"

    output_dir = project_root / "outputs" / "deepseek_ocr_uit_hwdb"

    train_deepseek_ocr_on_uit_hwdb(
        train_jsonl=str(train_jsonl),
        val_jsonl=str(val_jsonl),
        data_root=str(data_root),
        output_dir=str(output_dir),
    )
