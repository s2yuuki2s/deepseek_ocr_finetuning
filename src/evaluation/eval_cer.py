# src/evaluation/eval_cer.py

import csv
import json
from pathlib import Path

import pandas as pd
from jiwer import cer, wer
from tqdm import tqdm

from src.inference.deepseek_ocr_inference import DeepSeekOCRInference

# Config
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT
TEST_MANIFEST = DATA_ROOT / "data/UIT_HWDB_line/manifests/test.jsonl"
OUTPUT_REPORT = PROJECT_ROOT / "outputs/evaluation_report.csv"


def evaluate_model(model_path, num_samples=None):
    print(f"\n[INFO] Evaluating model: {model_path}")

    # Init Model
    try:
        # Nếu path không tồn tại (chưa train xong), dùng model gốc
        if not Path(model_path).exists() and "/" not in model_path:
            print(f"Path {model_path} not found.")
            return None

        infer_engine = DeepSeekOCRInference(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    # Load Data
    samples = []
    with open(TEST_MANIFEST, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))

    if num_samples:
        samples = samples[:num_samples]

    results = []
    gts = []
    preds = []

    for sample in tqdm(samples, desc="Inference"):
        img_rel = sample["image"]
        gt_text = sample["text"]
        img_full_path = PROJECT_ROOT / img_rel

        if not img_full_path.exists():
            continue

        # Predict
        try:
            pred_text = infer_engine.predict(str(img_full_path))
        except:
            pred_text = ""

        # Calc metric per sample
        # DeepSeek đôi khi trả về nhiều dòng, ta gộp lại để tính CER
        sample_cer = cer(gt_text, pred_text)

        gts.append(gt_text)
        preds.append(pred_text)

        results.append(
            {
                "image": img_rel,
                "ground_truth": gt_text,
                "prediction": pred_text,
                "cer": sample_cer,
            }
        )

    # Overall Metrics
    total_cer = cer(gts, preds)
    total_wer = wer(gts, preds)

    print(f"  -> Global CER: {total_cer:.4%}")
    print(f"  -> Global WER: {total_wer:.4%}")

    return results, total_cer, total_wer


def main():
    # 1. Eval Original Model
    print("=== EVALUATION ORIGINAL ===")
    res_orig, cer_orig, wer_orig = evaluate_model(
        "unsloth/DeepSeek-OCR", num_samples=50
    )  # Demo 50 mẫu

    # 2. Eval Fine-tuned Model
    ft_path = str(PROJECT_ROOT / "outputs/deepseek_ocr_uit_hwdb")
    print("\n=== EVALUATION FINE-TUNED ===")
    res_ft, cer_ft, wer_ft = evaluate_model(ft_path, num_samples=50)

    # 3. Compare & Save Report (Error Analysis - Req 2.3)
    if res_ft:
        df = pd.DataFrame(res_ft)
        # Sắp xếp theo CER giảm dần (những câu sai nhiều nhất)
        df_sorted = df.sort_values(by="cer", ascending=False)

        # Lưu file để viết báo cáo
        OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
        df_sorted.to_csv(OUTPUT_REPORT, index=False)
        print(f"\n[INFO] Detailed error report saved to {OUTPUT_REPORT}")

        print("\n=== IMPROVEMENT ===")
        print(f"CER Reduction: {cer_orig - cer_ft:.4%}")


if __name__ == "__main__":
    main()
