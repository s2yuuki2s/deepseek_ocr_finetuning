# src/evaluation/eval_cer.py

import json
from pathlib import Path
from jiwer import cer, wer
from tqdm import tqdm
import torch

from src.inference.deepseek_ocr_inference import DeepSeekOCRInference


class SimpleUITDataset:
    """Simple dataset for UIT_HWDB_line, only for evaluation."""
    
    def __init__(self, manifest_path, data_root):
        self.manifest_path = Path(manifest_path)
        self.data_root = Path(data_root)
        self.samples = []
        
        with open(self.manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.samples.append(obj)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = self.data_root / sample["image"]
        text = sample["text"]
        return str(img_path), text


def run_ocr_on_dataset(model_wrapper, dataset, max_samples=None):
    """Run OCR inference on entire dataset."""
    all_gt = []
    all_pred = []
    
    indices = range(len(dataset))
    if max_samples:
        indices = indices[:max_samples]
    
    for idx in tqdm(indices, desc="Processing samples"):
        try:
            img_path, text_gt = dataset[idx]
            text_pred = model_wrapper.predict(img_path)
            all_gt.append(text_gt)
            all_pred.append(text_pred)
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
    
    return all_gt, all_pred


def calculate_metrics(gt_texts, pred_texts):
    """Calculate CER and WER."""
    cer_score = cer(gt_texts, pred_texts)
    wer_score = wer(gt_texts, pred_texts)
    
    return {
        "CER": cer_score,
        "WER": wer_score,
        "num_samples": len(gt_texts)
    }


def main():
    project_root = Path(__file__).resolve().parents[2]
    data_root = project_root / "data/UIT_HWDB_line"
    test_manifest = data_root / "manifests" / "test.jsonl"
    
    # Load test dataset
    test_dataset = SimpleUITDataset(test_manifest, data_root)
    print(f"Test samples: {len(test_dataset)}")
    
    # Limit samples for quick evaluation (có thể điều chỉnh)
    max_eval_samples = min(100, len(test_dataset))
    
    # 1. Evaluate original model
    print("\n" + "="*50)
    print("Evaluating ORIGINAL DeepSeek-OCR model...")
    print("="*50)
    
    orig_model = DeepSeekOCRInference("unsloth/DeepSeek-OCR")
    gt_orig, pred_orig = run_ocr_on_dataset(
        orig_model, test_dataset, max_samples=max_eval_samples
    )
    
    orig_metrics = calculate_metrics(gt_orig, pred_orig)
    
    # 2. Evaluate fine-tuned model
    print("\n" + "="*50)
    print("Evaluating FINE-TUNED DeepSeek-OCR model...")
    print("="*50)
    
    ft_checkpoint = project_root / "outputs/deepseek_ocr_uit_hwdb"
    if ft_checkpoint.exists():
        ft_model = DeepSeekOCRInference(str(ft_checkpoint))
        gt_ft, pred_ft = run_ocr_on_dataset(
            ft_model, test_dataset, max_samples=max_eval_samples
        )
        
        ft_metrics = calculate_metrics(gt_ft, pred_ft)
    else:
        print(f"Fine-tuned checkpoint not found at {ft_checkpoint}")
        ft_metrics = None
    
    # 3. Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    print(f"\nOriginal Model:")
    print(f"  CER: {orig_metrics['CER']:.4%}")
    print(f"  WER: {orig_metrics['WER']:.4%}")
    print(f"  Samples: {orig_metrics['num_samples']}")
    
    if ft_metrics:
        print(f"\nFine-tuned Model:")
        print(f"  CER: {ft_metrics['CER']:.4%}")
        print(f"  WER: {ft_metrics['WER']:.4%}")
        print(f"  Samples: {ft_metrics['num_samples']}")
        
        print(f"\nImprovement:")
        print(f"  CER Reduction: {(orig_metrics['CER'] - ft_metrics['CER']):.4%}")
        print(f"  WER Reduction: {(orig_metrics['WER'] - ft_metrics['WER']):.4%}")
    
    # 4. Show some examples
    print("\n" + "="*50)
    print("EXAMPLE PREDICTIONS")
    print("="*50)
    
    for i in range(min(3, len(gt_orig))):
        print(f"\nSample {i+1}:")
        print(f"  Ground Truth: {gt_orig[i]}")
        print(f"  Original Pred: {pred_orig[i]}")
        if ft_metrics and i < len(pred_ft):
            print(f"  Fine-tuned Pred: {pred_ft[i]}")


if __name__ == "__main__":
    main()