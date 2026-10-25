"""
Label-only Precision/Recall evaluation for Thali Detection.

Compares predicted labels (from predictions.json) against ground truth labels.
Bounding box accuracy is ignored — only label presence per image is evaluated.

Usage:
    uv run evaluate.py --predictions outputs/predictions.json --ground-truth data/ground_truth.json

Ground truth JSON format (two options accepted):
    Option A — same format as predictions.json:
        {"image1.jpg": [{"label": "rice"}, {"label": "dal"}], ...}

    Option B — simple list of labels per image:
        {"image1.jpg": ["rice", "dal"], ...}
"""

import argparse
import json
from pathlib import Path


def load_label_sets(path: str) -> dict[str, set[str]]:
    """Load JSON and return {image_name: set_of_labels}."""
    data = json.loads(Path(path).read_text())
    result = {}
    for img, items in data.items():
        if items and isinstance(items[0], dict):
            result[img] = {d["label"] for d in items}
        else:
            result[img] = set(items)
    return result


def evaluate(predictions: dict[str, set], ground_truth: dict[str, set]):
    total_tp = total_fp = total_fn = 0
    per_image = {}

    for img, gt_labels in ground_truth.items():
        pred_labels = predictions.get(img, set())

        tp = len(pred_labels & gt_labels)
        fp = len(pred_labels - gt_labels)
        fn = len(gt_labels - pred_labels)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        per_image[img] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "predicted": sorted(pred_labels),
            "ground_truth": sorted(gt_labels),
            "missed": sorted(gt_labels - pred_labels),
            "extra": sorted(pred_labels - gt_labels),
        }

    # Micro-average (aggregate TP/FP/FN across all images)
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1        = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)
                       if (micro_precision + micro_recall) > 0 else 0.0)

    # Macro-average (mean of per-image metrics)
    n = len(per_image)
    macro_precision = sum(v["precision"] for v in per_image.values()) / n if n else 0.0
    macro_recall    = sum(v["recall"]    for v in per_image.values()) / n if n else 0.0
    macro_f1        = sum(v["f1"]        for v in per_image.values()) / n if n else 0.0

    return {
        "micro": {
            "precision": round(micro_precision, 4),
            "recall":    round(micro_recall,    4),
            "f1":        round(micro_f1,        4),
            "tp": total_tp, "fp": total_fp, "fn": total_fn,
        },
        "macro": {
            "precision": round(macro_precision, 4),
            "recall":    round(macro_recall,    4),
            "f1":        round(macro_f1,        4),
        },
        "per_image": per_image,
    }


def main():
    p = argparse.ArgumentParser(description="Label-only Precision/Recall evaluation")
    p.add_argument("--predictions",   required=True, help="Path to predictions.json")
    p.add_argument("--ground-truth",  required=True, help="Path to ground_truth.json")
    p.add_argument("--output",        default=None,  help="Save full results to this JSON file")
    p.add_argument("--verbose", "-v", action="store_true", help="Show per-image breakdown")
    args = p.parse_args()

    preds = load_label_sets(args.predictions)
    gt    = load_label_sets(args.ground_truth)

    # Warn about images in GT but not in predictions
    missing = set(gt) - set(preds)
    if missing:
        print(f"[warn] {len(missing)} GT images have no predictions: {sorted(missing)[:5]} ...")

    results = evaluate(preds, gt)

    print("\n=== Micro-averaged (aggregate) ===")
    m = results["micro"]
    print(f"  Precision : {m['precision']:.4f}")
    print(f"  Recall    : {m['recall']:.4f}")
    print(f"  F1        : {m['f1']:.4f}")
    print(f"  TP={m['tp']}  FP={m['fp']}  FN={m['fn']}")

    print("\n=== Macro-averaged (per-image mean) ===")
    m = results["macro"]
    print(f"  Precision : {m['precision']:.4f}")
    print(f"  Recall    : {m['recall']:.4f}")
    print(f"  F1        : {m['f1']:.4f}")

    if args.verbose:
        print("\n=== Per-image breakdown ===")
        for img, v in sorted(results["per_image"].items()):
            print(f"\n  {img}")
            print(f"    P={v['precision']:.3f}  R={v['recall']:.3f}  F1={v['f1']:.3f}")
            if v["missed"]:
                print(f"    missed : {v['missed']}")
            if v["extra"]:
                print(f"    extra  : {v['extra']}")

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
        print(f"\nFull results saved to {args.output}")


if __name__ == "__main__":
    main()
