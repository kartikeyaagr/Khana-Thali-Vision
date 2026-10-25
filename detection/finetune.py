"""
Fine-tunes YOLOv8n on the annotated thali compartment images.

What this script does:
  1. Splits the 19 Roboflow images 80/20 into train/val
  2. Offline-augments train set 10× using albumentations (16 → ~176 images)
  3. Writes a corrected data.yaml with absolute paths
  4. Trains with YOLO online augmentation on top

Run:
    uv run finetune.py

Output:
    runs/finetune/weights/best.pt   ← use this in run.py / debug_detect.py
"""

import random
import shutil
import yaml
from pathlib import Path

import cv2
import albumentations as A
import torch
from ultralytics import YOLO

# ── Paths ──────────────────────────────────────────────────────────────────────
FINETUNE_DIR = Path("detection/finetune_images")
TRAIN_IMGS   = FINETUNE_DIR / "train" / "images"
TRAIN_LBLS   = FINETUNE_DIR / "train" / "labels"
SPLIT_DIR    = Path("finetune_split")   # where we write the train/val split
RUNS_DIR     = Path("runs/finetune")

VAL_FRACTION = 0.20
SEED         = 42


def make_split():
    """Split train images 80/20 into finetune_split/train and finetune_split/val."""
    # Always start clean so stale augmented files don't accumulate across runs
    if SPLIT_DIR.exists():
        shutil.rmtree(SPLIT_DIR)

    images = sorted(TRAIN_IMGS.glob("*.jpg")) + sorted(TRAIN_IMGS.glob("*.png"))
    random.seed(SEED)
    random.shuffle(images)

    n_val   = max(1, int(len(images) * VAL_FRACTION))
    val_set = set(img.stem for img in images[:n_val])

    for split in ("train", "val"):
        (SPLIT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (SPLIT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    for img_path in images:
        split = "val" if img_path.stem in val_set else "train"
        shutil.copy(img_path, SPLIT_DIR / split / "images" / img_path.name)

        lbl_path = TRAIN_LBLS / (img_path.stem + ".txt")
        if lbl_path.exists():
            shutil.copy(lbl_path, SPLIT_DIR / split / "labels" / lbl_path.name)

    n_train = len(images) - n_val
    print(f"Split: {n_train} train / {n_val} val")
    return n_train, n_val


def augment_train_set(n_per_image: int = 10):
    """
    Generate n_per_image augmented copies of each train image in-place.

    Bounding boxes are transformed alongside images using albumentations YOLO
    format. Boxes that become < 30% visible after transform are dropped.
    Val set is never touched.
    """
    img_dir = SPLIT_DIR / "train" / "images"
    lbl_dir = SPLIT_DIR / "train" / "labels"

    # Full 360° rotation — thali is circular so any orientation is valid.
    # Reflection border prevents black triangles at corners that confuse YOLO.
    transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=180, border_mode=cv2.BORDER_REFLECT_101, p=0.9),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=40, val_shift_limit=30, p=0.7),
            A.Blur(blur_limit=3, p=0.2),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.3,   # drop boxes that become mostly clipped
        ),
    )

    originals = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))

    for img_path in originals:
        img = cv2.imread(str(img_path))
        lbl_path = lbl_dir / (img_path.stem + ".txt")

        bboxes, class_labels = [], []
        if lbl_path.exists():
            for line in lbl_path.read_text().strip().splitlines():
                parts = line.split()
                class_labels.append(int(parts[0]))
                bboxes.append([float(x) for x in parts[1:5]])

        for i in range(n_per_image):
            result = transform(image=img, bboxes=bboxes, class_labels=class_labels)
            stem = f"{img_path.stem}_aug{i:02d}"

            cv2.imwrite(str(img_dir / (stem + img_path.suffix)), result["image"])

            aug_lbl = lbl_dir / (stem + ".txt")
            if result["bboxes"]:
                rows = [
                    f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                    for cls, (cx, cy, w, h) in zip(result["class_labels"], result["bboxes"])
                ]
                aug_lbl.write_text("\n".join(rows))
            else:
                aug_lbl.write_text("")   # background image — still valid for training

    n_total = len(list(img_dir.glob("*.jpg"))) + len(list(img_dir.glob("*.png")))
    print(f"Augmented: {len(originals)} originals → {n_total} train images")


def write_yaml():
    """Write data.yaml with absolute paths so YOLO can find the splits anywhere."""
    cfg = {
        "path":  str(SPLIT_DIR.resolve()),
        "train": "train/images",
        "val":   "val/images",
        "nc":    1,
        "names": ["compartment"],
    }
    yaml_path = SPLIT_DIR / "data.yaml"
    yaml_path.write_text(yaml.dump(cfg, default_flow_style=False))
    return yaml_path


def train(yaml_path: Path, n_train: int):
    model = YOLO("yolov8n.pt")

    model.train(
        data=str(yaml_path.resolve()),
        project=str(RUNS_DIR.parent.resolve()),
        name=RUNS_DIR.name,
        exist_ok=True,

        # ── Core ──────────────────────────────────────────────────────────
        epochs=150,
        imgsz=416,
        batch=16,           # ~176 train images after augmentation
        workers=2,

        # ── Regularisation (important for small datasets) ─────────────────
        dropout=0.1,
        weight_decay=0.0005,
        patience=30,        # early stopping

        # ── Learning rate ─────────────────────────────────────────────────
        lr0=0.001,          # lower than default (0.01) — gentle on pre-trained weights
        lrf=0.01,
        warmup_epochs=5,

        # ── Augmentation tuned for top-down thali images ──────────────────
        # Thali can appear at any orientation → enable all flips and rotation
        fliplr=0.5,
        flipud=0.5,
        degrees=45.0,       # random rotation up to 45°
        # Colour augmentation — food colours vary across images
        hsv_h=0.02,
        hsv_s=0.5,
        hsv_v=0.3,
        # Mosaic combines 4 images — good for increasing effective dataset size
        mosaic=1.0,
        # Scale and translate
        scale=0.4,
        translate=0.1,

        # ── Misc ──────────────────────────────────────────────────────────
        amp=False,          # AMP grad scaler hangs on MPS
        device=0 if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
        verbose=True,
        plots=True,         # saves training curves to runs/finetune/
    )

    best_src = RUNS_DIR.resolve() / "weights" / "best.pt"
    best_dst = Path("models/thali_detector.pt")
    best_dst.parent.mkdir(parents=True, exist_ok=True)

    if best_src.exists():
        shutil.copy(best_src, best_dst)
        print(f"\nDone. Weights copied → {best_dst.resolve()}")
        print(f"Use with: uv run python -m detection.debug_detect --yolo {best_dst}")
    else:
        print(f"\n[warn] best.pt not found at {best_src} — check {RUNS_DIR.resolve()}")


def main():
    n_train, n_val = make_split()
    augment_train_set(n_per_image=10)   # 16 originals × 10 = 160 extra → ~176 total
    yaml_path = write_yaml()
    train(yaml_path, n_train)


if __name__ == "__main__":
    main()
