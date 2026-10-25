"""
Runs only YOLOv8 detection and saves annotated images to outputs/debug/.

Usage:
    uv run python -m detection.debug_detect
    uv run python -m detection.debug_detect --image data/images/thali1.jpg
    uv run python -m detection.debug_detect --conf 0.03 --max-box 0.20
    uv run python -m detection.debug_detect --clahe --unsharp
"""

import argparse
from pathlib import Path

from thali.detector import ThaliDetector
from thali.visualize import draw_detections


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image", default=None, help="Single image (optional)")
    p.add_argument("--images", default="data/images", help="Folder of images")
    p.add_argument("--conf", type=float, default=0.05, help="YOLO confidence threshold")
    p.add_argument("--nms-iou", type=float, default=0.25, help="NMS IoU threshold")
    p.add_argument(
        "--max-box", type=float, default=0.25, help="Max box area as fraction of image"
    )
    p.add_argument(
        "--min-box", type=float, default=0.01, help="Min box area as fraction of image"
    )
    p.add_argument("--ms-sp", type=int, default=20, help="Mean shift spatial radius")
    p.add_argument("--ms-sr", type=int, default=40, help="Mean shift colour radius")
    p.add_argument("--clahe", action="store_true", help="CLAHE contrast enhancement")
    p.add_argument(
        "--unsharp", action="store_true", help="Unsharp mask (sharpens edges)"
    )
    p.add_argument("--yolo", default="models/thali_detector.pt")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path("outputs/debug")
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = ThaliDetector(
        weights=args.yolo,
        conf=args.conf,
        nms_iou=args.nms_iou,
        max_box_area=args.max_box,
        min_box_area=args.min_box,
        mean_shift_sp=args.ms_sp,
        mean_shift_sr=args.ms_sr,
        use_clahe=args.clahe,
        use_unsharp=args.unsharp,
    )

    if args.image:
        paths = [Path(args.image)]
    else:
        paths = sorted(Path(args.images).glob("*"))
        paths = [p for p in paths if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]

    for img_path in paths:
        detections = detector.detect(str(img_path))
        source = (
            "YOLO"
            if detections and detections[0]["coco_label"] != "region"
            else "contour fallback"
        )
        print(f"\n[{img_path.name}] {len(detections)} regions ({source})")
        for d in detections:
            print(
                f"  {d['coco_label']:<10}  conf={d['yolo_conf']:.2f}  bbox={d['bbox']}"
            )

        for d in detections:
            d["label"] = d["coco_label"]
            d["conf"] = d["yolo_conf"]

        draw_detections(img_path, detections, out_dir / img_path.name)


if __name__ == "__main__":
    main()
