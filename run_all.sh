#!/usr/bin/env bash
# Runs the whole project end to end: Task 1 (classify) → Task 2 (detect + classify) → Task 3 (BEV + detect).
#
#   ./run_all.sh           inference with the released weights
#   ./run_all.sh --train   retrain the classifier (needs khana/ and a CUDA GPU) and the detector first
set -euo pipefail
cd "$(dirname "$0")"

RELEASE=https://github.com/kartikeyaagr/Khana-Thali-Vision/releases/download/v1.0
CLASSIFIER=models/320/best_ema_hires.pt
DETECTOR=models/thali_detector.pt
TRAIN=false; [[ "${1:-}" == "--train" ]] && TRAIN=true

fetch() {  # download a release asset unless it's already on disk
  [[ -f "$1" ]] && return
  mkdir -p "$(dirname "$1")"
  echo "Downloading $1 ..."
  curl -fL --progress-bar -o "$1.part" "$RELEASE/$(basename "$1")" && mv "$1.part" "$1"
}
has_images() { compgen -G "$1/*.[jJ][pP][gG]" >/dev/null || compgen -G "$1/*.[jJ][pP][eE][gG]" >/dev/null || compgen -G "$1/*.[pP][nN][gG]" >/dev/null; }

if $TRAIN; then uv sync --extra train; else uv sync; fi

echo; echo "== Task 1: Khana dish classification"
if [[ -d khana ]]; then
  [[ -f splits/manifest.json ]] || uv run python classification/prepare_splits.py
  if $TRAIN; then
    uv run python classification/pipeline.py          # 224px → 320px fine-tune → test-set eval
  else
    fetch "$CLASSIFIER"
    uv run python classification/evaluate.py --checkpoint "$CLASSIFIER" --image-size 320
  fi
else
  echo "No khana/ dataset found, so skipping training and test-set evaluation (see README)."
fi
fetch "$CLASSIFIER"

echo; echo "== Task 2: thali compartment detection"
if $TRAIN; then uv run python -m detection.finetune; fi
fetch "$DETECTOR"
uv run python -m detection.run --images data/images --output outputs/task2

echo; echo "== Task 3: top-down (BEV) correction + detection"
BEV_IN=data/angled; has_images "$BEV_IN" || BEV_IN=data/images
uv run python -m bev.bev_pipeline --images "$BEV_IN" --output outputs/task3

echo; echo "Done. Results are in outputs/task2 and outputs/task3."
