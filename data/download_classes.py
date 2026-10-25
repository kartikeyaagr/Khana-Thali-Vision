"""
Generates data/classes.txt from your Task 1 training checkpoint.

If your checkpoint contains a class-index mapping (common when training with
torchvision ImageFolder), this script extracts it automatically.

Run:
    uv run data/download_classes.py --checkpoint path/to/model.pt

If your checkpoint does NOT contain class names, you will get instructions on
how to create classes.txt manually from your Task 1 training code.
"""

import argparse
import json
import torch
from pathlib import Path


def extract_from_checkpoint(path: Path) -> list[str] | None:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        return None

    # Common keys used when saving class mappings alongside the model
    for key in ("class_to_idx", "classes", "idx_to_class", "class_names"):
        if key in ckpt:
            mapping = ckpt[key]
            if isinstance(mapping, dict):
                # class_to_idx → {name: idx}  or  idx_to_class → {idx: name}
                if all(isinstance(v, int) for v in mapping.values()):
                    ordered = sorted(mapping.items(), key=lambda x: x[1])
                    return [name for name, _ in ordered]
                else:
                    ordered = sorted(mapping.items(), key=lambda x: int(x[0]))
                    return [name for _, name in ordered]
            if isinstance(mapping, list):
                return mapping

    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", help="Path to Task 1 .pt file")
    args = p.parse_args()

    out = Path(__file__).parent / "classes.txt"

    if args.checkpoint:
        class_names = extract_from_checkpoint(Path(args.checkpoint))
        if class_names:
            out.write_text("\n".join(class_names) + "\n")
            print(f"Extracted {len(class_names)} classes → {out}")
            return
        print("Checkpoint found but no class list inside it.")

    print("""
Could not auto-extract classes. Create data/classes.txt manually from your
Task 1 training code by running something like:

    # If you used torchvision ImageFolder:
    dataset = ImageFolder("path/to/train")
    with open("data/classes.txt", "w") as f:
        f.write("\\n".join(dataset.classes))

    # If you used a custom mapping saved separately:
    import json
    mapping = json.load(open("class_to_idx.json"))
    names = sorted(mapping, key=mapping.__getitem__)
    with open("data/classes.txt", "w") as f:
        f.write("\\n".join(names))

The file must have one class name per line, where line N = class index N.
""")


if __name__ == "__main__":
    main()
