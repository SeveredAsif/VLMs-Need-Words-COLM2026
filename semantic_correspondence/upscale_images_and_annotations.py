"""Upscale all images in JPEGImages/ (4x bicubic) and scale annotations in PairAnnotation/."""

import json
from pathlib import Path
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms.functional as TF

SCALE = 4


def upscale(img: Image.Image, scale: int = SCALE) -> Image.Image:
    tensor = TF.to_tensor(img).unsqueeze(0)
    out = F.interpolate(tensor, scale_factor=scale, mode='bicubic', align_corners=False).clamp(0, 1)
    return TF.to_pil_image(out.squeeze(0))


def scale_annotation(ann: dict, scale: int = SCALE) -> dict:
    ann = ann.copy()
    # Scale image sizes (H, W, C) — only scale H and W
    for key in ("src_imsize", "trg_imsize"):
        h, w, c = ann[key]
        ann[key] = [h * scale, w * scale, c]
    # Scale bounding boxes [x1, y1, x2, y2]
    for key in ("src_bndbox", "trg_bndbox"):
        ann[key] = [v * scale for v in ann[key]]
    # Scale keypoints [[x, y], ...]
    for key in ("src_kps", "trg_kps"):
        ann[key] = [[x * scale, y * scale] for x, y in ann[key]]
    return ann


def upscale_images(root: Path):
    src_dir = root / "JPEGImages"
    dst_dir = root / "JPEGImages4x"

    images = sorted(src_dir.rglob("*.jpg"))
    total = len(images)
    print(f"Found {total} images to upscale")

    for i, src_path in enumerate(images, 1):
        rel = src_path.relative_to(src_dir)
        dst_path = dst_dir / rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        if dst_path.exists():
            continue

        img = Image.open(src_path).convert("RGB")
        up = upscale(img)
        up.save(dst_path, quality=95)

        if i % 50 == 0 or i == total:
            print(f"[images {i}/{total}] {rel}")


def upscale_annotations(root: Path):
    src_dir = root / "PairAnnotation"
    dst_dir = root / "PairAnnotation4x"

    jsons = sorted(src_dir.rglob("*.json"))
    total = len(jsons)
    print(f"Found {total} annotations to scale")

    for i, src_path in enumerate(jsons, 1):
        rel = src_path.relative_to(src_dir)
        dst_path = dst_dir / rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        if dst_path.exists():
            continue

        with open(src_path) as f:
            ann = json.load(f)

        ann = scale_annotation(ann)

        with open(dst_path, "w") as f:
            json.dump(ann, f, indent=4)

        if i % 200 == 0 or i == total:
            print(f"[annotations {i}/{total}] {rel}")


def main():
    root = Path("SPair-71k")
    upscale_images(root)
    upscale_annotations(root)
    print("Done.")


if __name__ == "__main__":
    main()
