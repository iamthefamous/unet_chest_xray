import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data import CLASSES, preprocess_image
from model import OUTPUT_DIR, load_model

PREDICTIONS_DIR = OUTPUT_DIR / "predictions"


def save_overlay(image, mask, path):
    """Save the X-ray with the predicted lungs tinted red, enlarged 2x."""
    gray = (image.numpy() * 255).astype(np.uint8)
    rgb = np.stack([gray] * 3, axis=-1).astype(np.float32)
    rgb[mask] = 0.55 * rgb[mask] + 0.45 * np.array([230, 40, 40])
    Image.fromarray(rgb.astype(np.uint8)).resize((256, 256), Image.NEAREST).save(path)


def main():
    parser = argparse.ArgumentParser(description="Segment the lungs and classify chest X-ray images")
    parser.add_argument("images", nargs="+", help="paths to X-ray images (PNG or JPG)")
    args = parser.parse_args()

    model = load_model()
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    for path in args.images:
        image = preprocess_image(path)
        with torch.no_grad():
            mask_logits, class_logits = model(image.unsqueeze(0))
        probs = class_logits.softmax(1)[0]
        mask = (mask_logits[0, 0] > 0).numpy()
        best = probs.argmax().item()

        overlay = PREDICTIONS_DIR / f"{Path(path).stem}_lungs.png"
        save_overlay(image[0], mask, overlay)
        details = "  ".join(f"{name} {p:.1%}" for name, p in zip(CLASSES, probs.tolist()))
        print(
            f"{path}: {CLASSES[best].upper()} ({probs[best].item():.1%})   [{details}]   "
            f"lungs {mask.mean():.0%} of image -> {os.path.relpath(overlay)}"
        )


if __name__ == "__main__":
    main()
