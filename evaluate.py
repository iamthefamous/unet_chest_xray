import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report

from data import CLASSES, XRayDataset
from model import OUTPUT_DIR, dice_scores, load_model


def plot_examples(test_set, mask_logits, probs, per_class=3):
    fig, axes = plt.subplots(len(CLASSES), per_class, figsize=(per_class * 3, len(CLASSES) * 3.3))
    for label, row in enumerate(axes):
        for ax, i in zip(row, np.flatnonzero(test_set.labels.numpy() == label)[:per_class]):
            dice = dice_scores(mask_logits[i : i + 1], test_set.masks[i : i + 1]).item()
            ax.imshow(test_set.images[i, 0], cmap="gray")
            ax.contour(test_set.masks[i, 0], levels=[0.5], colors="lime", linewidths=1.2)
            ax.contour(mask_logits[i, 0] > 0, levels=[0.5], colors="red", linewidths=1.2)
            predicted = probs[i].argmax().item()
            ax.set_title(
                f"true {CLASSES[label]}\npred {CLASSES[predicted]} {probs[i, predicted]:.0%} · Dice {dice:.2f}",
                fontsize=9,
                color="black" if predicted == label else "red",
            )
            ax.axis("off")
    fig.suptitle("Test images: true lung outline (green) vs U-Net mask (red)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "segmentation_examples.png", dpi=150)


def main():
    model = load_model()
    test_set = XRayDataset("test")
    with torch.no_grad():
        outputs = [model(batch) for batch in test_set.images.split(128)]
    mask_logits = torch.cat([m for m, _ in outputs])
    probs = torch.cat([c for _, c in outputs]).softmax(1)
    y_true = test_set.labels.numpy()
    y_pred = probs.argmax(1).numpy()

    dice = dice_scores(mask_logits, test_set.masks)
    iou = dice / (2 - dice)
    report = classification_report(y_true, y_pred, target_names=CLASSES, digits=3)
    summary = (
        f"Test images: {len(y_true)}\n\n"
        f"Classification accuracy: {accuracy_score(y_true, y_pred):.3f}\n\n{report}\n"
        f"Lung segmentation: mean Dice {dice.mean():.3f}, mean IoU {iou.mean():.3f}, "
        f"worst Dice {dice.min():.3f}\n"
    )
    print(summary)
    (OUTPUT_DIR / "test_metrics.txt").write_text(summary)

    display = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=CLASSES, cmap="Blues", colorbar=False
    )
    display.ax_.set(title="Confusion matrix (test set)", xlabel="Predicted", ylabel="True")
    display.figure_.tight_layout()
    display.figure_.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)

    plot_examples(test_set, mask_logits, probs)


if __name__ == "__main__":
    main()
