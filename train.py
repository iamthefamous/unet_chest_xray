import json
import time

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

from data import XRayDataset
from model import MODEL_PATH, OUTPUT_DIR, LungUNet, dice_scores, save_model

EPOCHS = 12
BATCH_SIZE = 32
ENCODER_LR = 1e-4  # pretrained ResNet-18 layers: small steps, so the ImageNet features are adapted, not erased
DECODER_LR = 1e-3  # new layers (decoder and heads) start from random weights


def run_epoch(model, loader, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, dice = 0.0, 0, 0.0
    with torch.set_grad_enabled(training):
        for images, masks, labels in loader:
            mask_logits, class_logits = model(images)
            loss = nn.functional.cross_entropy(class_logits, labels) + nn.functional.binary_cross_entropy_with_logits(
                mask_logits, masks
            )
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (class_logits.argmax(1) == labels).sum().item()
            dice += dice_scores(mask_logits, masks).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n, dice / n


def plot_history(history):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    titles = ("Loss (cross-entropy + mask BCE)", "Classification accuracy", "Lung mask Dice")
    for ax, metric, title in zip(axes, ("loss", "acc", "dice"), titles):
        for split in ("train", "val"):
            ax.plot(epochs, history[f"{split}_{metric}"], marker="o", markersize=3, label=split)
        ax.set(title=title, xlabel="Epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "learning_curves.png", dpi=150)


def main():
    torch.manual_seed(0)
    train_loader = DataLoader(XRayDataset("train", augment=True), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(XRayDataset("val"), batch_size=128)

    model = LungUNet(pretrained=True)
    encoder = model.encoder_parameters()
    encoder_ids = {id(p) for p in encoder}
    new_layers = [p for p in model.parameters() if id(p) not in encoder_ids]
    optimizer = torch.optim.AdamW([{"params": encoder, "lr": ENCODER_LR}, {"params": new_layers, "lr": DECODER_LR}])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    keys = [f"{split}_{metric}" for split in ("train", "val") for metric in ("loss", "acc", "dice")]
    history = {key: [] for key in keys}
    best_val_acc = 0.0
    OUTPUT_DIR.mkdir(exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        results = (*run_epoch(model, train_loader, optimizer), *run_epoch(model, val_loader))
        scheduler.step()
        for key, value in zip(keys, results):
            history[key].append(value)
        (OUTPUT_DIR / "history.json").write_text(json.dumps(history, indent=2))
        train_loss, train_acc, train_dice, val_loss, val_acc, val_dice = results
        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            save_model(model)
        print(
            f"epoch {epoch:2d}/{EPOCHS}  "
            f"train loss {train_loss:.3f} acc {train_acc:.3f} dice {train_dice:.3f}  "
            f"val loss {val_loss:.3f} acc {val_acc:.3f} dice {val_dice:.3f}  "
            f"{time.time() - start:.0f}s{'  saved' if improved else ''}",
            flush=True,
        )

    plot_history(history)
    print(f"Best validation accuracy {best_val_acc:.3f}, model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
