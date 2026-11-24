import io
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import openpyxl
import torch
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2

ROOT = Path(__file__).parent
IMG_SIZE = 128
PER_CLASS = 2000
CLASSES = ["normal", "lung_opacity", "covid"]
SEED = 0

ARCHIVE = ROOT / "data" / "raw" / "covid19-radiography-database.zip"
ARCHIVE_URL = "https://www.kaggle.com/api/v1/datasets/download/tawsifurrahman/covid19-radiography-database"
ARCHIVE_ROOT = "COVID-19_Radiography_Dataset"
RSNA_URL = "https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data"
# class folder in the archive, and the only source we take it from (None = all sources)
SOURCES = {"normal": ("Normal", RSNA_URL), "lung_opacity": ("Lung_Opacity", RSNA_URL), "covid": ("COVID", None)}
DATA_FILE = ROOT / "data" / f"lungs_{len(CLASSES)}class_{IMG_SIZE}.npz"
SAMPLES_DIR = ROOT / "samples"

# image and mask get the same random rotation, shift and zoom; no flips, because the heart sits on the left
AUGMENT = v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05))


def download():
    if ARCHIVE.exists():
        return
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    partial = ARCHIVE.with_suffix(".part")
    print(f"Downloading {ARCHIVE.name} (816 MB) ...")
    urllib.request.urlretrieve(ARCHIVE_URL, partial)
    partial.rename(ARCHIVE)


def fit_square(image):
    return ImageOps.fit(image.convert("L"), (IMG_SIZE, IMG_SIZE), Image.BICUBIC)


def load_mask(file):
    # masks are 256x256 RGB images with values 0 and 255
    mask = Image.open(file).convert("L").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return np.asarray(mask) > 127


def image_files(archive, folder, source):
    prefix = f"{ARCHIVE_ROOT}/{folder}/images/"
    files = {PurePosixPath(name).stem.lower(): name for name in archive.namelist() if name.startswith(prefix)}
    if source is None:
        return sorted(files.values())
    metadata = io.BytesIO(archive.read(f"{ARCHIVE_ROOT}/{folder}.metadata.xlsx"))
    sheet = openpyxl.load_workbook(metadata, read_only=True).active
    rows = sheet.iter_rows(min_row=2, values_only=True)
    return sorted(files[name.lower()] for name, *_, url in rows if url == source)


def build():
    download()
    rng = np.random.default_rng(SEED)
    paths, labels = [], []
    with zipfile.ZipFile(ARCHIVE) as archive:
        for label, name in enumerate(CLASSES):
            chosen = rng.choice(image_files(archive, *SOURCES[name]), size=PER_CLASS, replace=False)
            paths += sorted(chosen.tolist())
            labels += [label] * PER_CLASS
        print(f"Resizing {len(paths)} images and lung masks to {IMG_SIZE}x{IMG_SIZE} ...")
        images = np.stack([np.asarray(fit_square(Image.open(archive.open(p)))) for p in paths])
        masks = np.stack([load_mask(archive.open(p.replace("/images/", "/masks/"))) for p in paths])
    labels = np.array(labels, dtype=np.uint8)

    indices = np.arange(len(labels))
    train, rest = train_test_split(indices, test_size=0.3, stratify=labels, random_state=SEED)
    val, test = train_test_split(rest, test_size=0.5, stratify=labels[rest], random_state=SEED)
    splits = {"train": train, "val": val, "test": test}
    np.savez_compressed(
        DATA_FILE,
        **{f"{split}_images": images[i] for split, i in splits.items()},
        **{f"{split}_masks": masks[i] for split, i in splits.items()},
        **{f"{split}_labels": labels[i] for split, i in splits.items()},
    )


def load_split(split):
    if not DATA_FILE.exists():
        build()
    with np.load(DATA_FILE) as data:
        return data[f"{split}_images"], data[f"{split}_masks"], data[f"{split}_labels"]


class XRayDataset(Dataset):
    """Returns (image 1xHxW in 0..1, lung mask 1xHxW of 0/1, class index)."""

    def __init__(self, split, augment=False):
        images, masks, labels = load_split(split)
        self.images = torch.from_numpy(images).unsqueeze(1).float() / 255
        self.masks = torch.from_numpy(masks).unsqueeze(1).float()
        self.labels = torch.from_numpy(labels).long()
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        image, mask = self.images[i], self.masks[i]
        if self.augment:
            image, mask = AUGMENT(tv_tensors.Image(image), tv_tensors.Mask(mask))
            image, mask = image.as_subclass(torch.Tensor), mask.as_subclass(torch.Tensor)
        return image, mask, self.labels[i]


def preprocess_image(path):
    image = fit_square(Image.open(path))
    return torch.from_numpy(np.asarray(image, dtype=np.float32) / 255).unsqueeze(0)


def export_samples(per_class=3):
    images, _, labels = load_split("test")
    SAMPLES_DIR.mkdir(exist_ok=True)
    for label, name in enumerate(CLASSES):
        for i in np.flatnonzero(labels == label)[:per_class]:
            Image.fromarray(images[i]).save(SAMPLES_DIR / f"test{i:04d}_{name}.png")


def describe():
    for split in ("train", "val", "test"):
        _, masks, labels = load_split(split)
        counts = ", ".join(f"{name} {np.sum(labels == i)}" for i, name in enumerate(CLASSES))
        print(f"{split:<6}{len(labels):>6} images  ({counts})  lungs cover {masks.mean():.0%} of pixels")


if __name__ == "__main__":
    build()
    describe()
    export_samples()
