# Data Preparation

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split


IMAGE_SIZE = 224
BATCH_SIZE = 32
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "validation", "test")

DATASET_CLASSES = {
    "dataset1": (
        ("Mild Impairment", ("mildimpairment", "milddemented")),
        ("Moderate Impairment", ("moderateimpairment", "moderatedemented")),
        ("No Impairment", ("noimpairment", "nonimpairment", "nondemented")),
        ("Very Mild Impairment", ("verymildimpairment", "verymilddemented")),
    ),
    "dataset2": (
        ("Mild Demented", ("milddemented", "milddementia")),
        ("Moderate Demented", ("moderatedemented", "moderatedementia")),
        ("Non Demented", ("nondemented", "nondementia", "nodementia")),
        ("Very Mild Demented", ("verymilddemented", "verymilddementia")),
    ),
    "dataset3": (
        ("Alzheimer's Disease (AD)", ("ad", "alzheimer", "alzheimers", "alzheimersdisease", "alzheimerdisease")),
        ("Cognitively Normal Controls (CN)", ("cn", "control", "controls", "normal", "cognitivelynormal", "cognitivelynormalcontrols")),
        ("Parkinson's Disease (PD)", ("pd", "parkinson", "parkinsons", "parkinsonsdisease")),
    ),
}


def normalize_name(name: str) -> str:
    return "".join(char.lower() for char in name if char.isalnum())


def identify_class(path: Path, dataset: str) -> int:
    matches = set()
    for folder in path.parts[:-1]:
        name = normalize_name(folder)
        for label_id, (_, aliases) in enumerate(DATASET_CLASSES[dataset]):
            if name in aliases:
                matches.add(label_id)
    if len(matches) != 1:
        raise ValueError(f"Cannot identify a unique class for: {path}")
    return matches.pop()


def scan_images(root: str | Path, dataset: str) -> list[dict]:
    if dataset not in DATASET_CLASSES:
        raise ValueError(f"Unknown dataset: {dataset}")
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    rows = []
    for file in sorted(root.rglob("*")):
        if not file.is_file() or file.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative = file.relative_to(root)
        if dataset == "dataset2" and any(
            "augment" in normalize_name(part) for part in relative.parts[:-1]
        ):
            continue
        label_id = identify_class(relative, dataset)
        rows.append({
            "relative_path": relative.as_posix(),
            "label_idx": label_id,
            "label_name": DATASET_CLASSES[dataset][label_id][0],
        })
    if not rows:
        raise ValueError(f"No class-labelled MRI images found at: {root}")
    return rows


def split_images(rows: list[dict], dataset: str, seed: int = 42) -> list[dict]:
    rows = sorted(rows, key=lambda item: item["relative_path"])
    labels = [int(row["label_idx"]) for row in rows]
    training, held_out = train_test_split(
        rows, test_size=0.2, stratify=labels, random_state=seed
    )
    validation, testing = train_test_split(
        held_out,
        test_size=0.5,
        stratify=[int(row["label_idx"]) for row in held_out],
        random_state=seed,
    )
    result = [
        {**row, "split": split}
        for split, samples in (
            ("train", training), ("validation", validation), ("test", testing)
        )
        for row in samples
    ]
    if len({row["relative_path"] for row in result}) != len(result):
        raise ValueError("An MRI image appears more than once")
    return result


def load_existing_split(root: str | Path, dataset: str) -> list[dict]:
    root = Path(root).expanduser().resolve()
    split_folders = {"train": "train", "val": "validation", "validation": "validation", "test": "test"}
    rows = []
    for folder, split in split_folders.items():
        split_root = root / folder
        if not split_root.is_dir():
            continue
        for row in scan_images(split_root, dataset):
            rows.append({**row, "relative_path": f"{folder}/{row['relative_path']}", "split": split})
    if {row["split"] for row in rows} != set(SPLITS):
        raise ValueError("Existing split directory must contain train, val/validation, and test")
    return rows


def save_manifest(rows: list[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["relative_path", "label_idx", "label_name", "split"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def read_manifest(path: str | Path) -> list[dict]:
    with Path(path).open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        if row["split"] not in SPLITS:
            raise ValueError(f"Invalid split: {row['split']}")
        row["label_idx"] = int(row["label_idx"])
    if len({row["relative_path"] for row in rows}) != len(rows):
        raise ValueError("Duplicate image path found in manifest")
    return rows


def train_transform(image_size: int = IMAGE_SIZE):
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def evaluation_transform(image_size: int = IMAGE_SIZE):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


class MRIImageDataset(Dataset):
    def __init__(self, root: str | Path, rows: list[dict], transform):
        self.root = Path(root).expanduser().resolve()
        self.rows = list(rows)
        self.transform = transform
        if not self.rows:
            raise ValueError("Image dataset split is empty")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        path = (self.root / row["relative_path"]).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"Invalid image path: {path}")
        with Image.open(path) as image:
            rgb = image.convert("RGB")
        return self.transform(rgb), int(row["label_idx"])


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def create_dataloaders(
    root: str | Path,
    dataset: str,
    manifest_csv: str | Path,
    batch_size: int = BATCH_SIZE,
    image_size: int = IMAGE_SIZE,
    num_workers: int = 2,
    seed: int = 42,
    weighted_sampling: bool | None = None,
) -> dict[str, DataLoader]:
    rows = read_manifest(manifest_csv)
    for row in rows:
        label_id = int(row["label_idx"])
        if not 0 <= label_id < len(DATASET_CLASSES[dataset]):
            raise ValueError("Invalid class index")
        if row["label_name"] != DATASET_CLASSES[dataset][label_id][0]:
            raise ValueError("Class label does not match its class index")
    splits = {
        split: MRIImageDataset(
            root,
            [row for row in rows if row["split"] == split],
            train_transform(image_size) if split == "train" else evaluation_transform(image_size),
        )
        for split in SPLITS
    }
    sampler = None
    if weighted_sampling is None:
        weighted_sampling = True
    if weighted_sampling:
        labels = [int(row["label_idx"]) for row in splits["train"].rows]
        counts = Counter(labels)
        weights = torch.tensor([1.0 / counts[label] for label in labels], dtype=torch.double)
        sampler = WeightedRandomSampler(
            weights, len(weights), replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
    loaders = {}
    for offset, split in enumerate(SPLITS):
        loaders[split] = DataLoader(
            splits[split],
            batch_size=batch_size,
            sampler=sampler if split == "train" else None,
            shuffle=split == "train" and sampler is None,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(seed + offset),
        )
    return loaders


def main() -> None:
    parser = argparse.ArgumentParser(description="Data preparation for ViT-GCN MRI experiments")
    parser.add_argument("--dataset", choices=DATASET_CLASSES, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("splits"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=("original", "existing"), default="original")
    args = parser.parse_args()
    if args.mode == "existing":
        rows = load_existing_split(args.root, args.dataset)
    else:
        rows = split_images(scan_images(args.root, args.dataset), args.dataset, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = save_manifest(rows, args.output / f"{args.dataset}_splits.csv")
    print(f"Saved: {manifest}")


if __name__ == "__main__":
    main()
