# Classification and Model Training

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_recall_fscore_support
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import models

from data_preparation import (
    BATCH_SIZE,
    DATASET_CLASSES,
    MRIImageDataset,
    create_dataloaders,
    evaluation_transform,
    read_manifest,
    seed_worker,
)
from vit_gcn import (
    GCNClassifier,
    ViTBackbone,
    ViTGCN,
    build_training_graph,
    extract_embeddings,
)


CNN_MODELS = (
    "VGG16", "ResNet18", "ResNet50", "EfficientNetB3",
    "DenseNet-121", "InceptionV3",
)
ALL_MODELS = CNN_MODELS + ("ViT-B/16", "ViT-GCN")


class TrainingConfig:
    def __init__(
        self,
        epochs: int = 50,
        patience: int = 10,
        vit_lr: float = 3e-5,
        vit_weight_decay: float = 1e-2,
        gcn_lr: float = 1e-3,
        gcn_weight_decay: float = 5e-4,
        batch_size: int = BATCH_SIZE,
    ):
        if epochs < 1 or patience < 1 or batch_size < 1:
            raise ValueError("epochs, patience and batch_size must be positive")
        self.epochs = epochs
        self.patience = patience
        self.vit_lr = vit_lr
        self.vit_weight_decay = vit_weight_decay
        self.gcn_lr = gcn_lr
        self.gcn_weight_decay = gcn_weight_decay
        self.batch_size = batch_size


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if name not in ALL_MODELS:
        raise ValueError(f"Unsupported model: {name}")
    if num_classes < 2:
        raise ValueError("num_classes must be >= 2")

    if name == "ViT-B/16":
        return ViTBackbone(num_classes, pretrained=pretrained)
    if name == "ViT-GCN":
        return ViTGCN(num_classes, pretrained=pretrained)

    weight_map = {
        "VGG16": models.VGG16_Weights.IMAGENET1K_V1,
        "ResNet18": models.ResNet18_Weights.IMAGENET1K_V1,
        "ResNet50": models.ResNet50_Weights.IMAGENET1K_V1,
        "EfficientNetB3": models.EfficientNet_B3_Weights.IMAGENET1K_V1,
        "DenseNet-121": models.DenseNet121_Weights.IMAGENET1K_V1,
        "InceptionV3": models.Inception_V3_Weights.IMAGENET1K_V1,
    }
    weights = weight_map[name] if pretrained else None

    if name == "VGG16":
        model = models.vgg16(weights=weights)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif name == "ResNet18":
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "ResNet50":
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "EfficientNetB3":
        model = models.efficientnet_b3(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == "DenseNet-121":
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    else:
        model = models.inception_v3(
            weights=weights,
            aux_logits=True if pretrained else False,
            init_weights=False,
        )
        model.aux_logits = False
        model.AuxLogits = None
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def as_logits(output: Any) -> Tensor:
    if isinstance(output, Tensor):
        return output
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, (tuple, list)):
        return output[0]
    raise TypeError("Model output is not a tensor or recognized logits container")


def compute_metrics(logits: Tensor, targets: Tensor) -> dict[str, float]:
    if logits.size(0) != targets.numel():
        raise ValueError("Predictions and targets have different lengths")
    y = targets.detach().cpu().numpy()
    predicted = logits.detach().argmax(dim=1).cpu().numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, predicted, average="weighted", zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "weighted_precision": float(precision),
        "weighted_recall": float(recall),
        "weighted_f1": float(f1),
        "cohens_kappa": float(cohen_kappa_score(y, predicted)),
    }


def image_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_items = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = as_logits(model(images))
            loss = F.cross_entropy(logits, targets)
            if training:
                loss.backward()
                optimizer.step()
            items = targets.size(0)
            total_loss += loss.item() * items
            total_correct += (logits.argmax(dim=1) == targets).sum().item()
            total_items += items

    if total_items == 0:
        raise ValueError("Empty data loader")
    return total_loss / total_items, total_correct / total_items


@torch.no_grad()
def collect_image_predictions(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[Tensor, Tensor]:
    model.eval()
    logits_all, targets_all = [], []
    for images, targets in loader:
        logits_all.append(as_logits(model(images.to(device))).detach().cpu())
        targets_all.append(targets.cpu())
    if not logits_all:
        raise ValueError("Empty evaluation loader")
    return torch.cat(logits_all), torch.cat(targets_all)


def fit_image_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    device: torch.device,
) -> list[dict[str, float]]:
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.vit_lr, weight_decay=config.vit_weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )
    best_state = None
    best_accuracy = -1.0
    stale_epochs = 0
    history = []

    for epoch in range(config.epochs):
        train_loss, train_accuracy = image_epoch(model, train_loader, device, optimizer)
        val_loss, val_accuracy = image_epoch(model, val_loader, device)
        scheduler.step()
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        })
        print(
            f"Image stage {epoch + 1:02d}/{config.epochs}: "
            f"train={train_accuracy:.4f} val={val_accuracy:.4f}"
        )
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= config.patience:
            break

    if best_state is None:
        raise RuntimeError("No valid checkpoint was selected")
    model.load_state_dict(best_state)
    model.eval()
    return history


def make_reference_loader(
    root: Path,
    manifest: Path,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> DataLoader:
    rows = [row for row in read_manifest(manifest) if row["split"] == "train"]
    dataset = MRIImageDataset(root, rows, evaluation_transform())
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )


@torch.no_grad()
def evaluate_gcn(
    gcn: GCNClassifier,
    query_features: Tensor,
    query_labels: Tensor,
    training_graph,
    batch_size: int = 32,
) -> tuple[float, dict[str, float], Tensor]:
    gcn.eval()
    logits = gcn.predict_independent(
        query_features, training_graph, batch_size=batch_size
    ).cpu()
    metrics = compute_metrics(logits, query_labels)
    loss = F.cross_entropy(logits, query_labels.long()).item()
    return loss, metrics, logits


def fit_gcn(
    model: ViTGCN,
    train_features: Tensor,
    train_labels: Tensor,
    val_features: Tensor,
    val_labels: Tensor,
    config: TrainingConfig,
    device: torch.device,
) -> tuple[Any, list[dict[str, float]]]:
    graph = build_training_graph(train_features, k=5)
    model.gcn.to(device)
    features = graph.features.to(device)
    adjacency = graph.adjacency.to(device)
    targets = train_labels.long().to(device)
    optimizer = torch.optim.Adam(
        model.gcn.parameters(),
        lr=config.gcn_lr,
        weight_decay=config.gcn_weight_decay,
    )
    best_state = None
    best_accuracy = -1.0
    stale_epochs = 0
    history = []

    for epoch in range(config.epochs):
        model.gcn.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model.gcn(features, adjacency)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()

        val_loss, val_metrics, _ = evaluate_gcn(
            model.gcn, val_features, val_labels, graph, batch_size=config.batch_size
        )
        with torch.no_grad():
            model.gcn.eval()
            train_accuracy = float(
                (model.gcn(features, adjacency).argmax(dim=1) == targets)
                .float().mean().item()
            )

        accuracy = val_metrics["accuracy"]
        history.append({
            "epoch": epoch + 1,
            "train_loss": float(loss.item()),
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": accuracy,
        })
        print(
            f"GCN stage {epoch + 1:02d}/{config.epochs}: "
            f"train={train_accuracy:.4f} val={accuracy:.4f}"
        )
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = copy.deepcopy(model.gcn.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= config.patience:
            break

    if best_state is None:
        raise RuntimeError("No valid GCN checkpoint was selected")
    model.gcn.load_state_dict(best_state)
    model.gcn.eval()
    return graph, history


def experiment(
    dataset_name: str,
    root: Path,
    manifest: Path,
    model_name: str,
    output_dir: Path,
    config: TrainingConfig,
    seed: int = 42,
    pretrained: bool = True,
    num_workers: int = 2,
    device: torch.device | None = None,
) -> dict[str, Any]:
    if dataset_name not in DATASET_CLASSES:
        raise ValueError("Invalid dataset")
    if model_name not in ALL_MODELS:
        raise ValueError("Invalid model")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    num_classes = len(DATASET_CLASSES[dataset_name])
    loaders = create_dataloaders(
        root=root,
        dataset=dataset_name,
        manifest_csv=manifest,
        batch_size=config.batch_size,
        seed=seed,
        num_workers=num_workers,
    )
    reference_loader = make_reference_loader(
        root, manifest, config.batch_size, num_workers, seed
    )
    model = build_model(model_name, num_classes, pretrained=pretrained).to(device)
    output_dir.mkdir(parents=True, exist_ok=True)

    if model_name == "ViT-GCN":
        vit_history = fit_image_model(
            model.vit, loaders["train"], loaders["validation"], config, device
        )
        train_features, train_targets = extract_embeddings(
            model.vit, reference_loader, device=device
        )
        val_features, val_targets = extract_embeddings(
            model.vit, loaders["validation"], device=device
        )
        test_features, test_targets = extract_embeddings(
            model.vit, loaders["test"], device=device
        )
        graph, gcn_history = fit_gcn(
            model, train_features, train_targets,
            val_features, val_targets, config, device
        )
        with torch.no_grad():
            model.gcn.eval()
            train_logits = model.gcn(
                graph.features.to(device), graph.adjacency.to(device)
            ).detach().cpu()
        train_metrics = compute_metrics(train_logits, train_targets)
        _, val_metrics, val_logits = evaluate_gcn(
            model.gcn, val_features, val_targets, graph,
            batch_size=config.batch_size
        )
        _, test_metrics, test_logits = evaluate_gcn(
            model.gcn, test_features, test_targets, graph,
            batch_size=config.batch_size
        )
        history = {"vit": vit_history, "gcn": gcn_history}
        prediction_data = {
            "train": (train_logits, train_targets),
            "validation": (val_logits, val_targets),
            "test": (test_logits, test_targets),
        }
        checkpoint = {
            "model_name": model_name,
            "dataset": dataset_name,
            "num_classes": num_classes,
            "model_state": model.state_dict(),
            "train_embeddings": train_features,
            "train_labels": train_targets,
            "k": 5,
            "seed": seed,
            "history": history,
        }
    else:
        history = fit_image_model(
            model, loaders["train"], loaders["validation"], config, device
        )
        prediction_data = {}
        for split, loader in (
            ("train", reference_loader),
            ("validation", loaders["validation"]),
            ("test", loaders["test"]),
        ):
            prediction_data[split] = collect_image_predictions(model, loader, device)
        train_metrics = compute_metrics(*prediction_data["train"])
        val_metrics = compute_metrics(*prediction_data["validation"])
        test_metrics = compute_metrics(*prediction_data["test"])
        checkpoint = {
            "model_name": model_name,
            "dataset": dataset_name,
            "num_classes": num_classes,
            "model_state": model.state_dict(),
            "seed": seed,
            "history": history,
        }

    name = model_name.replace("/", "_").replace(" ", "_")
    stem = f"{dataset_name}_{name}"
    torch.save(checkpoint, output_dir / f"{stem}.pt")
    torch.save(prediction_data, output_dir / f"{stem}_predictions.pt")
    result = {
        "dataset": dataset_name,
        "model": model_name,
        "train": train_metrics,
        "validation": val_metrics,
        "test": test_metrics,
        "checkpoint": f"{stem}.pt",
        "predictions": f"{stem}_predictions.pt",
        "seed": seed,
        "epochs_max": config.epochs,
        "early_stopping_patience": config.patience,
    }
    (output_dir / f"{stem}_results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(f"{dataset_name} | {model_name} | test accuracy={test_metrics['accuracy']:.4f}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the six CNN baselines, standalone ViT, or ViT-GCN"
    )
    parser.add_argument("--dataset", choices=DATASET_CLASSES, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["ViT-GCN"])
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()
    model_names = list(ALL_MODELS) if args.models == ["all"] else args.models
    invalid_names = [name for name in model_names if name not in ALL_MODELS]
    if invalid_names:
        parser.error("Unknown models: " + ", ".join(invalid_names))
    config = TrainingConfig(
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
    )
    for name in model_names:
        experiment(
            dataset_name=args.dataset,
            root=args.root,
            manifest=args.manifest,
            model_name=name,
            output_dir=args.output,
            config=config,
            seed=args.seed,
            pretrained=not args.no_pretrained,
            num_workers=args.num_workers,
        )


if __name__ == "__main__":
    main()
