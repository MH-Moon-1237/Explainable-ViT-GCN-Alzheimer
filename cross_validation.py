# Cross Validation

from __future__ import annotations

import argparse
import gc
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from data_preparation import (
    DATASET_CLASSES,
    MRIImageDataset,
    evaluation_transform,
    read_manifest,
    scan_images,
    seed_worker,
    split_images,
    train_transform,
)
from classification import (
    TrainingConfig,
    build_model,
    collect_image_predictions,
    compute_metrics,
    evaluate_gcn,
    fit_gcn,
    fit_image_model,
    set_seed,
)
from vit_gcn import extract_embeddings


CV_MODELS = ("ViT-B/16", "ViT-GCN")


def select_cv_pool(
    dataset: str,
    root: Path,
    manifest: Path | None,
    pool: str,
    seed: int,
) -> list[dict]:
    if manifest is not None:
        rows = read_manifest(manifest)
    else:
        raw = scan_images(root, dataset)
        rows = split_images(raw, dataset, seed=seed)

    if pool == "trainval":
        rows = [row for row in rows if row.get("split") in ("train", "validation")]
    elif pool != "all":
        raise ValueError("pool must be 'trainval' or 'all'")

    rows = sorted(rows, key=lambda row: row["relative_path"])
    if not rows:
        raise ValueError("No images found in the cross-validation pool")
    if len({row["relative_path"] for row in rows}) != len(rows):
        raise ValueError("The cross-validation pool contains duplicate image paths")

    for row in rows:
        label = int(row["label_idx"])
        if not 0 <= label < len(DATASET_CLASSES[dataset]):
            raise ValueError(f"Invalid class index: {label}")
        if row["label_name"] != DATASET_CLASSES[dataset][label][0]:
            raise ValueError("Class label and class index disagree")

    return rows


def get_folds(rows: list[dict], n_splits: int = 10, seed: int = 42):
    labels = np.asarray([int(row["label_idx"]) for row in rows], dtype=np.int64)
    counts = Counter(labels.tolist())
    if len(counts) < 2 or min(counts.values()) < n_splits:
        raise ValueError("Each class needs at least n_splits images for stratified CV")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    seen_validation = np.zeros(len(rows), dtype=np.int64)
    splits = []
    for fold_number, (train_idx, val_idx) in enumerate(
        skf.split(np.zeros(len(rows)), labels), start=1
    ):
        if np.intersect1d(train_idx, val_idx).size:
            raise RuntimeError("A training/validation overlap was detected")
        seen_validation[val_idx] += 1
        splits.append((fold_number, train_idx, val_idx))
    if not np.all(seen_validation == 1):
        raise RuntimeError("Each image must appear in exactly one validation fold")
    return splits


def make_fold_loaders(
    root: Path,
    train_rows: list[dict],
    val_rows: list[dict],
    batch_size: int,
    num_workers: int,
    seed: int,
):
    train_data = MRIImageDataset(root, train_rows, train_transform())
    train_ref_data = MRIImageDataset(root, train_rows, evaluation_transform())
    val_data = MRIImageDataset(root, val_rows, evaluation_transform())

    train_labels = [int(row["label_idx"]) for row in train_rows]
    counts = Counter(train_labels)
    weights = torch.as_tensor(
        [1.0 / counts[label] for label in train_labels], dtype=torch.double
    )
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )

    def loader(dataset, shuffle=False, sampler=None, offset=0):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(seed + offset),
        )

    return (
        loader(train_data, sampler=sampler),
        loader(val_data, offset=1),
        loader(train_ref_data, offset=2),
    )


def per_class_roc(logits: torch.Tensor, targets: torch.Tensor, classes: int) -> dict:
    probabilities = logits.detach().float().softmax(dim=1).cpu().numpy()
    labels = targets.detach().long().cpu().numpy()
    results = {}
    for label in range(classes):
        binary = (labels == label).astype(np.int64)
        if np.unique(binary).size != 2:
            results[str(label)] = None
            continue
        fpr, tpr, _ = roc_curve(binary, probabilities[:, label])
        results[str(label)] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": float(roc_auc_score(binary, probabilities[:, label])),
        }
    return results


def run_fold_model(
    model_name: str,
    num_classes: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    reference_loader: DataLoader,
    config: TrainingConfig,
    device: torch.device,
    pretrained: bool,
    save_checkpoint: Path | None = None,
):
    model = build_model(model_name, num_classes, pretrained=pretrained).to(device)

    if model_name == "ViT-B/16":
        history = {"vit": fit_image_model(model, train_loader, val_loader, config, device)}
        logits, targets = collect_image_predictions(model, val_loader, device)
        checkpoint = {"model_state": model.state_dict()}
    elif model_name == "ViT-GCN":
        vit_history = fit_image_model(model.vit, train_loader, val_loader, config, device)
        training_features, training_targets = extract_embeddings(
            model.vit, reference_loader, device=device
        )
        validation_features, validation_targets = extract_embeddings(
            model.vit, val_loader, device=device
        )
        graph, gcn_history = fit_gcn(
            model,
            training_features,
            training_targets,
            validation_features,
            validation_targets,
            config,
            device,
        )
        _, _, logits = evaluate_gcn(
            model.gcn,
            validation_features,
            validation_targets,
            graph,
            batch_size=config.batch_size,
        )
        targets = validation_targets
        history = {"vit": vit_history, "gcn": gcn_history}
        checkpoint = {
            "model_state": model.state_dict(),
            "training_features": training_features,
            "training_labels": training_targets,
            "graph_neighbors": 5,
        }
    else:
        raise ValueError(f"CV includes only {CV_MODELS}")

    logits = logits.detach().cpu()
    targets = targets.detach().cpu().long()
    metrics = compute_metrics(logits, targets)
    cm = confusion_matrix(
        targets.numpy(), logits.argmax(dim=1).numpy(), labels=list(range(num_classes))
    )
    roc_data = per_class_roc(logits, targets, num_classes)

    if save_checkpoint is not None:
        save_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model_name, "classes": num_classes, **checkpoint}, save_checkpoint)

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "roc": roc_data,
        "history": history,
    }, logits, targets


def summarize(folds: list[dict]) -> dict:
    scores = {}
    for name in CV_MODELS:
        accuracies = np.asarray(
            [fold["models"][name]["metrics"]["accuracy"] * 100 for fold in folds],
            dtype=float,
        )
        metrics = (
            "accuracy", "weighted_precision", "weighted_recall",
            "weighted_f1", "cohens_kappa",
        )
        mean_metrics = {
            key: float(np.mean([
                fold["models"][name]["metrics"][key] for fold in folds
            ]))
            for key in metrics
        }
        scores[name] = {
            "mean_accuracy_pct": float(np.mean(accuracies)),
            "sd_accuracy_pp": float(np.std(accuracies, ddof=1)),
            "fold_accuracies_pct": accuracies.tolist(),
            "mean_metrics": mean_metrics,
        }
    improvements = np.asarray(
        [
            (fold["models"]["ViT-GCN"]["metrics"]["accuracy"] -
             fold["models"]["ViT-B/16"]["metrics"]["accuracy"]) * 100
            for fold in folds
        ], dtype=float,
    )
    scores["paired_improvement"] = {
        "mean_pp": float(np.mean(improvements)),
        "sd_pp": float(np.std(improvements, ddof=1)),
        "fold_improvements_pp": improvements.tolist(),
    }
    return scores


def cross_validate(
    dataset: str,
    root: Path,
    manifest: Path | None,
    output: Path,
    config: TrainingConfig,
    seed: int = 42,
    num_workers: int = 2,
    pretrained: bool = True,
    n_splits: int = 10,
    pool: str = "trainval",
    save_checkpoints: bool = False,
    dry_run: bool = False,
    device: torch.device | None = None,
) -> dict:
    if dataset not in DATASET_CLASSES:
        raise ValueError(f"Unsupported dataset: {dataset}")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(root).expanduser().resolve()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = select_cv_pool(dataset, root, manifest, pool, seed)
    folds = get_folds(rows, n_splits=n_splits, seed=seed)

    summary = {
        "dataset": dataset,
        "seed": seed,
        "pool": pool,
        "source_manifest": str(manifest) if manifest else None,
        "num_images": len(rows),
        "n_splits": n_splits,
        "fold_results": [],
    }

    for fold_number, train_idx, val_idx in folds:
        train_rows = [rows[i] for i in train_idx]
        val_rows = [rows[i] for i in val_idx]
        fold_dir = output / dataset / f"fold_{fold_number:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_info = {
            "fold": fold_number,
            "train_samples": len(train_rows),
            "validation_samples": len(val_rows),
            "models": {},
        }

        (fold_dir / "partition.json").write_text(
            json.dumps(
                {
                    "train_paths": [r["relative_path"] for r in train_rows],
                    "validation_paths": [r["relative_path"] for r in val_rows],
                }, indent=2
            ),
            encoding="utf-8",
        )
        if dry_run:
            summary["fold_results"].append(fold_info)
            continue

        for name in CV_MODELS:
            model_seed = seed + fold_number
            set_seed(model_seed)
            train_loader, val_loader, reference_loader = make_fold_loaders(
                root, train_rows, val_rows, config.batch_size, num_workers, model_seed
            )
            checkpoint_path = (
                fold_dir / f"{name.replace('/', '_').replace('-', '_')}.pt"
                if save_checkpoints else None
            )
            print(f"{dataset} | fold {fold_number:02d}/{n_splits} | {name}")
            result, logits, labels = run_fold_model(
                name,
                len(DATASET_CLASSES[dataset]),
                train_loader,
                val_loader,
                reference_loader,
                config,
                device,
                pretrained,
                checkpoint_path,
            )
            (fold_dir / f"{name.replace('/', '_').replace('-', '_')}_metrics.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
            torch.save(
                {
                    "logits": logits,
                    "labels": labels,
                    "paths": [r["relative_path"] for r in val_rows],
                },
                fold_dir / f"{name.replace('/', '_').replace('-', '_')}_predictions.pt",
            )
            fold_info["models"][name] = {
                "metrics": result["metrics"],
                "confusion_matrix": result["confusion_matrix"],
            }

        summary["fold_results"].append(fold_info)
        interim = output / dataset / "cross_validation_results.json"
        interim.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if not dry_run:
        summary["summary"] = summarize(summary["fold_results"])
        print(json.dumps(summary["summary"], indent=2))
    (output / dataset / "cross_validation_results.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Stratified 10-fold ViT vs ViT-GCN evaluation")
    parser.add_argument("--dataset", choices=DATASET_CLASSES, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("results_cv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--cv-pool", choices=("trainval", "all"), default="trainval")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = TrainingConfig(
        epochs=args.epochs, patience=args.patience, batch_size=args.batch_size
    )
    cross_validate(
        dataset=args.dataset,
        root=args.root,
        manifest=args.manifest,
        output=args.output,
        config=config,
        seed=args.seed,
        num_workers=args.num_workers,
        pretrained=not args.no_pretrained,
        n_splits=args.n_splits,
        pool=args.cv_pool,
        save_checkpoints=args.save_checkpoints,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
