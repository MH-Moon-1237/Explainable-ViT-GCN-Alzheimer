# Performance Evaluation

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from data_preparation import DATASET_CLASSES


MODELS = (
    "VGG16", "ResNet18", "ResNet50", "EfficientNetB3",
    "DenseNet-121", "InceptionV3", "ViT-B/16", "ViT-GCN",
)
CV_MODELS = ("ViT-B/16", "ViT-GCN")


def class_names(dataset: str) -> list[str]:
    if dataset not in DATASET_CLASSES:
        raise ValueError(f"Unknown dataset: {dataset}")
    return [name for name, _ in DATASET_CLASSES[dataset]]


def model_stem(dataset: str, model: str) -> str:
    return f"{dataset}_{model.replace('/', '_').replace(' ', '_')}"


def cv_model_stem(model: str) -> str:
    return model.replace("/", "_").replace("-", "_")


def load_torch(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=True)


def load_predictions(path: Path, split: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    payload = load_torch(path)
    if split is not None:
        if split not in payload:
            raise ValueError(f"Split {split!r} missing from {path}")
        logits, targets = payload[split]
    else:
        logits, targets = payload["logits"], payload["labels"]
    logits = torch.as_tensor(logits).detach().cpu().numpy().astype(np.float64)
    targets = torch.as_tensor(targets).detach().cpu().numpy().astype(np.int64)
    if logits.ndim != 2 or targets.ndim != 1 or len(logits) != len(targets):
        raise ValueError(f"Invalid prediction dimensions in {path}")
    if len(targets) == 0 or not np.isfinite(logits).all():
        raise ValueError(f"Empty or non-finite predictions in {path}")
    if np.any(targets < 0) or np.any(targets >= logits.shape[1]):
        raise ValueError(f"Class indices outside the logits range in {path}")
    return logits, targets


def softmax(logits: np.ndarray) -> np.ndarray:
    centered = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(centered)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def calculate_roc(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    curves = {}
    supports = np.bincount(labels, minlength=probabilities.shape[1])
    aucs = []
    weights = []
    for idx in range(probabilities.shape[1]):
        binary_labels = (labels == idx).astype(np.int64)
        if np.unique(binary_labels).size < 2:
            curves[str(idx)] = None
            continue
        false_positive, true_positive, _ = roc_curve(binary_labels, probabilities[:, idx])
        auc = float(roc_auc_score(binary_labels, probabilities[:, idx]))
        curves[str(idx)] = {
            "fpr": false_positive.tolist(),
            "tpr": true_positive.tolist(),
            "auc": auc,
        }
        aucs.append(auc)
        weights.append(int(supports[idx]))
    return {
        "per_class": curves,
        "macro_roc_auc_ovr": float(np.mean(aucs)) if aucs else None,
        "weighted_roc_auc_ovr": (
            float(np.average(aucs, weights=weights)) if aucs and sum(weights) else None
        ),
    }


def evaluate_logits(
    logits: np.ndarray, labels: np.ndarray, names: list[str]
) -> dict:
    if logits.shape[1] != len(names):
        raise ValueError("Logit dimensions do not match dataset class labels")
    predicted = logits.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicted, labels=np.arange(len(names)),
        average="weighted", zero_division=0,
    )
    cm = confusion_matrix(labels, predicted, labels=np.arange(len(names)))
    per_class = classification_report(
        labels, predicted, labels=np.arange(len(names)),
        target_names=names, output_dict=True, zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "weighted_precision": float(precision),
        "weighted_recall": float(recall),
        "weighted_f1": float(f1),
        "cohens_kappa": float(cohen_kappa_score(labels, predicted)),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "roc": calculate_roc(labels, softmax(logits)),
        "samples": int(len(labels)),
    }


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def save_csv(path: Path, rows: list[dict], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_confusion(matrix: list[list[int]], names: list[str], title: str, path: Path) -> None:
    values = np.asarray(matrix, dtype=np.int64)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(values)
    ax.set_xticks(range(len(names)), names, rotation=35, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    for row in range(len(names)):
        for col in range(len(names)):
            ax.text(col, row, str(values[row, col]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_roc(roc: dict, names: list[str], title: str, path: Path) -> None:
    curves = roc["per_class"]
    fig, ax = plt.subplots(figsize=(7, 6))
    plotted = False
    for idx, name in enumerate(names):
        item = curves.get(str(idx))
        if item is None:
            continue
        ax.plot(item["fpr"], item["tpr"], label=f"{name} (AUC={item['auc']:.4f})")
        plotted = True
    ax.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title=title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    if plotted:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_learning_history(history, title: str, folder: Path) -> None:
    if not history:
        return
    stages = history if isinstance(history, dict) else {"image": history}
    for stage, values in stages.items():
        if not values:
            continue
        epochs = [int(row["epoch"]) for row in values]
        for metric in ("accuracy", "loss"):
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(epochs, [row[f"train_{metric}"] for row in values], label="Training")
            ax.plot(epochs, [row[f"val_{metric}"] for row in values], label="Validation")
            ax.set(xlabel="Epoch", ylabel=metric.title(), title=f"{title} | {stage} {metric}")
            ax.legend()
            fig.tight_layout()
            fig.savefig(folder / f"{stage}_{metric}.png", dpi=200)
            plt.close(fig)


def plot_fixed_comparison(rows: list[dict], dataset: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [row["Model"] for row in rows]
    values = [100.0 * row["Test Acc"] for row in rows]
    ax.bar(labels, values)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(f"{dataset}: Fixed test-set model comparison")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def analyze_fixed(dataset: str, result_dir: Path, output: Path, figures: bool) -> list[dict]:
    names = class_names(dataset)
    folder = output / "fixed" / dataset
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    reference_labels = None
    for model in MODELS:
        stem = model_stem(dataset, model)
        meta_path = result_dir / f"{stem}_results.json"
        if not meta_path.is_file():
            print(f"Skipping {model}: no training output at {meta_path}")
            continue
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        prediction_path = result_dir / metadata["predictions"]
        metrics = {}
        for split in ("train", "validation", "test"):
            logits, targets = load_predictions(prediction_path, split)
            metrics[split] = evaluate_logits(logits, targets, names)
            if split == "test":
                if reference_labels is None:
                    reference_labels = targets
                elif not np.array_equal(reference_labels, targets):
                    raise ValueError(f"Test image ordering or labels differ for {model}")
                if figures:
                    model_folder = folder / stem
                    model_folder.mkdir(exist_ok=True)
                    plot_confusion(
                        metrics[split]["confusion_matrix"], names,
                        f"{dataset} | {model} | Confusion matrix",
                        model_folder / "confusion_matrix.png",
                    )
                    plot_roc(
                        metrics[split]["roc"], names,
                        f"{dataset} | {model} | ROC-AUC",
                        model_folder / "roc_auc.png",
                    )
        save_json(folder / f"{stem}_evaluation.json", metrics)
        rows.append({
            "Model": model,
            "Optimizer": "AdamW + Adam" if model == "ViT-GCN" else "AdamW",
            "LR": "3e-5 + 1e-3" if model == "ViT-GCN" else "3e-5",
            "Batch Size": 32,
            "Train Acc": metrics["train"]["accuracy"],
            "Val Acc": metrics["validation"]["accuracy"],
            "Test Acc": metrics["test"]["accuracy"],
            "Precision": metrics["test"]["weighted_precision"],
            "Recall": metrics["test"]["weighted_recall"],
            "F1 Score": metrics["test"]["weighted_f1"],
            "Cohen's Kappa": metrics["test"]["cohens_kappa"],
        })
        if figures:
            checkpoint = result_dir / metadata["checkpoint"]
            if checkpoint.is_file():
                saved = load_torch(checkpoint)
                history = saved.get("history")
                if history:
                    curves_folder = folder / stem
                    curves_folder.mkdir(exist_ok=True)
                    plot_learning_history(history, f"{dataset} | {model}", curves_folder)
    if rows:
        fields = [
            "Model", "Optimizer", "LR", "Batch Size", "Train Acc", "Val Acc",
            "Test Acc", "Precision", "Recall", "F1 Score", "Cohen's Kappa",
        ]
        save_csv(folder / "model_comparison.csv", rows, fields)
        if figures:
            plot_fixed_comparison(rows, dataset, folder / "accuracy_comparison.png")
    else:
        print(f"No trained fixed-split models found for {dataset}")
    return rows


def plot_cv_accuracy(folds: list[dict], dataset: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    x = [fold["Fold"] for fold in folds if fold["Model"] == CV_MODELS[0]]
    for model in CV_MODELS:
        rows = [fold for fold in folds if fold["Model"] == model]
        if rows:
            ax.plot(x, [row["Accuracy (%)"] for row in rows], marker="o", label=model)
    ax.set(xlabel="Fold", ylabel="Accuracy (%)", title=f"{dataset} | Stratified 10-fold CV")
    ax.set_xticks(x)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)



def plot_three_dataset_comparison(all_fixed: dict[str, list[dict]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    positions = np.arange(len(MODELS))
    available = [(dataset, rows) for dataset, rows in all_fixed.items() if rows]
    width = 0.78 / len(available)
    for index, (dataset, rows) in enumerate(available):
        scores = {row["Model"]: 100.0 * row["Test Acc"] for row in rows}
        values = [scores.get(model, np.nan) for model in MODELS]
        offsets = positions + (index - (len(available) - 1) / 2) * width
        ax.bar(offsets, values, width=width, label=dataset)
    ax.set_xticks(positions, MODELS, rotation=35, ha="right")
    ax.set(ylabel="Held-out test accuracy (%)", title="Three-dataset fixed test-set comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

def analyze_cv(
    dataset: str, cv_dir: Path, output: Path,
    figures: bool, fold_figures: bool,
) -> list[dict]:
    names = class_names(dataset)
    source = cv_dir / dataset
    summary_path = source / "cross_validation_results.json"
    if not summary_path.is_file():
        print(f"No CV results for {dataset}: {summary_path}")
        return []
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("dataset") != dataset:
        raise ValueError(f"CV dataset mismatch at {summary_path}")
    output_folder = output / "cv" / dataset
    output_folder.mkdir(parents=True, exist_ok=True)
    fold_rows = []
    records = []
    for fold in summary["fold_results"]:
        number = int(fold["fold"])
        fold_source = source / f"fold_{number:02d}"
        label_reference = None
        for model in CV_MODELS:
            file = fold_source / f"{cv_model_stem(model)}_predictions.pt"
            if not file.is_file():
                continue
            logits, labels = load_predictions(file)
            if label_reference is None:
                label_reference = labels
            elif not np.array_equal(label_reference, labels):
                raise ValueError(f"CV validation labels do not match in fold {number}")
            result = evaluate_logits(logits, labels, names)
            records.append({"fold": number, "model": model, "metrics": result})
            fold_rows.append({
                "Fold": number,
                "Model": model,
                "Accuracy (%)": result["accuracy"] * 100.0,
                "Weighted Precision": result["weighted_precision"],
                "Weighted Recall": result["weighted_recall"],
                "Weighted F1": result["weighted_f1"],
                "Cohen's Kappa": result["cohens_kappa"],
            })
            if figures and fold_figures:
                folder = output_folder / f"fold_{number:02d}" / cv_model_stem(model)
                folder.mkdir(parents=True, exist_ok=True)
                plot_confusion(
                    result["confusion_matrix"], names,
                    f"{dataset} | Fold {number} | {model}", folder / "confusion_matrix.png",
                )
                plot_roc(
                    result["roc"], names,
                    f"{dataset} | Fold {number} | {model}", folder / "roc_auc.png",
                )
        if len(fold.get("models", {})) == 2 and label_reference is None:
            raise FileNotFoundError(f"Missing validation predictions in {fold_source}")
    if not fold_rows:
        print(f"No saved CV predictions for {dataset}")
        return []
    save_csv(
        output_folder / "fold_metrics.csv", fold_rows,
        ["Fold", "Model", "Accuracy (%)", "Weighted Precision",
         "Weighted Recall", "Weighted F1", "Cohen's Kappa"],
    )
    model_summaries = {}
    for model in CV_MODELS:
        values = np.asarray(
            [row["Accuracy (%)"] for row in fold_rows if row["Model"] == model],
            dtype=np.float64,
        )
        if len(values):
            model_summaries[model] = {
                "fold_count": int(len(values)),
                "mean_accuracy_pct": float(values.mean()),
                "sd_accuracy_pp": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            }
    paired = []
    for number in sorted({row["Fold"] for row in fold_rows}):
        models = {row["Model"]: row for row in fold_rows if row["Fold"] == number}
        if set(CV_MODELS).issubset(models):
            paired.append({
                "Fold": number,
                "ViT-B/16 (%)": models["ViT-B/16"]["Accuracy (%)"],
                "ViT-GCN (%)": models["ViT-GCN"]["Accuracy (%)"],
                "Improvement (pp)": (
                    models["ViT-GCN"]["Accuracy (%)"] -
                    models["ViT-B/16"]["Accuracy (%)"]
                ),
            })
    if paired:
        save_csv(
            output_folder / "vit_vs_vit_gcn_10fold.csv", paired,
            ["Fold", "ViT-B/16 (%)", "ViT-GCN (%)", "Improvement (pp)"],
        )
    differences = np.array([row["Improvement (pp)"] for row in paired], dtype=float)
    save_json(output_folder / "cross_validation_evaluation.json", {
        "dataset": dataset,
        "models": model_summaries,
        "paired_improvement": {
            "mean_pp": float(differences.mean()) if len(differences) else None,
            "sd_pp": float(differences.std(ddof=1)) if len(differences) > 1 else None,
        },
        "fold_metrics": records,
    })
    if figures:
        plot_cv_accuracy(fold_rows, dataset, output_folder / "fold_accuracy.png")
    return fold_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate fixed-split models and 10-fold ViT/ViT-GCN results"
    )
    parser.add_argument("--mode", choices=("fixed", "cv", "all"), default="all")
    parser.add_argument(
        "--dataset", choices=(*DATASET_CLASSES, "all"), default="all"
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--cv-dir", type=Path, default=Path("results_cv"))
    parser.add_argument("--output", type=Path, default=Path("evaluation_outputs"))
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--skip-fold-plots", action="store_true")
    args = parser.parse_args()
    datasets = list(DATASET_CLASSES) if args.dataset == "all" else [args.dataset]
    all_fixed = {}
    for dataset in datasets:
        if args.mode in ("fixed", "all"):
            all_fixed[dataset] = analyze_fixed(
                dataset, args.results_dir, args.output, not args.no_figures
            )
        if args.mode in ("cv", "all"):
            analyze_cv(
                dataset, args.cv_dir, args.output,
                not args.no_figures, not args.skip_fold_plots,
            )
    if args.mode in ("fixed", "all") and any(all_fixed.values()):
        combined = []
        for dataset, rows in all_fixed.items():
            combined.extend({"Dataset": dataset, **row} for row in rows)
        save_csv(
            args.output / "three_dataset_model_comparison.csv",
            combined,
            ["Dataset", "Model", "Optimizer", "LR", "Batch Size", "Train Acc",
             "Val Acc", "Test Acc", "Precision", "Recall", "F1 Score", "Cohen's Kappa"],
        )
        if not args.no_figures:
            plot_three_dataset_comparison(
                all_fixed, args.output / "three_dataset_accuracy_comparison.png"
            )


if __name__ == "__main__":
    main()
