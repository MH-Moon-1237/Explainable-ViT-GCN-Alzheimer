# Explainability

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

from torch.nn.attention import SDPBackend, sdpa_kernel

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from skimage.segmentation import mark_boundaries, slic
from torch import Tensor
from torch.nn import functional as F

from data_preparation import DATASET_CLASSES, evaluation_transform
from vit_gcn import TrainingGraph, ViTGCN, build_training_graph


IMAGE_SIZE = 224
LIME_SUPERPIXELS = 50
LIME_COMPACTNESS = 10
LIME_PERTURBATIONS = 500


def normalize_heatmap(values: Tensor | np.ndarray) -> np.ndarray:
    values = torch.as_tensor(values, dtype=torch.float32).detach().cpu()
    values = values - values.min()
    maximum = values.max()
    if maximum > 0:
        values = values / maximum
    return values.numpy()


def resize_heatmap(values: Tensor | np.ndarray, size: tuple[int, int]) -> np.ndarray:
    values = torch.as_tensor(values, dtype=torch.float32)
    if values.ndim != 2:
        raise ValueError("A heatmap must be a two-dimensional array")
    values = normalize_heatmap(values)
    tensor = torch.from_numpy(values)[None, None]
    tensor = F.interpolate(
        tensor, size=(size[1], size[0]), mode="bilinear", align_corners=False
    )[0, 0]
    return normalize_heatmap(tensor)


def prepare_image(image: Image.Image, device: torch.device) -> Tensor:
    return evaluation_transform(IMAGE_SIZE)(image.convert("RGB")).unsqueeze(0).to(device)


def load_experiment(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[ViTGCN, TrainingGraph, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_name") != "ViT-GCN":
        raise ValueError("The checkpoint must belong to the trained ViT-GCN model")
    for key in ("num_classes", "model_state", "train_embeddings", "k"):
        if key not in checkpoint:
            raise ValueError(f"Checkpoint is missing: {key}")

    model = ViTGCN(num_classes=int(checkpoint["num_classes"]), pretrained=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    graph = build_training_graph(
        checkpoint["train_embeddings"].float(), k=int(checkpoint["k"])
    )
    return model, graph, checkpoint


class HybridExplainer:
    def __init__(
        self,
        model: ViTGCN,
        training_graph: TrainingGraph,
        device: torch.device | str | None = None,
    ) -> None:
        if not callable(getattr(model, "forward_independent", None)):
            raise RuntimeError("Use the corrected vit_gcn.py with forward_independent().")
        self.model = model.eval()
        self.training_graph = training_graph
        self.device = torch.device(device) if device else next(model.parameters()).device
        self.target_layer = self.model.vit.vit.encoder.layers[-1].ln_1
        self.attention_layer = self.model.vit.vit.encoder.layers[-1].self_attention

    @torch.no_grad()
    def probabilities(self, images: Tensor) -> Tensor:
        logits = self.model.forward_independent(images.to(self.device), self.training_graph)
        return logits.softmax(dim=1)

    def lime(
        self,
        image: Image.Image,
        num_samples: int = LIME_PERTURBATIONS,
        num_features: int = 3,
        seed: int = 42,
        batch_size: int = 8,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        try:
            from lime.lime_image import LimeImageExplainer
        except ImportError as error:
            raise ImportError("LIME requires the 'lime' package: pip install lime") from error

        original = np.asarray(image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE)))
        segmentation: Callable[[np.ndarray], np.ndarray] = lambda array: slic(
            array, n_segments=LIME_SUPERPIXELS,
            compactness=LIME_COMPACTNESS, start_label=0,
        )

        def predict(array_batch: np.ndarray) -> np.ndarray:
            tensors = torch.stack([
                evaluation_transform(IMAGE_SIZE)(
                    Image.fromarray(np.uint8(np.clip(array, 0, 255)))
                )
                for array in array_batch
            ]).to(self.device)
            return self.probabilities(tensors).cpu().numpy()

        explanation = LimeImageExplainer(random_state=seed).explain_instance(
            original,
            classifier_fn=predict,
            top_labels=1,
            num_features=num_features,
            num_samples=num_samples,
            batch_size=batch_size,
            hide_color=0,
            segmentation_fn=segmentation,
        )
        predicted_class = int(explanation.top_labels[0])
        visual, mask = explanation.get_image_and_mask(
            predicted_class,
            positive_only=True,
            hide_rest=False,
            num_features=num_features,
        )
        visual = np.asarray(visual, dtype=np.float32)
        if visual.max() > 1:
            visual /= 255.0
        marked = mark_boundaries(visual, mask, color=(1, 0, 0), mode="thick")
        return marked, np.asarray(mask, dtype=np.uint8), predicted_class

    def _gradients(
        self,
        images: Tensor,
        class_index: int | None = None,
        higher_order_probes: int = 4,
        seed: int = 42,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, int]:
        """Estimate diagonal higher-order derivatives with Rademacher probes."""
        if images.shape != (1, 3, IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError("Grad-CAM methods require one 224x224 RGB image")
        if higher_order_probes < 1:
            raise ValueError("higher_order_probes must be at least 1")

        activations: dict[str, Tensor] = {}

        def capture(_module, _inputs, output: Tensor) -> None:
            activations["tokens"] = output

        hook = self.target_layer.register_forward_hook(capture)
        try:
            with torch.enable_grad(), sdpa_kernel([SDPBackend.MATH]):
                logits = self.model.forward_independent(
                    images.to(self.device), self.training_graph, batch_size=1
                )
                index = int(logits.argmax(dim=1).item()) if class_index is None else int(class_index)
                if not 0 <= index < logits.size(1):
                    raise ValueError(f"Invalid class index: {index}")

                tokens = activations["tokens"]
                if tokens.shape[1] != 197:
                    raise RuntimeError("ViT-B/16 must provide 196 patch tokens and one CLS token")

                class_logit = logits[0, index]
                cam_grad = torch.autograd.grad(
                    class_logit, tokens, retain_graph=True
                )[0]

                centered_exp_score = torch.exp(class_logit - class_logit.detach())
                first = torch.autograd.grad(
                    centered_exp_score, tokens, create_graph=True, retain_graph=True
                )[0]

                random = torch.Generator(device=tokens.device)
                random.manual_seed(seed)
                second_est = torch.zeros_like(tokens)
                third_est = torch.zeros_like(tokens)

                for probe in range(higher_order_probes):
                    directions = [
                        torch.randint(
                            0, 2, tokens.shape,
                            generator=random, device=tokens.device
                        ).to(tokens.dtype).mul_(2).sub_(1)
                        for _ in range(2)
                    ]
                    z, w = directions
                    hessian_z = torch.autograd.grad(
                        first, tokens, grad_outputs=z,
                        create_graph=True, retain_graph=True
                    )[0]
                    third_zw = torch.autograd.grad(
                        hessian_z, tokens, grad_outputs=w,
                        retain_graph=probe < higher_order_probes - 1
                    )[0]
                    second_est.add_(z * hessian_z.detach())
                    third_est.add_(z * w * third_zw.detach())

                second_est.div_(higher_order_probes)
                third_est.div_(higher_order_probes)

        finally:
            hook.remove()

        return (
            tokens.detach()[:, 1:],
            cam_grad.detach()[:, 1:],
            first.detach()[:, 1:],
            second_est[:, 1:],
            third_est[:, 1:],
            index,
        )

    def grad_cam_maps(
        self,
        image: Image.Image,
        class_index: int | None = None,
        higher_order_probes: int = 4,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        tensor = prepare_image(image, self.device)
        features, gradients, first, second, third, predicted_class = self._gradients(
            tensor, class_index, higher_order_probes=higher_order_probes, seed=seed
        )

        activations = features[0]
        grad_cam_weights = gradients[0].mean(dim=0)
        cam = F.relu((activations * grad_cam_weights).sum(dim=1)).reshape(14, 14)

        first, second, third = first[0], second[0], third[0]
        denominator = 2 * second + activations.sum(dim=0, keepdim=True) * third
        epsilon = torch.full_like(denominator, 1e-8)
        signed_epsilon = torch.where(denominator >= 0, epsilon, -epsilon)
        denominator = torch.where(denominator.abs() < 1e-8, signed_epsilon, denominator)
        alpha = torch.where(second > 0, second / denominator, torch.zeros_like(second))
        alpha = torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)
        weights_plus = (alpha * F.relu(first)).sum(dim=0)
        cam_plus = F.relu((activations * weights_plus).sum(dim=1)).reshape(14, 14)

        return (
            resize_heatmap(cam, image.size),
            resize_heatmap(cam_plus, image.size),
            predicted_class,
        )

    @torch.no_grad()
    def attention_map(self, image: Image.Image) -> np.ndarray:
        stored: dict[str, Tensor] = {}

        def capture(_module, _inputs, output: Tensor) -> None:
            stored["tokens"] = output.detach()

        hook = self.target_layer.register_forward_hook(capture)
        try:
            tensor = prepare_image(image, self.device)
            self.model.forward_independent(tensor, self.training_graph, batch_size=1)
        finally:
            hook.remove()

        tokens = stored["tokens"]
        attention = self.attention_layer
        if attention.in_proj_weight is None:
            raise RuntimeError("Expected fused Q/K/V projection weights")
        qkv = F.linear(tokens, attention.in_proj_weight, attention.in_proj_bias)
        queries, keys, _ = qkv.chunk(3, dim=-1)
        batch, length, dim = queries.shape
        heads = attention.num_heads
        depth = dim // heads
        queries = queries.reshape(batch, length, heads, depth).transpose(1, 2)
        keys = keys.reshape(batch, length, heads, depth).transpose(1, 2)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(depth)
        attention_weights = scores.softmax(dim=-1)
        cls_to_patches = attention_weights[0, :, 0, 1:].mean(dim=0)
        if cls_to_patches.numel() != 196:
            raise RuntimeError("Expected 14x14 ViT-B/16 patch attention")
        return resize_heatmap(cls_to_patches.reshape(14, 14), image.size)


def save_heatmap(image: Image.Image, heatmap: np.ndarray, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image.convert("RGB"))
    ax.imshow(heatmap, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    ax.axis("off")
    fig.savefig(output, bbox_inches="tight", pad_inches=0, dpi=200)
    plt.close(fig)


def save_lime(visual: np.ndarray, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(np.clip(visual, 0, 1))
    ax.axis("off")
    fig.savefig(output, bbox_inches="tight", pad_inches=0, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hybrid ViT-GCN LIME, Attention, Grad-CAM, and Grad-CAM++"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("xai_output"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lime-samples", type=int, default=LIME_PERTURBATIONS)
    parser.add_argument("--skip-lime", action="store_true")
    parser.add_argument("--higher-order-probes", type=int, default=4)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, graph, checkpoint = load_experiment(args.checkpoint, device)
    explainer = HybridExplainer(model, graph, device)
    with Image.open(args.image) as image_file:
        image = image_file.convert("RGB")
    with torch.no_grad():
        probabilities = explainer.probabilities(prepare_image(image, device))[0].cpu()
    predicted_class = int(probabilities.argmax().item())

    cam, cam_plus, cam_class = explainer.grad_cam_maps(
        image, predicted_class, higher_order_probes=args.higher_order_probes, seed=args.seed
    )
    attention = explainer.attention_map(image)
    save_heatmap(image, cam, args.output / "grad_cam.png")
    save_heatmap(image, cam_plus, args.output / "grad_cam_plus_plus.png")
    save_heatmap(image, attention, args.output / "attention_map.png")

    if not args.skip_lime:
        lime_visual, _, _ = explainer.lime(
            image, num_samples=args.lime_samples, seed=args.seed
        )
        save_lime(lime_visual, args.output / "lime.png")

    dataset = checkpoint.get("dataset")
    labels = (
        [name for name, _ in DATASET_CLASSES[dataset]]
        if dataset in DATASET_CLASSES else None
    )
    results = {
        "dataset": dataset,
        "predicted_class": predicted_class,
        "predicted_label": labels[predicted_class] if labels else str(predicted_class),
        "probabilities": probabilities.numpy().tolist(),
        "grad_cam_class": cam_class,
        "lime_samples": None if args.skip_lime else args.lime_samples,
    }
    (args.output / "xai_prediction.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
