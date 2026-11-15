# Proposed ViT-GCN Architecture

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.neighbors import NearestNeighbors
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import ViT_B_16_Weights, vit_b_16


EMBEDDING_DIM = 768
HIDDEN_DIM = 256
NEIGHBORS = 5


def _validate_embeddings(embeddings: Tensor, min_samples: int = 1) -> Tensor:
    if not isinstance(embeddings, Tensor) or embeddings.ndim != 2:
        raise ValueError("Expected a two-dimensional tensor of ViT embeddings")
    if embeddings.size(1) != EMBEDDING_DIM:
        raise ValueError(f"Expected {EMBEDDING_DIM}-dimensional ViT embeddings")
    if embeddings.size(0) < min_samples:
        raise ValueError(f"At least {min_samples} embeddings are required")
    if not torch.isfinite(embeddings).all():
        raise ValueError("Embeddings contain non-finite values")
    return embeddings


class ViTBackbone(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        self.vit = vit_b_16(weights=weights, image_size=224)
        self.vit.heads.head = nn.Linear(EMBEDDING_DIM, num_classes)

    def extract_features(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 224, 224):
            raise ValueError("ViT-B/16 expects images of shape [N, 3, 224, 224]")
        tokens = self.vit._process_input(images)
        class_tokens = self.vit.class_token.expand(tokens.size(0), -1, -1)
        tokens = torch.cat((class_tokens, tokens), dim=1)
        tokens = self.vit.encoder(tokens)
        return tokens[:, 0]

    def forward(self, images: Tensor) -> Tensor:
        return self.vit.heads(self.extract_features(images))


@dataclass
class TrainingGraph:
    features: Tensor
    adjacency: Tensor
    degrees: Tensor
    edge_index: Tensor
    mean: Tensor
    std: Tensor
    nearest_neighbors: NearestNeighbors
    k: int

    def standardize(self, embeddings: Tensor) -> Tensor:
        embeddings = _validate_embeddings(embeddings)
        mean = self.mean.to(device=embeddings.device, dtype=embeddings.dtype)
        std = self.std.to(device=embeddings.device, dtype=embeddings.dtype)
        return (embeddings - mean) / std


def build_training_graph(
    train_embeddings: Tensor,
    k: int = NEIGHBORS,
) -> TrainingGraph:
    features = _validate_embeddings(train_embeddings, min_samples=k + 1)
    if k < 1:
        raise ValueError("k must be positive")

    features = features.detach().to(device="cpu", dtype=torch.float32)
    mean = features.mean(dim=0)
    std = features.std(dim=0, unbiased=False).clamp_min(1e-6)
    features = (features - mean) / std
    features_np = features.numpy()

    index = NearestNeighbors(metric="cosine", algorithm="brute")
    index.fit(features_np)
    candidate_indices = index.kneighbors(
        features_np, n_neighbors=k + 1, return_distance=False
    )

    rows: list[int] = []
    cols: list[int] = []
    for row, candidates in enumerate(candidate_indices):
        neighbors = [int(col) for col in candidates if int(col) != row][:k]
        if len(neighbors) != k:
            raise RuntimeError("Could not construct k distinct non-self neighbors")
        rows.extend([row] * k)
        cols.extend(neighbors)

    n = features.size(0)
    adjacency = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n, n),
        dtype=np.float32,
    )
    adjacency = adjacency.maximum(adjacency.T).tocsr()
    adjacency.setdiag(1.0)
    adjacency.eliminate_zeros()

    row_indices, col_indices = adjacency.nonzero()
    edge_index = torch.from_numpy(
        np.stack([row_indices, col_indices], axis=0).astype(np.int64)
    )

    degrees_np = np.asarray(adjacency.sum(axis=1)).reshape(-1).astype(np.float32)
    inverse_sqrt = np.power(degrees_np, -0.5)
    diagonal = sp.diags(inverse_sqrt)
    adjacency = (diagonal @ adjacency @ diagonal).tocoo()

    indices = torch.from_numpy(
        np.stack([adjacency.row, adjacency.col], axis=0).astype(np.int64)
    )
    values = torch.from_numpy(adjacency.data.astype(np.float32))
    normalized = torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()

    return TrainingGraph(
        features=features,
        adjacency=normalized,
        degrees=torch.from_numpy(degrees_np),
        edge_index=edge_index,
        mean=mean,
        std=std,
        nearest_neighbors=index,
        k=k,
    )


class GCNClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_dim: int = EMBEDDING_DIM,
        hidden_dim: int = HIDDEN_DIM,
    ):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, features: Tensor, adjacency: Tensor) -> Tensor:
        if adjacency.shape != (features.size(0), features.size(0)):
            raise ValueError("Adjacency and node feature dimensions do not match")
        if adjacency.is_sparse:
            aggregation = torch.sparse.mm
        else:
            aggregation = torch.matmul
        hidden = F.relu(self.fc1(aggregation(adjacency, features)))
        return self.fc2(aggregation(adjacency, hidden))

    def forward_independent(
        self,
        query_embeddings: Tensor,
        training_graph: TrainingGraph,
        batch_size: int = 128,
    ) -> Tensor:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        query_embeddings = _validate_embeddings(query_embeddings)
        device = self.fc1.weight.device
        dtype = self.fc1.weight.dtype

        reference_features = training_graph.features.to(device=device, dtype=dtype)
        reference_degrees = training_graph.degrees.to(device=device, dtype=dtype)
        reference_edges = training_graph.edge_index.to(device=device)
        standardized_queries = training_graph.standardize(
            query_embeddings.to(device=device, dtype=dtype)
        )
        neighbors_np = training_graph.nearest_neighbors.kneighbors(
            standardized_queries.detach().cpu().numpy(),
            n_neighbors=training_graph.k,
            return_distance=False,
        )

        reference_count = reference_features.size(0)
        query_index = reference_count
        outputs: list[Tensor] = []

        for start in range(0, standardized_queries.size(0), batch_size):
            stop = min(start + batch_size, standardized_queries.size(0))
            for query, neighbors_row in zip(
                standardized_queries[start:stop], neighbors_np[start:stop]
            ):
                neighbors = torch.as_tensor(
                    neighbors_row, device=device, dtype=torch.long
                )
                degrees = reference_degrees.clone()
                degrees[neighbors] += 1.0
                degrees = torch.cat(
                    [degrees, degrees.new_tensor([training_graph.k + 1])]
                )
                new_query_index = torch.full(
                    (training_graph.k,), query_index,
                    device=device, dtype=torch.long,
                )
                query_self = torch.tensor(
                    [query_index], device=device, dtype=torch.long
                )
                edge_sources = torch.cat(
                    [reference_edges[0], neighbors, new_query_index, query_self]
                )
                edge_targets = torch.cat(
                    [reference_edges[1], new_query_index, neighbors, query_self]
                )
                weights = torch.rsqrt(
                    degrees[edge_sources] * degrees[edge_targets]
                )
                adjacency = torch.sparse_coo_tensor(
                    torch.stack([edge_sources, edge_targets], dim=0),
                    weights,
                    size=(query_index + 1, query_index + 1),
                    device=device,
                ).coalesce()
                node_features = torch.cat(
                    [reference_features, query.unsqueeze(0)], dim=0
                )
                outputs.append(self.forward(node_features, adjacency)[-1])

        return torch.stack(outputs, dim=0)

    @torch.no_grad()
    def predict_independent(
        self,
        query_embeddings: Tensor,
        training_graph: TrainingGraph,
        batch_size: int = 128,
    ) -> Tensor:
        self.eval()
        return self.forward_independent(
            query_embeddings, training_graph, batch_size=batch_size
        )


class ViTGCN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        hidden_dim: int = HIDDEN_DIM,
    ):
        super().__init__()
        self.vit = ViTBackbone(num_classes=num_classes, pretrained=pretrained)
        self.gcn = GCNClassifier(
            num_classes=num_classes,
            input_dim=EMBEDDING_DIM,
            hidden_dim=hidden_dim,
        )

    def forward_independent(
        self,
        images: Tensor,
        training_graph: TrainingGraph,
        batch_size: int = 128,
    ) -> Tensor:
        embeddings = self.vit.extract_features(images)
        return self.gcn.forward_independent(
            embeddings, training_graph, batch_size=batch_size
        )

    @torch.no_grad()
    def predict_independent(
        self,
        images: Tensor,
        training_graph: TrainingGraph,
        batch_size: int = 128,
    ) -> Tensor:
        self.eval()
        return self.forward_independent(
            images, training_graph, batch_size=batch_size
        )


@torch.no_grad()
def extract_embeddings(
    vit: ViTBackbone,
    loader: torch.utils.data.DataLoader,
    device: Optional[torch.device | str] = None,
) -> tuple[Tensor, Tensor]:
    if device is None:
        device = next(vit.parameters()).device
    was_training = vit.training
    vit.eval()
    features: list[Tensor] = []
    labels: list[Tensor] = []

    for images, targets in loader:
        images = images.to(device)
        features.append(vit.extract_features(images).detach().cpu())
        labels.append(targets.detach().cpu().long())

    vit.train(was_training)
    if not features:
        raise ValueError("Cannot extract embeddings from an empty data loader")
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)
