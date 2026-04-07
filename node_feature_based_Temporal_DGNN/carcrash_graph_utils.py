"""
Shared CarCrash graph construction from NPZ (VGG16 bbox embeddings + det bboxes).

Each frame: 19 vehicle nodes, fully connected directed edges, edge_weight = centroid distance.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from torch_geometric.data import Data


def build_frame_graph(
    node_features: np.ndarray,
    det_row: np.ndarray,
) -> Data:
    """
    One frame: 19 nodes (bbox embeddings), fully connected directed edges,
    edge_weight = Euclidean distance between bbox centroids in image space.
    """
    x = torch.from_numpy(node_features.astype(np.float32))
    n = x.shape[0]
    x1, y1, x2, y2 = det_row[:, 0], det_row[:, 1], det_row[:, 2], det_row[:, 3]
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    centroids = np.stack([cx, cy], axis=1)

    src: List[int] = []
    dst: List[int] = []
    weights: List[float] = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = float(np.linalg.norm(centroids[i] - centroids[j]))
            src.append(i)
            dst.append(j)
            weights.append(d)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, edge_weight=edge_weight)


def load_npz_graph_sequence(
    npz_path: str,
    num_frames: int = 50,
    feature_mean: Optional[np.ndarray] = None,
    feature_std: Optional[np.ndarray] = None,
) -> List[Data]:
    """Load one video as a list of per-frame graphs from a CarCrash .npz bundle."""
    bundle = np.load(npz_path, allow_pickle=True)
    data = bundle["data"]  # (T, 20, 4096)
    det = bundle["det"]  # (T, 19, 6)
    graphs: List[Data] = []
    for t in range(num_frames):
        box_feats = data[t, 1:, :]  # (19, 4096)
        if feature_mean is not None and feature_std is not None:
            box_feats = (box_feats - feature_mean) / feature_std
        g = build_frame_graph(box_feats, det[t])
        graphs.append(g)
    return graphs


def save_graph_sequence_to_folder(
    graphs: List[Data],
    folder: str,
) -> None:
    """Write graph_{t:03d}.pt for each frame (used by the NPZ export pipeline)."""
    import os

    os.makedirs(folder, exist_ok=True)
    for t, g in enumerate(graphs):
        path = os.path.join(folder, f"graph_{t:03d}.pt")
        torch.save(g, path)
