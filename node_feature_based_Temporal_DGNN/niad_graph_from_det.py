"""
NiAD-style PyG graphs from per-frame bbox tensors (no YOLO).

Node features [8]: centroid_x, centroid_y, bbox_x, bbox_y, bbox_w, bbox_h,
class_id, normalized_timestamp — aligned with
``data_pipeline_node_feature_based/vehicle_processor.create_frame_graph``.

CarCrash NPZ ``det`` layout: (T, 19, 6) with columns
``x1, y1, x2, y2, aux_a, aux_b`` (pixels; class id from aux columns if non-zero).
"""

from __future__ import annotations

import os
from typing import List

import numpy as np
import torch
from torch_geometric.data import Data


def _class_from_det_row(row: np.ndarray) -> float:
    a, b = float(row[4]), float(row[5])
    if a != 0.0:
        return a
    if b != 0.0:
        return b
    return 0.0


def build_niad_graph_from_det_frame(
    det_frame: np.ndarray,
    frame_idx: int,
    total_frames: int,
) -> Data:
    """
    Build one NiAD-style ``Data`` from a single frame's detections.

    Args:
        det_frame: (N, 6+) array — one row per slot (e.g. 19 vehicles).
        frame_idx: Frame index for normalized timestamp.
        total_frames: T in the clip (for t / (T-1)).
    """
    if det_frame.ndim != 2 or det_frame.shape[1] < 6:
        raise ValueError(f"det_frame must be (N, 6+), got {det_frame.shape}")

    normalized_t = frame_idx / (total_frames - 1) if total_frames > 1 else 0.0

    node_features: List[List[float]] = []
    node_id_list: List[int] = []
    centroids: List[tuple] = []

    n_slots = det_frame.shape[0]
    for slot in range(n_slots):
        row = det_frame[slot]
        x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        if x2 <= x1 + 1e-3 or y2 <= y1 + 1e-3:
            continue

        bbox_x, bbox_y = x1, y1
        w, h = x2 - x1, y2 - y1
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        class_id = _class_from_det_row(row)

        node_features.append(
            [
                cx,
                cy,
                bbox_x,
                bbox_y,
                w,
                h,
                class_id,
                float(normalized_t),
            ]
        )
        node_id_list.append(slot)
        centroids.append((cx, cy))

    num_nodes = len(node_features)
    if num_nodes == 0:
        return Data(
            x=torch.zeros((0, 8), dtype=torch.float32),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_weight=torch.zeros((0,), dtype=torch.float32),
            node_ids=torch.zeros((0,), dtype=torch.long),
            t=torch.tensor([frame_idx], dtype=torch.long),
        )

    x = torch.tensor(node_features, dtype=torch.float32)
    node_ids = torch.tensor(node_id_list, dtype=torch.long)
    centroids_array = np.array(centroids, dtype=np.float32)

    if num_nodes > 1:
        edge_list: List[List[int]] = []
        edge_weights: List[float] = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    continue
                edge_list.append([i, j])
                dist = float(np.linalg.norm(centroids_array[i] - centroids_array[j]))
                edge_weights.append(dist)
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weights, dtype=torch.float32)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_weight = torch.zeros((0,), dtype=torch.float32)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_weight=edge_weight,
        node_ids=node_ids,
        t=torch.tensor([frame_idx], dtype=torch.long),
    )


def load_npz_niad_graph_sequence(
    npz_path: str,
    num_frames: int = 50,
    feature_mean: np.ndarray | None = None,
    feature_std: np.ndarray | None = None,
) -> List[Data]:
    """Load CarCrash NPZ; build NiAD-style graphs from ``det`` only (no VGG)."""
    bundle = np.load(npz_path, allow_pickle=True)
    det = bundle["det"]
    graphs: List[Data] = []
    for t in range(num_frames):
        g = build_niad_graph_from_det_frame(det[t], t, num_frames)
        if feature_mean is not None and feature_std is not None and g.x.shape[0] > 0:
            xm = torch.tensor(feature_mean, dtype=g.x.dtype, device=g.x.device)
            xs = torch.tensor(feature_std, dtype=g.x.dtype, device=g.x.device)
            g.x = (g.x - xm) / xs
        graphs.append(g)
    return graphs


def save_graph_sequence_to_folder(graphs: List[Data], folder: str) -> None:
    os.makedirs(folder, exist_ok=True)
    for t, g in enumerate(graphs):
        torch.save(g, os.path.join(folder, f"graph_{t:03d}.pt"))
