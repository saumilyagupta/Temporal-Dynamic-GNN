"""
Build PyG graphs from CarCrash ``det`` arrays (bounding boxes only, no VGG).

Node features (13-D): cx, cy, w, h, area, aspect_ratio, x1, y1, x2, y2, conf, cls, normalized_t.
Edges: fully connected directed (excluding self-loops).
"""

from __future__ import annotations

from enum import Enum
from typing import List

import numpy as np
import torch
from torch_geometric.data import Data

NODE_FEATURE_DIM = 13


class EdgeStrategy(str, Enum):
    CENTROID_DIST = "centroid_dist"
    IOU = "iou"
    BOTH = "both"


def _valid_box(row: np.ndarray) -> bool:
    x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
    return x2 > x1 + 1e-3 and y2 > y1 + 1e-3


def _bbox_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """IoU for two boxes [x1,y1,x2,y2]."""
    ax1, ay1, ax2, ay2 = float(a[0]), float(a[1]), float(a[2]), float(a[3])
    bx1, by1, bx2, by2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 1e-12:
        return 0.0
    return float(inter / union)


def build_frame_graph(
    det_frame: np.ndarray,
    frame_idx: int,
    total_frames: int,
    edge_strategy: EdgeStrategy | str = EdgeStrategy.CENTROID_DIST,
) -> Data:
    """
    One frame: variable number of nodes from valid detection rows.

    Args:
        det_frame: (N, 6) with x1, y1, x2, y2, conf, cls.
        frame_idx: Current frame index.
        total_frames: T (for normalized timestamp).
        edge_strategy: How to set edge_weight / edge_attr.
    """
    if isinstance(edge_strategy, str):
        edge_strategy = EdgeStrategy(edge_strategy)

    if det_frame.ndim != 2 or det_frame.shape[1] < 6:
        raise ValueError(f"det_frame must be (N, 6+), got {det_frame.shape}")

    normalized_t = frame_idx / (total_frames - 1) if total_frames > 1 else 0.0

    rows_kept: List[np.ndarray] = []
    centroids: List[tuple[float, float]] = []
    corners: List[np.ndarray] = []

    n_slots = det_frame.shape[0]
    for slot in range(n_slots):
        row = det_frame[slot]
        if not _valid_box(row):
            continue
        x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        conf = float(row[4])
        cls = float(row[5])
        w = x2 - x1
        h = y2 - y1
        area = w * h
        aspect = (w / h) if h > 1e-6 else 0.0
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        feat = np.array(
            [
                cx,
                cy,
                w,
                h,
                area,
                aspect,
                x1,
                y1,
                x2,
                y2,
                conf,
                cls,
                float(normalized_t),
            ],
            dtype=np.float32,
        )
        rows_kept.append(feat)
        centroids.append((cx, cy))
        corners.append(np.array([x1, y1, x2, y2], dtype=np.float32))

    num_nodes = len(rows_kept)
    if num_nodes == 0:
        return Data(
            x=torch.zeros((0, NODE_FEATURE_DIM), dtype=torch.float32),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_weight=torch.zeros((0,), dtype=torch.float32),
            t=torch.tensor([frame_idx], dtype=torch.long),
        )

    x = torch.from_numpy(np.stack(rows_kept, axis=0))
    centroids_array = np.array(centroids, dtype=np.float32)

    src: List[int] = []
    dst: List[int] = []
    weights: List[float] = []
    ious: List[float] = []

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue
            d = float(np.linalg.norm(centroids_array[i] - centroids_array[j]))
            # Convert pixel distance to similarity score in (0, 1]
            sim = float(np.exp(-d / 100.0))
            iou = _bbox_iou_xyxy(corners[i], corners[j])
            
            src.append(i)
            dst.append(j)
            
            if edge_strategy == EdgeStrategy.CENTROID_DIST or edge_strategy == EdgeStrategy.BOTH:
                weights.append(sim)
            elif edge_strategy == EdgeStrategy.IOU:
                weights.append(iou)
                
            ious.append(iou)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float32)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_weight=edge_weight,
        t=torch.tensor([frame_idx], dtype=torch.long),
    )

    if edge_strategy == EdgeStrategy.BOTH:
        data.edge_attr = torch.tensor(ious, dtype=torch.float32).view(-1, 1)

    return data


def build_clip_graph_sequence(
    det_array: np.ndarray,
    num_frames: int,
    edge_strategy: EdgeStrategy | str = EdgeStrategy.CENTROID_DIST,
) -> List[Data]:
    """
    Build a list of per-frame graphs from det with shape (T, 19, 6) or more frames.
    """
    if det_array.ndim != 3 or det_array.shape[2] < 6:
        raise ValueError(f"det_array must be (T, N, 6+), got {det_array.shape}")

    t_max = min(int(det_array.shape[0]), num_frames)
    graphs: List[Data] = []
    for t in range(t_max):
        g = build_frame_graph(det_array[t], t, num_frames, edge_strategy=edge_strategy)
        graphs.append(g)
    return graphs


def get_node_feature_dim() -> int:
    return NODE_FEATURE_DIM
