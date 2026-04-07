"""
Load CarCrash NPZ detection tensors and split-file entries.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class ClipInfo:
    """One clip from train.txt / test.txt."""

    rel_path: str
    label: int
    video_id: str


def _parse_txt_line(line: str) -> ClipInfo:
    line = line.strip()
    if not line:
        raise ValueError("empty line")
    parts = line.split()
    rel_path = parts[0]
    label = int(parts[1])
    base = os.path.basename(rel_path).replace(".npz", "")
    video_id = base.zfill(6)
    return ClipInfo(rel_path=rel_path, label=label, video_id=video_id)


def read_split_file(txt_path: str) -> List[ClipInfo]:
    """Load all ClipInfo rows from train.txt or test.txt."""
    entries: List[ClipInfo] = []
    with open(txt_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(_parse_txt_line(line))
    return entries


def load_npz_det(npz_path: str) -> Tuple[np.ndarray, int]:
    """
    Load ``det`` and binary label from a CarCrash .npz bundle.

    Returns:
        det: (T, 19, 6) array — x1, y1, x2, y2, conf, cls
        label: 0 = Normal, 1 = Anomalous (from one-hot ``labels`` if present, else 0)
    """
    bundle = np.load(npz_path, allow_pickle=True)
    if "det" not in bundle:
        raise KeyError(f"'det' missing in {npz_path}")
    det = np.asarray(bundle["det"], dtype=np.float32)

    if det.ndim != 3 or det.shape[2] < 6:
        raise ValueError(f"Expected det (T, N, 6+), got {det.shape} in {npz_path}")

    label = 0
    if "labels" in bundle:
        lab = np.asarray(bundle["labels"]).reshape(-1)
        if lab.size >= 2:
            # [1,0] normal, [0,1] accident per README
            if lab[0] < lab[1]:
                label = 1
            else:
                label = 0
        elif lab.size == 1:
            label = int(lab[0])

    return det, label
