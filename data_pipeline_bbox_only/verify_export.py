#!/usr/bin/env python3
"""
Sample exported graph_*.pt files and manifests; report basic stats.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from typing import Any, Dict, List

import torch


def _read_manifest_rows(manifest_path: str) -> List[Dict[str, str]]:
    with open(manifest_path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _sample_folder_paths(rows: List[Dict[str, str]], sample_size: int, seed: int) -> List[str]:
    if not rows:
        return []
    rng = random.Random(seed)
    n = min(sample_size, len(rows))
    sampled = rng.sample(rows, n)
    return [str(r["video_folder_path"]) for r in sampled]


def _stats_for_folder(folder: str, num_frames: int) -> Dict[str, Any]:
    nodes: List[int] = []
    edges: List[int] = []
    feat_dims: set[int] = set()
    empty_frames = 0
    
    seq_path = os.path.join(folder, "sequence.pt")
    if os.path.isfile(seq_path):
        graphs = torch.load(seq_path, weights_only=False)
        for data in graphs[:num_frames]:
            n = int(data.x.shape[0]) if data.x is not None else 0
            ei = data.edge_index
            e = int(ei.shape[1]) if ei is not None else 0
            nodes.append(n)
            edges.append(e)
            if data.x is not None and data.x.ndim == 2:
                feat_dims.add(int(data.x.shape[1]))
            if n == 0:
                empty_frames += 1
                
    return {
        "nodes_min": min(nodes) if nodes else 0,
        "nodes_max": max(nodes) if nodes else 0,
        "nodes_mean": float(sum(nodes) / len(nodes)) if nodes else 0.0,
        "edges_mean": float(sum(edges) / len(edges)) if edges else 0.0,
        "feature_dims": sorted(feat_dims),
        "empty_frames": empty_frames,
        "frames_checked": len(nodes),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Verify bbox graph export")
    p.add_argument(
        "--export_root",
        type=str,
        default="/workspace/GNN/CarCrash/carcrash_bbox_graphs",
    )
    p.add_argument("--sample_size", type=int, default=5)
    p.add_argument("--num_frames", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    for name in ("train_manifest.csv", "val_manifest.csv", "test_manifest.csv"):
        mp = os.path.join(args.export_root, name)
        if not os.path.isfile(mp):
            print(f"Missing {mp}")
            continue
        rows = _read_manifest_rows(mp)
        print(f"\n=== {name} rows={len(rows)} ===")
        if not rows:
            continue
        folders = _sample_folder_paths(rows, args.sample_size, args.seed)
        for folder in folders:
            if not os.path.isdir(folder):
                print(f"  missing dir: {folder}")
                continue
            st = _stats_for_folder(folder, args.num_frames)
            print(f"  {folder}")
            print(
                f"    nodes: min={st['nodes_min']} max={st['nodes_max']} "
                f"mean={st['nodes_mean']:.2f}"
            )
            print(
                f"    edges mean={st['edges_mean']:.1f} feature_dims={st['feature_dims']} "
                f"empty_frames={st['empty_frames']}/{st['frames_checked']}"
            )


if __name__ == "__main__":
    main()
