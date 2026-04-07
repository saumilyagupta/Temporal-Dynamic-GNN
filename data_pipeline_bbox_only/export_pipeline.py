#!/usr/bin/env python3
"""
Export CarCrash NPZ ``det`` tensors to per-frame PyG graphs (bbox-only, no VGG).

Produces train/ (80%), val/ (20% stratified from train.txt), test/, and manifests.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Tuple

import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from bbox_graph_builder import EdgeStrategy, build_clip_graph_sequence, get_node_feature_dim
from npz_reader import ClipInfo, load_npz_det, read_split_file


def _class_folder(label: int) -> str:
    return "Normal" if label == 0 else "Anomalous"


def _export_one_clip(
    rel: str,
    lab: int,
    vid: str,
    npz_root: str,
    out_root: str,
    split_name: str,
    num_frames: int,
    edge_strategy: str,
) -> Dict[str, Any]:
    """Load one NPZ, build graphs, save sequence.pt; return manifest row dict."""
    npz_path = os.path.join(npz_root, rel)
    det, _ = load_npz_det(npz_path)
    graphs = build_clip_graph_sequence(
        det,
        num_frames=num_frames,
        edge_strategy=edge_strategy,
    )
    clip_dir = os.path.join(out_root, split_name, _class_folder(lab), vid)
    os.makedirs(clip_dir, exist_ok=True)
    
    # Save the entire sequence of graphs into a single file to prevent disk IO bottlenecks
    torch.save(graphs, os.path.join(clip_dir, "sequence.pt"))

    nodes_per_frame = [int(g.x.shape[0]) for g in graphs]
    max_n = max(nodes_per_frame) if nodes_per_frame else 0
    return {
        "video_folder_path": os.path.abspath(clip_dir),
        "label": _class_folder(lab),
        "source_video": vid,
        "num_frames": num_frames,
        "max_nodes_per_frame": max_n,
        "node_feature_dim": get_node_feature_dim(),
    }


def _job_tuple(
    e: ClipInfo,
    npz_root: str,
    out_root: str,
    split_name: str,
    num_frames: int,
    edge_strategy: str,
) -> Tuple[str, int, str, str, str, str, int, str]:
    return (
        e.rel_path,
        e.label,
        e.video_id,
        npz_root,
        out_root,
        split_name,
        num_frames,
        edge_strategy,
    )


def _export_one_clip_star(args: Tuple[str, int, str, str, str, str, int, str]) -> Dict[str, Any]:
    rel, lab, vid, npz_root, out_root, split_name, num_frames, edge_strategy = args
    return _export_one_clip(
        rel, lab, vid, npz_root, out_root, split_name, num_frames, edge_strategy
    )


def write_manifest(path: str, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "video_folder_path",
        "label",
        "source_video",
        "num_frames",
        "max_nodes_per_frame",
        "node_feature_dim",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def export_entries(
    entries: List[ClipInfo],
    npz_root: str,
    out_root: str,
    split_name: str,
    num_frames: int,
    edge_strategy: str,
    workers: int,
) -> List[Dict[str, Any]]:
    jobs = [
        _job_tuple(e, npz_root, out_root, split_name, num_frames, edge_strategy)
        for e in entries
    ]
    if workers <= 1:
        return [
            _export_one_clip_star(job)
            for job in tqdm(jobs, desc=f"export {split_name}")
        ]

    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(
            tqdm(
                ex.map(_export_one_clip_star, jobs),
                total=len(jobs),
                desc=f"export {split_name}",
            )
        )


def load_yaml_config(path: str) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as err:
        raise RuntimeError("PyYAML is required for --config") from err
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    p = argparse.ArgumentParser(description="CarCrash bbox-only NPZ → graph .pt export")
    p.add_argument(
        "--npz_root",
        type=str,
        default="/workspace/GNN/CarCrash/vgg16_features",
        help="Directory containing train.txt, test.txt, and NPZ files",
    )
    p.add_argument(
        "--output_root",
        type=str,
        default="/workspace/GNN/CarCrash/carcrash_bbox_graphs",
        help="Output root for train/val/test and manifests",
    )
    p.add_argument("--num_frames", type=int, default=50)
    p.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Fraction of train.txt for validation (0 disables val export)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--edge_strategy",
        type=str,
        default=EdgeStrategy.CENTROID_DIST.value,
        choices=[e.value for e in EdgeStrategy],
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML to override defaults (npz_root, output_root, ...)",
    )
    args = p.parse_args()

    if args.config:
        cfg = load_yaml_config(args.config)
        for key, val in cfg.items():
            k = key.replace("-", "_")
            if val is not None and hasattr(args, k):
                setattr(args, k, val)

    train_txt = os.path.join(args.npz_root, "train.txt")
    test_txt = os.path.join(args.npz_root, "test.txt")

    train_entries = read_split_file(train_txt)
    test_entries = read_split_file(test_txt)

    os.makedirs(args.output_root, exist_ok=True)

    val_entries: List[ClipInfo] = []
    tr_entries = train_entries
    if args.val_split > 0.0 and len(train_entries) > 1:
        labels = [e.label for e in train_entries]
        cnt = Counter(labels)
        can_stratify = len(cnt) > 1 and min(cnt.values()) >= 2
        strat = labels if can_stratify else None
        tr_idx, va_idx = train_test_split(
            range(len(train_entries)),
            test_size=args.val_split,
            stratify=strat,
            random_state=args.seed,
        )
        tr_entries = [train_entries[i] for i in tr_idx]
        val_entries = [train_entries[i] for i in va_idx]
    elif args.val_split > 0.0:
        val_entries = []

    print(f"Train clips: {len(tr_entries)}, Val clips: {len(val_entries)}, Test clips: {len(test_entries)}")

    train_rows = export_entries(
        tr_entries,
        args.npz_root,
        args.output_root,
        "train",
        args.num_frames,
        args.edge_strategy,
        args.workers,
    )
    write_manifest(os.path.join(args.output_root, "train_manifest.csv"), train_rows)

    if val_entries:
        val_rows = export_entries(
            val_entries,
            args.npz_root,
            args.output_root,
            "val",
            args.num_frames,
            args.edge_strategy,
            args.workers,
        )
        write_manifest(os.path.join(args.output_root, "val_manifest.csv"), val_rows)
    else:
        write_manifest(os.path.join(args.output_root, "val_manifest.csv"), [])

    test_rows = export_entries(
        test_entries,
        args.npz_root,
        args.output_root,
        "test",
        args.num_frames,
        args.edge_strategy,
        args.workers,
    )
    write_manifest(os.path.join(args.output_root, "test_manifest.csv"), test_rows)

    print("Done.")
    print(f"  {args.output_root}/train_manifest.csv")
    print(f"  {args.output_root}/val_manifest.csv")
    print(f"  {args.output_root}/test_manifest.csv")


if __name__ == "__main__":
    main()
