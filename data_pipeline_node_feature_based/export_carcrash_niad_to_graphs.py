#!/usr/bin/env python3
"""
Export CarCrash NPZ → NiAD-style graph_*.pt (8-D nodes from ``det`` only).

Same folder layout as export_carcrash_npz_to_graphs.py but uses
``niad_graph_from_det.load_npz_niad_graph_sequence`` (no VGG features).

Training: set config_carcrash_niad.yaml source=precomputed and graph_export_root to --output_root.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

_NF = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "node_feature_based_Temporal_DGNN"))
if _NF not in sys.path:
    sys.path.insert(0, _NF)

from dataset_carcrash import load_split_entries  # noqa: E402
from niad_graph_from_det import (  # noqa: E402
    load_npz_niad_graph_sequence,
    save_graph_sequence_to_folder,
)


def _class_folder(label: int) -> str:
    return "Normal" if label == 0 else "Anomalous"


def export_split(entries, npz_root: str, out_root: str, split_name: str, num_frames: int) -> list[dict]:
    rows: list[dict] = []
    for rel, lab, vid in entries:
        npz_path = os.path.join(npz_root, rel)
        clip_dir = os.path.join(out_root, split_name, _class_folder(lab), vid)
        graphs = load_npz_niad_graph_sequence(npz_path, num_frames=num_frames)
        save_graph_sequence_to_folder(graphs, clip_dir)
        nodes_per_frame = [int(g.x.shape[0]) for g in graphs]
        max_n = max(nodes_per_frame) if nodes_per_frame else 0
        rows.append(
            {
                "video_folder_path": os.path.abspath(clip_dir),
                "label": _class_folder(lab),
                "source_video": vid,
                "num_vehicles": 19,
                "num_frames": num_frames,
                "max_nodes_per_frame": max_n,
            }
        )
    return rows


def write_manifest(path: str, rows: list[dict]) -> None:
    fieldnames = [
        "video_folder_path",
        "label",
        "source_video",
        "num_vehicles",
        "num_frames",
        "max_nodes_per_frame",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    p = argparse.ArgumentParser(description="CarCrash NPZ → NiAD-style graph .pt export")
    p.add_argument("--npz_root", type=str, default="/workspace/GNN/CarCrash/vgg16_features")
    p.add_argument(
        "--output_root",
        type=str,
        default="/workspace/GNN/CarCrash/carcrash_niad_graphs_export",
    )
    p.add_argument("--num_frames", type=int, default=50)
    args = p.parse_args()

    train_entries = load_split_entries(os.path.join(args.npz_root, "train.txt"))
    test_entries = load_split_entries(os.path.join(args.npz_root, "test.txt"))
    os.makedirs(args.output_root, exist_ok=True)

    print(f"Exporting {len(train_entries)} train clips (NiAD 8-D from det)...")
    train_rows = export_split(train_entries, args.npz_root, args.output_root, "train", args.num_frames)
    print(f"Exporting {len(test_entries)} test clips...")
    test_rows = export_split(test_entries, args.npz_root, args.output_root, "test", args.num_frames)

    write_manifest(os.path.join(args.output_root, "train_manifest.csv"), train_rows)
    write_manifest(os.path.join(args.output_root, "test_manifest.csv"), test_rows)
    print("Done.")
    print(f"  {args.output_root}/train_manifest.csv")
    print(f"  {args.output_root}/test_manifest.csv")


if __name__ == "__main__":
    main()
