#!/usr/bin/env python3
"""
Export CarCrash NPZ (VGG16 + bbox det) to on-disk PyG graphs for training.

Graphs match node_feature_based_Temporal_DGNN: fully connected edges weighted by
bbox centroid distances; node features = per-vehicle VGG16 embeddings.

Output layout (same spirit as main.py / NiAD pipeline):
  {output_root}/train/{Normal|Anomalous}/{video_id}/graph_{t:03d}.pt
  {output_root}/test/{Normal|Anomalous}/{video_id}/graph_{t:03d}.pt
  {output_root}/train_manifest.csv
  {output_root}/test_manifest.csv

Training still uses 20%% of train_manifest stratified as validation (see
create_carcrash_precomputed_dataloaders + config val_split).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

# Repo layout: data_pipeline_node_feature_based/ -> ../node_feature_based_Temporal_DGNN/
_NF_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "node_feature_based_Temporal_DGNN"))
if _NF_DIR not in sys.path:
    sys.path.insert(0, _NF_DIR)

from carcrash_graph_utils import (  # noqa: E402
    load_npz_graph_sequence,
    save_graph_sequence_to_folder,
)
from dataset_carcrash import load_split_entries  # noqa: E402


def _class_folder(label: int) -> str:
    return "Normal" if label == 0 else "Anomalous"


def export_split(
    entries: list,
    npz_root: str,
    out_root: str,
    split_name: str,
    num_frames: int,
) -> list[dict]:
    """Write graphs + return rows for manifest."""
    rows: list[dict] = []
    for rel, lab, vid in entries:
        npz_path = os.path.join(npz_root, rel)
        clip_dir = os.path.join(out_root, split_name, _class_folder(lab), vid)
        graphs = load_npz_graph_sequence(npz_path, num_frames=num_frames)
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
    p = argparse.ArgumentParser(description="CarCrash NPZ → per-frame graph .pt export")
    p.add_argument(
        "--npz_root",
        type=str,
        default="/workspace/GNN/CarCrash/vgg16_features",
        help="Directory containing train.txt, test.txt, and NPZ files",
    )
    p.add_argument(
        "--output_root",
        type=str,
        default="/workspace/GNN/CarCrash/carcrash_graphs_export",
        help="Where to write train/, test/, manifests",
    )
    p.add_argument("--num_frames", type=int, default=50)
    args = p.parse_args()

    train_txt = os.path.join(args.npz_root, "train.txt")
    test_txt = os.path.join(args.npz_root, "test.txt")

    train_entries = load_split_entries(train_txt)
    test_entries = load_split_entries(test_txt)

    os.makedirs(args.output_root, exist_ok=True)

    print(f"Exporting {len(train_entries)} train clips...")
    train_rows = export_split(
        train_entries, args.npz_root, args.output_root, "train", args.num_frames
    )
    print(f"Exporting {len(test_entries)} test clips...")
    test_rows = export_split(
        test_entries, args.npz_root, args.output_root, "test", args.num_frames
    )

    write_manifest(os.path.join(args.output_root, "train_manifest.csv"), train_rows)
    write_manifest(os.path.join(args.output_root, "test_manifest.csv"), test_rows)

    print("Done.")
    print(f"  {args.output_root}/train_manifest.csv")
    print(f"  {args.output_root}/test_manifest.csv")


if __name__ == "__main__":
    main()
