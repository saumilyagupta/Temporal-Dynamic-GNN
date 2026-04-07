"""
Post-training evaluation for CarCrash — NiAD-style 8-D graphs (det-only).
"""

from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.multiprocessing
import yaml

torch.multiprocessing.set_sharing_strategy("file_system")

from dataset_carcrash import (
    collate_carcrash_with_ids,
    load_carcrash_manifest_entries,
    load_split_entries,
)
from dataset_carcrash_niad import CarCrashNiadDataset, CarCrashNiadPrecomputedDataset
from torch.utils.data import DataLoader
from utils import (
    compute_metrics,
    format_metrics,
    get_device,
    plot_confusion_matrix,
    plot_roc_curve,
    set_seed,
)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_crash_timing(annotation_file: str) -> Dict[str, str]:
    timing: Dict[str, str] = {}
    with open(annotation_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vidname = line.split(",", 1)[0].strip().zfill(6)
            try:
                bracket_end = line.index("]")
            except ValueError:
                continue
            after = line[bracket_end + 2 :]
            fields = after.split(",")
            if len(fields) < 3:
                continue
            t = fields[2].strip()
            if t in ("Day", "Night"):
                timing[vidname] = t
    return timing


def _subset_mask(
    labels: np.ndarray,
    video_ids: List[str],
    timing_map: Dict[str, str],
    subset: str,
) -> np.ndarray:
    n = len(labels)
    mask = np.ones(n, dtype=bool)
    if subset == "combined":
        return mask

    crash_day = np.zeros(n, dtype=bool)
    crash_night = np.zeros(n, dtype=bool)
    for i in range(n):
        if labels[i] != 1:
            continue
        vid = video_ids[i]
        t = timing_map.get(vid)
        if t == "Day":
            crash_day[i] = True
        elif t == "Night":
            crash_night[i] = True

    normal = labels == 0
    if subset == "day":
        return normal | crash_day
    if subset == "night":
        return normal | crash_night
    raise ValueError(subset)


def run_inference_with_ids(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    model.eval()
    model.to(device)
    all_preds: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_vids: List[str] = []

    with torch.no_grad():
        for batch in dataloader:
            graphs_batch, labels, vids = batch
            graphs_batch = graphs_batch.to(device)
            logits = model(graphs_batch)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_vids.extend(vids)

    preds = np.concatenate(all_preds)
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    return preds, probs, labels, all_vids


def _write_metrics_txt(path: str, name: str, metrics: Dict[str, float], extra: str = "") -> None:
    with open(path, "w") as f:
        f.write(f"{name}\n")
        f.write("=" * 40 + "\n")
        if extra:
            f.write(extra + "\n\n")
        f.write(format_metrics(metrics))


def evaluate_and_save(
    checkpoint_path: str,
    config: dict,
    exp_dir: str,
    seed: int = 42,
) -> None:
    from loss import calculate_alpha_from_dataset
    from lightning_module import TemporalGNNLightning
    from model import TemporalGNN

    set_seed(seed)
    results_dir = os.path.join(exp_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    data_cfg = config["data"]
    train_cfg = config["training"]
    root = data_cfg["root_dir"]
    ann_path = data_cfg.get("crash_annotation_file", "")
    timing_map = load_crash_timing(ann_path) if ann_path and os.path.isfile(ann_path) else {}

    gpu_id = train_cfg.get("gpu_id", 0)
    if gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    device = get_device()

    from sklearn.model_selection import train_test_split

    num_frames = int(data_cfg.get("num_frames", 50))
    val_split = float(data_cfg.get("val_split", 0.2))
    norm = data_cfg.get("normalize_features", True)

    if data_cfg.get("source", "npz") == "precomputed":
        graph_root = data_cfg["graph_export_root"]
        train_m = os.path.join(graph_root, data_cfg.get("train_manifest", "train_manifest.csv"))
        test_m = os.path.join(graph_root, data_cfg.get("test_manifest", "test_manifest.csv"))
        train_entries = load_carcrash_manifest_entries(train_m)
        test_entries = load_carcrash_manifest_entries(test_m)
        labels_tr = [e[1] for e in train_entries]
        tr_idx, _ = train_test_split(
            np.arange(len(train_entries)),
            test_size=val_split,
            stratify=labels_tr,
            random_state=seed,
        )
        train_only = [train_entries[i] for i in tr_idx]

        train_ds = CarCrashNiadPrecomputedDataset(
            train_only,
            num_frames=num_frames,
            normalize_features=norm,
        )
        stats = train_ds.get_normalization_stats() if norm else None
        test_ds = CarCrashNiadPrecomputedDataset(
            test_entries,
            num_frames=num_frames,
            normalize_features=norm,
            feature_stats=stats,
        )
    else:
        test_entries = load_split_entries(os.path.join(root, data_cfg["test_txt"]))
        train_entries = load_split_entries(os.path.join(root, data_cfg["train_txt"]))
        labels_tr = [e[1] for e in train_entries]
        tr_idx, _ = train_test_split(
            np.arange(len(train_entries)),
            test_size=val_split,
            stratify=labels_tr,
            random_state=seed,
        )
        train_only = [train_entries[i] for i in tr_idx]

        train_ds = CarCrashNiadDataset(
            train_only,
            root_dir=root,
            normalize_features=norm,
            num_frames=num_frames,
        )
        stats = train_ds.get_normalization_stats() if norm else None
        test_ds = CarCrashNiadDataset(
            test_entries,
            root_dir=root,
            normalize_features=norm,
            feature_stats=stats,
            num_frames=num_frames,
        )

    test_loader = DataLoader(
        test_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=data_cfg.get("pin_memory", True),
        collate_fn=collate_carcrash_with_ids,
    )

    model = TemporalGNN(
        input_dim=config["model"]["input_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        temporal_dim=config["model"]["temporal_dim"],
        num_gcn_layers=config["model"]["num_gcn_layers"],
        num_gru_layers=config["model"]["num_gru_layers"],
        dropout=config["model"]["dropout"],
        num_classes=config["model"]["num_classes"],
    )

    alpha = calculate_alpha_from_dataset(train_ds)
    lit = TemporalGNNLightning.load_from_checkpoint(
        checkpoint_path,
        model=model,
        focal_loss_alpha=alpha,
        focal_loss_gamma=float(config["loss"]["gamma"]),
        weights_only=False,
    )
    net = lit.model

    preds, probs, y_true, video_ids = run_inference_with_ids(net, test_loader, device)

    pred_path = os.path.join(results_dir, "predictions.csv")
    with open(pred_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["video_id", "true_label", "pred_label", "prob_crash", "timing_annotation"]
        )
        for i in range(len(y_true)):
            vid = video_ids[i]
            timing = ""
            if y_true[i] == 1:
                timing = timing_map.get(vid, "")
            else:
                timing = "N/A_normal"
            w.writerow(
                [
                    vid,
                    int(y_true[i]),
                    int(preds[i]),
                    float(probs[i, 1]),
                    timing,
                ]
            )

    summaries: List[str] = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for subset in ("combined", "day", "night"):
        mask = _subset_mask(y_true, video_ids, timing_map, subset)
        if mask.sum() == 0:
            continue
        yt = y_true[mask]
        yp = preds[mask]
        pr = probs[mask, 1]
        m = compute_metrics(yt, yp, pr)
        name = {
            "combined": "Combined (all test)",
            "day": "Day subset (normal + Day crash)",
            "night": "Night subset (normal + Night crash)",
        }[subset]
        extra = f"n_samples: {mask.sum()}\n"
        if subset != "combined":
            extra += (
                f"(includes all normal videos; crash videos filtered by {subset} timing)\n"
            )
        path = os.path.join(results_dir, f"{subset}_metrics.txt")
        _write_metrics_txt(path, name, m, extra)
        summaries.append(f"\n{name}\n{format_metrics(m)}\n{extra}")

        plot_confusion_matrix(
            yt,
            yp,
            save_path=os.path.join(results_dir, f"confusion_matrix_{subset}.png"),
        )
        if len(np.unique(yt)) > 1:
            plot_roc_curve(
                yt,
                pr,
                save_path=os.path.join(results_dir, f"roc_curve_{subset}.png"),
            )

    summary_path = os.path.join(results_dir, "all_results_summary.txt")
    with open(summary_path, "w") as f:
        f.write("CarCrash TemporalGNN (NiAD-style det) — Evaluation summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("=" * 60 + "\n")
        for block in summaries:
            f.write(block)
            f.write("\n")

    print(f"\n[evaluate_carcrash_niad] Results saved under: {results_dir}")


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--config", type=str, default="config_carcrash_niad.yaml")
    p.add_argument("--exp_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    cfg = load_config(args.config)
    evaluate_and_save(args.checkpoint, cfg, args.exp_dir, seed=args.seed)


if __name__ == "__main__":
    main()
