"""
CarCrash with NiAD-style node features (8-D from bbox ``det`` only, no VGG, no YOLO).

Graph topology matches ``vehicle_processor.create_frame_graph``: fully connected,
edge weights = centroid distances, ``node_ids`` = detection slot indices.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data

from dataset_carcrash import (
    collate_carcrash,
    collate_carcrash_with_ids,
    load_carcrash_manifest_entries,
    load_split_entries,
)
from niad_graph_from_det import (
    build_niad_graph_from_det_frame,
    load_npz_niad_graph_sequence,
    save_graph_sequence_to_folder,
)


class CarCrashNiadDataset(Dataset):
    """NiAD-style graphs built from NPZ ``det`` at read time."""

    def __init__(
        self,
        entries: List[Tuple[str, int, str]],
        root_dir: str,
        num_frames: int = 50,
        normalize_features: bool = True,
        feature_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ):
        self.entries = entries
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.normalize_features = normalize_features
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_std: Optional[np.ndarray] = None

        labels_str = ["Normal" if lab == 0 else "Anomalous" for _, lab, _ in entries]
        self.df = pd.DataFrame({"label": labels_str})

        if normalize_features:
            if feature_stats is not None:
                self.feature_mean, self.feature_std = feature_stats
            else:
                self._compute_norm_stats()

    def _compute_norm_stats(self, max_samples: int = 100) -> None:
        rng = np.random.default_rng(42)
        idxs = np.arange(len(self.entries))
        if len(idxs) > max_samples:
            idxs = rng.choice(idxs, size=max_samples, replace=False)

        feats: List[np.ndarray] = []
        for idx in idxs:
            rel, _, _ = self.entries[int(idx)]
            path = os.path.join(self.root_dir, rel)
            bundle = np.load(path, allow_pickle=True)
            det = bundle["det"]
            for t in (0, self.num_frames // 2, self.num_frames - 1):
                g = build_niad_graph_from_det_frame(det[t], t, self.num_frames)
                if g.x is not None and g.x.shape[0] > 0:
                    feats.append(g.x.numpy())
        if feats:
            all_f = np.concatenate(feats, axis=0)
            self.feature_mean = np.mean(all_f, axis=0)
            self.feature_std = np.std(all_f, axis=0)
            self.feature_std = np.where(self.feature_std < 1e-6, 1.0, self.feature_std)
        else:
            self.feature_mean = np.zeros(8, dtype=np.float32)
            self.feature_std = np.ones(8, dtype=np.float32)

    def get_normalization_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        assert self.feature_mean is not None and self.feature_std is not None
        return self.feature_mean, self.feature_std

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        rel, lab, vid = self.entries[idx]
        path = os.path.join(self.root_dir, rel)
        stats = None
        if self.normalize_features and self.feature_mean is not None:
            stats = (self.feature_mean, self.feature_std)
        graphs = load_npz_niad_graph_sequence(
            path,
            num_frames=self.num_frames,
            feature_mean=stats[0] if stats else None,
            feature_std=stats[1] if stats else None,
        )
        return graphs, lab, vid


class CarCrashNiadPrecomputedDataset(Dataset):
    """NiAD-style graphs loaded from exported ``graph_*.pt`` (8-D)."""

    def __init__(
        self,
        entries: List[Tuple[str, int, str]],
        num_frames: int = 50,
        normalize_features: bool = True,
        feature_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ):
        self.entries = entries
        self.num_frames = num_frames
        self.normalize_features = normalize_features
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_std: Optional[np.ndarray] = None

        labels_str = ["Normal" if lab == 0 else "Anomalous" for _, lab, _ in entries]
        self.df = pd.DataFrame({"label": labels_str})

        if normalize_features:
            if feature_stats is not None:
                self.feature_mean, self.feature_std = feature_stats
            else:
                self._compute_norm_stats()

    def _compute_norm_stats(self, max_samples: int = 100) -> None:
        rng = np.random.default_rng(42)
        idxs = np.arange(len(self.entries))
        if len(idxs) > max_samples:
            idxs = rng.choice(idxs, size=max_samples, replace=False)

        feats: List[np.ndarray] = []
        for idx in idxs:
            folder, _, _ = self.entries[int(idx)]
            for t in (0, self.num_frames // 2, self.num_frames - 1):
                path = os.path.join(folder, f"graph_{t:03d}.pt")
                data = torch.load(path, weights_only=False)
                if data.x is not None and data.x.shape[0] > 0:
                    feats.append(data.x.numpy())
        if feats:
            all_f = np.concatenate(feats, axis=0)
            self.feature_mean = np.mean(all_f, axis=0)
            self.feature_std = np.std(all_f, axis=0)
            self.feature_std = np.where(self.feature_std < 1e-6, 1.0, self.feature_std)
        else:
            self.feature_mean = np.zeros(8, dtype=np.float32)
            self.feature_std = np.ones(8, dtype=np.float32)

    def get_normalization_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        assert self.feature_mean is not None and self.feature_std is not None
        return self.feature_mean, self.feature_std

    def __len__(self) -> int:
        return len(self.entries)

    def _normalize_x(self, data: Data) -> Data:
        if (
            not self.normalize_features
            or self.feature_mean is None
            or data.x is None
            or data.x.shape[0] == 0
        ):
            return data
        mean = torch.tensor(self.feature_mean, dtype=data.x.dtype, device=data.x.device)
        std = torch.tensor(self.feature_std, dtype=data.x.dtype, device=data.x.device)
        data.x = (data.x - mean) / std
        return data

    def __getitem__(self, idx: int):
        folder, lab, vid = self.entries[idx]
        graphs: List[Data] = []
        for t in range(self.num_frames):
            path = os.path.join(folder, f"graph_{t:03d}.pt")
            data = torch.load(path, weights_only=False)
            data = self._normalize_x(data)
            graphs.append(data)
        return graphs, lab, vid


def create_carcrash_niad_dataloaders(
    data_root: str,
    train_txt: str = "train.txt",
    test_txt: str = "test.txt",
    batch_size: int = 4,
    val_split: float = 0.2,
    seed: int = 42,
    num_workers: int = 4,
    pin_memory: bool = True,
    normalize_features: bool = True,
    num_frames: int = 50,
) -> Tuple[DataLoader, DataLoader, DataLoader, CarCrashNiadDataset]:
    train_path = os.path.join(data_root, train_txt)
    test_path = os.path.join(data_root, test_txt)

    train_entries = load_split_entries(train_path)
    test_entries = load_split_entries(test_path)

    labels = [e[1] for e in train_entries]
    tr_idx, va_idx = train_test_split(
        np.arange(len(train_entries)),
        test_size=val_split,
        stratify=labels,
        random_state=seed,
    )
    train_e = [train_entries[i] for i in tr_idx]
    val_e = [train_entries[i] for i in va_idx]

    train_dataset = CarCrashNiadDataset(
        train_e,
        root_dir=data_root,
        normalize_features=normalize_features,
        num_frames=num_frames,
    )
    stats = train_dataset.get_normalization_stats() if normalize_features else None

    val_dataset = CarCrashNiadDataset(
        val_e,
        root_dir=data_root,
        normalize_features=normalize_features,
        feature_stats=stats,
        num_frames=num_frames,
    )
    test_dataset = CarCrashNiadDataset(
        test_entries,
        root_dir=data_root,
        normalize_features=normalize_features,
        feature_stats=stats,
        num_frames=num_frames,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_carcrash,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_carcrash,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_carcrash,
    )

    return train_loader, val_loader, test_loader, train_dataset


def create_carcrash_niad_precomputed_dataloaders(
    graph_export_root: str,
    train_manifest: str = "train_manifest.csv",
    test_manifest: str = "test_manifest.csv",
    batch_size: int = 4,
    val_split: float = 0.2,
    seed: int = 42,
    num_workers: int = 4,
    pin_memory: bool = True,
    normalize_features: bool = True,
    num_frames: int = 50,
) -> Tuple[DataLoader, DataLoader, DataLoader, CarCrashNiadPrecomputedDataset]:
    train_m_path = os.path.join(graph_export_root, train_manifest)
    test_m_path = os.path.join(graph_export_root, test_manifest)

    train_entries = load_carcrash_manifest_entries(train_m_path)
    test_entries = load_carcrash_manifest_entries(test_m_path)

    labels = [e[1] for e in train_entries]
    tr_idx, va_idx = train_test_split(
        np.arange(len(train_entries)),
        test_size=val_split,
        stratify=labels,
        random_state=seed,
    )
    train_e = [train_entries[i] for i in tr_idx]
    val_e = [train_entries[i] for i in va_idx]

    train_dataset = CarCrashNiadPrecomputedDataset(
        train_e,
        num_frames=num_frames,
        normalize_features=normalize_features,
    )
    stats = train_dataset.get_normalization_stats() if normalize_features else None

    val_dataset = CarCrashNiadPrecomputedDataset(
        val_e,
        num_frames=num_frames,
        normalize_features=normalize_features,
        feature_stats=stats,
    )
    test_dataset = CarCrashNiadPrecomputedDataset(
        test_entries,
        num_frames=num_frames,
        normalize_features=normalize_features,
        feature_stats=stats,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_carcrash,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_carcrash,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_carcrash,
    )

    return train_loader, val_loader, test_loader, train_dataset


__all__ = [
    "CarCrashNiadDataset",
    "CarCrashNiadPrecomputedDataset",
    "create_carcrash_niad_dataloaders",
    "create_carcrash_niad_precomputed_dataloaders",
]
