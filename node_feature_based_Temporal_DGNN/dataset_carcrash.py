"""
CarCrash dataset: load VGG16 bbox embeddings from .npz or precomputed .pt graphs;
build per-frame PyG graphs (bbox centroids -> edge weights).
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data, Batch

from carcrash_graph_utils import load_npz_graph_sequence


def _parse_txt_line(line: str) -> Tuple[str, int, str]:
    """
    Parse a line from train.txt / test.txt.

    Returns:
        relative_path, binary_label (0=normal, 1=crash), video_id (6-digit string)
    """
    line = line.strip()
    if not line:
        raise ValueError("empty line")
    parts = line.split()
    rel_path = parts[0]
    label = int(parts[1])
    base = os.path.basename(rel_path).replace(".npz", "")
    video_id = base.zfill(6)
    return rel_path, label, video_id


def load_split_entries(txt_path: str) -> List[Tuple[str, int, str]]:
    """Load all (relative_path, label, video_id) from a split file."""
    entries = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(_parse_txt_line(line))
    return entries


class CarCrashDataset(Dataset):
    """
    CarCrash VGG16 feature graphs loaded from .npz on the fly.

    Each item: (list of T Data graphs, binary label, video_id str).
    """

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
        """Sample up to max_samples videos and aggregate feature statistics."""
        rng = np.random.default_rng(42)
        idxs = np.arange(len(self.entries))
        if len(idxs) > max_samples:
            idxs = rng.choice(idxs, size=max_samples, replace=False)

        feats: List[np.ndarray] = []
        for idx in idxs:
            rel, _, _ = self.entries[int(idx)]
            path = os.path.join(self.root_dir, rel)
            bundle = np.load(path, allow_pickle=True)
            d = bundle["data"][:, 1:, :]  # (T, 19, 4096)
            for t in (0, self.num_frames // 2, self.num_frames - 1):
                feats.append(d[t].reshape(-1, 4096))
        if feats:
            all_f = np.concatenate(feats, axis=0)
            self.feature_mean = np.mean(all_f, axis=0)
            self.feature_std = np.std(all_f, axis=0)
            self.feature_std = np.where(self.feature_std < 1e-6, 1.0, self.feature_std)
        else:
            self.feature_mean = np.zeros(4096, dtype=np.float32)
            self.feature_std = np.ones(4096, dtype=np.float32)

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
        graphs = load_npz_graph_sequence(
            path,
            num_frames=self.num_frames,
            feature_mean=stats[0] if stats else None,
            feature_std=stats[1] if stats else None,
        )
        return graphs, lab, vid


class CarCrashPrecomputedDataset(Dataset):
    """
    CarCrash graphs loaded from disk (exported by export_carcrash_npz_to_graphs.py).
    Same graph definition as CarCrashDataset; normalization matches NPZ path.
    """

    def __init__(
        self,
        entries: List[Tuple[str, int, str]],
        num_frames: int = 50,
        normalize_features: bool = True,
        feature_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ):
        """
        Args:
            entries: list of (clip_folder_path, label, video_id)
        """
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
        feature_dim = 13  # Default fallback for bbox pipeline
        for idx in idxs:
            folder, _, _ = self.entries[int(idx)]
            seq_path = os.path.join(folder, "sequence.pt")
            if os.path.isfile(seq_path):
                graphs = torch.load(seq_path, weights_only=False)
                for t in (0, self.num_frames // 2, self.num_frames - 1):
                    if t < len(graphs):
                        data = graphs[t]
                        if data.x is not None and data.x.shape[0] > 0:
                            feats.append(data.x.numpy())
                            feature_dim = data.x.shape[1]
            else:
                # Fallback to individual files if sequence.pt doesn't exist
                for t in (0, self.num_frames // 2, self.num_frames - 1):
                    path = os.path.join(folder, f"graph_{t:03d}.pt")
                    if os.path.isfile(path):
                        data = torch.load(path, weights_only=False)
                        if data.x is not None and data.x.shape[0] > 0:
                            feats.append(data.x.numpy())
                            feature_dim = data.x.shape[1]
                            
        if feats:
            all_f = np.concatenate(feats, axis=0)
            self.feature_mean = np.mean(all_f, axis=0)
            self.feature_std = np.std(all_f, axis=0)
            self.feature_std = np.where(self.feature_std < 1e-6, 1.0, self.feature_std)
        else:
            self.feature_mean = np.zeros(feature_dim, dtype=np.float32)
            self.feature_std = np.ones(feature_dim, dtype=np.float32)

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
        
        seq_path = os.path.join(folder, "sequence.pt")
        if os.path.isfile(seq_path):
            graphs = torch.load(seq_path, weights_only=False)
            # Ensure we only take num_frames if sequence is longer
            graphs = graphs[:self.num_frames]
            # Pad if shorter
            while len(graphs) < self.num_frames:
                # Add empty graph
                empty_graph = Data(
                    x=torch.zeros((0, graphs[0].x.shape[1] if graphs and graphs[0].x is not None else 13), dtype=torch.float32),
                    edge_index=torch.zeros((2, 0), dtype=torch.long),
                    edge_weight=torch.zeros((0,), dtype=torch.float32),
                    t=torch.tensor([len(graphs)], dtype=torch.long)
                )
                graphs.append(empty_graph)
                
            for i in range(len(graphs)):
                graphs[i] = self._normalize_x(graphs[i])
            return graphs, lab, vid
            
        # Fallback to individual files
        graphs: List[Data] = []
        for t in range(self.num_frames):
            path = os.path.join(folder, f"graph_{t:03d}.pt")
            if os.path.isfile(path):
                data = torch.load(path, weights_only=False)
            else:
                data = Data(
                    x=torch.zeros((0, 13), dtype=torch.float32),
                    edge_index=torch.zeros((2, 0), dtype=torch.long),
                    edge_weight=torch.zeros((0,), dtype=torch.float32),
                    t=torch.tensor([t], dtype=torch.long)
                )
            data = self._normalize_x(data)
            graphs.append(data)
        return graphs, lab, vid


def collate_carcrash(
    batch: List[Tuple[List[Data], int, str]],
) -> Tuple[Batch, torch.Tensor]:
    """Collate for PyTorch Lightning. Flattens all graphs into a single PyG Batch."""
    # Flatten the list of lists into a single list of graphs
    flat_graphs = [g for item in batch for g in item[0]]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return Batch.from_data_list(flat_graphs), labels


def collate_carcrash_with_ids(
    batch: List[Tuple[List[Data], int, str]],
) -> Tuple[Batch, torch.Tensor, List[str]]:
    """Collate including video ids for evaluation."""
    flat_graphs = [g for item in batch for g in item[0]]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    vids = [item[2] for item in batch]
    return Batch.from_data_list(flat_graphs), labels, vids


def create_carcrash_dataloaders(
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
) -> Tuple[DataLoader, DataLoader, DataLoader, CarCrashDataset]:
    """
    Stratified (1 - val_split) / val_split split of train.txt -> train / val; test.txt -> test.
    """
    train_path = os.path.join(data_root, train_txt)
    test_path = os.path.join(data_root, test_txt)

    train_entries = load_split_entries(train_path)
    test_entries = load_split_entries(test_path)

    labels = [e[1] for e in train_entries]
    if val_split > 0.0 and val_split < 1.0 and len(train_entries) > 1:
        tr_idx, va_idx = train_test_split(
            np.arange(len(train_entries)),
            test_size=val_split,
            stratify=labels,
            random_state=seed,
        )
        train_e = [train_entries[i] for i in tr_idx]
        val_e = [train_entries[i] for i in va_idx]
    else:
        train_e = train_entries
        val_e = train_entries

    train_dataset = CarCrashDataset(
        train_e,
        root_dir=data_root,
        normalize_features=normalize_features,
        num_frames=num_frames,
    )
    stats = train_dataset.get_normalization_stats() if normalize_features else None

    val_dataset = CarCrashDataset(
        val_e,
        root_dir=data_root,
        normalize_features=normalize_features,
        feature_stats=stats,
        num_frames=num_frames,
    )
    test_dataset = CarCrashDataset(
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


def load_carcrash_manifest_entries(
    manifest_csv: str,
) -> List[Tuple[str, int, str]]:
    """Build (folder, label, video_id) from a NiAD-style manifest CSV."""
    df = pd.read_csv(manifest_csv)
    entries: List[Tuple[str, int, str]] = []
    for _, row in df.iterrows():
        folder = str(row["video_folder_path"])
        lab_str = str(row["label"]).strip()
        lab = 0 if lab_str == "Normal" else 1
        vid = str(row.get("source_video", os.path.basename(folder))).strip()
        vid = os.path.splitext(vid)[0].zfill(6)
        entries.append((folder, lab, vid))
    return entries


def create_carcrash_precomputed_dataloaders(
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
) -> Tuple[DataLoader, DataLoader, DataLoader, CarCrashPrecomputedDataset]:
    """
    Same 80/20 stratified split of training manifest as create_carcrash_dataloaders.
    Expects manifests produced by data_pipeline/export_carcrash_npz_to_graphs.py.
    """
    train_m_path = os.path.join(graph_export_root, train_manifest)
    test_m_path = os.path.join(graph_export_root, test_manifest)

    train_entries = load_carcrash_manifest_entries(train_m_path)
    test_entries = load_carcrash_manifest_entries(test_m_path)

    labels = [e[1] for e in train_entries]
    if val_split > 0.0 and val_split < 1.0 and len(train_entries) > 1:
        tr_idx, va_idx = train_test_split(
            np.arange(len(train_entries)),
            test_size=val_split,
            stratify=labels,
            random_state=seed,
        )
        train_e = [train_entries[i] for i in tr_idx]
        val_e = [train_entries[i] for i in va_idx]
    else:
        train_e = train_entries
        val_e = train_entries

    train_dataset = CarCrashPrecomputedDataset(
        train_e,
        num_frames=num_frames,
        normalize_features=normalize_features,
    )
    stats = train_dataset.get_normalization_stats() if normalize_features else None

    val_dataset = CarCrashPrecomputedDataset(
        val_e,
        num_frames=num_frames,
        normalize_features=normalize_features,
        feature_stats=stats,
    )
    test_dataset = CarCrashPrecomputedDataset(
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
