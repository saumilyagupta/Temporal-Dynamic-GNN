"""
Dataset implementation for loading temporal graph sequences from .pt files.
Each video clip contains 30 PyG Data objects representing frames.
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from typing import List, Tuple, Optional


class TemporalGraphDataset(Dataset):
    """
    Dataset for loading temporal graph sequences from .pt files.
    
    Each sample is a video clip with 30 dynamic graphs (one per frame).
    Graphs have variable node counts - no padding required.
    """
    
    def __init__(
        self,
        manifest_path: str,
        normalize_features: bool = True,
        feature_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ):
        """
        Args:
            manifest_path: Full path to CSV manifest file
            normalize_features: Whether to normalize spatial features
            feature_stats: Optional (mean, std) tuple for normalization
        """
        self.manifest_path = manifest_path
        self.normalize_features = normalize_features
        
        # Load manifest
        self.df = pd.read_csv(manifest_path)
        
        # Convert labels to binary (Normal=0, Anomalous=1)
        self.df['label_binary'] = self.df['label'].map({'Normal': 0, 'Anomalous': 1})
        
        # Calculate or use provided normalization stats
        if normalize_features:
            if feature_stats is not None:
                self.feature_mean, self.feature_std = feature_stats
            else:
                self._calculate_normalization_stats()
    
    def _calculate_normalization_stats(self):
        """Calculate mean and std for feature normalization from a sample of data."""
        print("Calculating normalization statistics...")
        sample_size = min(100, len(self.df))
        sample_indices = np.random.choice(len(self.df), sample_size, replace=False)
        
        all_features = []
        for idx in sample_indices:
            video_folder = self.df.iloc[idx]['video_folder_path']
            # Sample a few frames per video
            for t in [0, 10, 20, 29]:
                graph_path = os.path.join(video_folder, f'graph_{t:03d}.pt')
                if os.path.exists(graph_path):
                    data = torch.load(graph_path, weights_only=False)
                    if data.x is not None and data.x.shape[0] > 0:
                        all_features.append(data.x.numpy())
        
        if all_features:
            all_features = np.concatenate(all_features, axis=0)
            self.feature_mean = np.mean(all_features, axis=0)
            self.feature_std = np.std(all_features, axis=0)
            # Avoid division by zero
            self.feature_std = np.where(self.feature_std < 1e-6, 1.0, self.feature_std)
        else:
            # Default values if no valid data found
            self.feature_mean = np.zeros(8)
            self.feature_std = np.ones(8)
        
        print(f"Normalization stats - Mean: {self.feature_mean}")
        print(f"Normalization stats - Std: {self.feature_std}")
    
    def get_normalization_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return normalization statistics for use in other datasets."""
        return self.feature_mean, self.feature_std
    
    def _normalize_features(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize node features using precomputed statistics."""
        if x.shape[0] == 0:
            return x
        
        mean = torch.tensor(self.feature_mean, dtype=x.dtype, device=x.device)
        std = torch.tensor(self.feature_std, dtype=x.dtype, device=x.device)
        
        return (x - mean) / std
    
    def _load_graph_sequence(self, video_folder: str) -> List[Data]:
        """
        Load all 30 graphs for a video clip.
        
        Returns:
            List of 30 Data objects (one per frame)
        """
        graphs = []
        
        for t in range(30):
            graph_path = os.path.join(video_folder, f'graph_{t:03d}.pt')
            
            if os.path.exists(graph_path):
                data = torch.load(graph_path, weights_only=False)
                
                # Normalize features if enabled
                if self.normalize_features and data.x is not None and data.x.shape[0] > 0:
                    data.x = self._normalize_features(data.x)
                
                graphs.append(data)
            else:
                # Create empty graph if file doesn't exist
                graphs.append(Data(
                    x=torch.zeros((0, 8), dtype=torch.float32),
                    edge_index=torch.zeros((2, 0), dtype=torch.long),
                    edge_weight=torch.zeros(0, dtype=torch.float32),
                    node_ids=torch.zeros(0, dtype=torch.long),
                    t=torch.tensor([t], dtype=torch.float32),
                ))
        
        return graphs
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[List[Data], int]:
        """
        Get a temporal graph sequence.
        
        Returns:
            graphs: List of 30 Data objects (one per frame)
            label: Binary label (0=Normal, 1=Anomalous)
        """
        video_folder = self.df.iloc[idx]['video_folder_path']
        graphs = self._load_graph_sequence(video_folder)
        label = self.df.iloc[idx]['label_binary']
        
        return graphs, label


def collate_temporal_graphs(
    batch: List[Tuple[List[Data], int]]
) -> Tuple[List[List[Data]], torch.Tensor]:
    """
    Custom collate function for temporal graph sequences.
    
    Args:
        batch: List of (graphs, label) tuples
    
    Returns:
        batched_graphs: List of lists of Data objects (one list per sample)
        labels: Tensor of labels
    """
    graphs_list = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    
    return graphs_list, labels


def create_dataloaders(
    data_root: str,
    train_manifest: str = 'train_manifest.csv',
    val_manifest: str = 'val_manifest.csv',
    test_manifest: str = 'test_manifest.csv',
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
    normalize_features: bool = True,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create DataLoaders for train, validation, and test sets.
    
    Args:
        data_root: Root directory containing the dataset
        train_manifest: Name of training manifest CSV
        val_manifest: Name of validation manifest CSV
        test_manifest: Name of test manifest CSV (optional)
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for faster GPU transfer
        normalize_features: Whether to normalize node features
    
    Returns:
        train_loader, val_loader, test_loader (if test_manifest exists)
    """
    train_manifest_path = os.path.join(data_root, train_manifest)
    val_manifest_path = os.path.join(data_root, val_manifest)
    test_manifest_path = os.path.join(data_root, test_manifest)
    
    # Create training dataset first to get normalization stats
    train_dataset = TemporalGraphDataset(
        manifest_path=train_manifest_path,
        normalize_features=normalize_features,
    )
    
    # Use same normalization stats for val/test
    feature_stats = train_dataset.get_normalization_stats() if normalize_features else None
    
    val_dataset = TemporalGraphDataset(
        manifest_path=val_manifest_path,
        normalize_features=normalize_features,
        feature_stats=feature_stats,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_temporal_graphs,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_temporal_graphs,
    )
    
    test_loader = None
    if os.path.exists(test_manifest_path):
        test_dataset = TemporalGraphDataset(
            manifest_path=test_manifest_path,
            normalize_features=normalize_features,
            feature_stats=feature_stats,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_temporal_graphs,
        )
    
    return train_loader, val_loader, test_loader



