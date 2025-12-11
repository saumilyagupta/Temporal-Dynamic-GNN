"""
Dataset implementation for loading temporal graph data from HDF5 files.
Handles variable-sized graphs with masking for dynamic graph neural networks.
"""

import os
import h5py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from typing import List, Tuple, Optional
import re


class TemporalGraphDataset(Dataset):
    """
    Dataset for loading temporal graph sequences from HDF5 files.
    
    Each sample is a video clip with temporal graphs across multiple frames.
    """
    
    def __init__(
        self,
        manifest_path: str,
        root_dir: str,
        normalize_features: bool = True,
        edge_threshold: Optional[float] = None,
    ):
        """
        Args:
            manifest_path: Path to CSV manifest file
            root_dir: Root directory containing the processed graphs
            normalize_features: Whether to normalize spatial features
            edge_threshold: Optional threshold for creating sparse edges from distances
        """
        self.root_dir = root_dir
        self.normalize_features = normalize_features
        self.edge_threshold = edge_threshold
        
        # Load manifest
        manifest_full_path = os.path.join(root_dir, manifest_path)
        self.df = pd.read_csv(manifest_full_path)
        
        # Convert labels to binary (Normal=0, Anomalous=1)
        self.df['label_binary'] = self.df['label'].map({'Normal': 0, 'Anomalous': 1})
        
        # Calculate normalization stats if needed
        if normalize_features:
            self._calculate_normalization_stats()
    
    def _calculate_normalization_stats(self):
        """Calculate mean and std for feature normalization from a sample of data."""
        print("Calculating normalization statistics...")
        sample_size = min(100, len(self.df))
        sample_indices = np.random.choice(len(self.df), sample_size, replace=False)
        
        all_features = []
        for idx in sample_indices:
            h5_path = self._get_h5_path(idx)
            if os.path.exists(h5_path):
                with h5py.File(h5_path, 'r') as f:
                    # Get all frame keys
                    frame_keys = sorted([k for k in f['graphs'].keys() if k.startswith('t_')])
                    for frame_key in frame_keys[:5]:  # Sample first 5 frames
                        node_features = f[f'graphs/{frame_key}/node_features'][:]
                        node_mask = f[f'graphs/{frame_key}/node_mask'][:]
                        # Only use valid nodes
                        valid_mask = node_mask.astype(bool)
                        if valid_mask.any():
                            valid_features = node_features[valid_mask]
                            # Filter out missing values (-1)
                            valid_features = valid_features[valid_features[:, 0] != -1]
                            if len(valid_features) > 0:
                                all_features.append(valid_features)
        
        if all_features:
            all_features = np.concatenate(all_features, axis=0)
            # Calculate stats for each feature dimension
            self.feature_mean = np.nanmean(all_features, axis=0)
            self.feature_std = np.nanstd(all_features, axis=0)
            # Avoid division by zero
            self.feature_std = np.where(self.feature_std < 1e-6, 1.0, self.feature_std)
        else:
            # Default values if no valid data found
            self.feature_mean = np.zeros(8)
            self.feature_std = np.ones(8)
        
        print(f"Normalization stats - Mean: {self.feature_mean}, Std: {self.feature_std}")
    
    def _get_h5_path(self, idx: int) -> str:
        """Get the full path to the HDF5 file for a given index."""
        video_folder = self.df.iloc[idx]['video_folder_path']
        # Handle both absolute and relative paths
        if os.path.isabs(video_folder):
            h5_path = os.path.join(video_folder, 'temporal_graphs.h5')
        else:
            h5_path = os.path.join(self.root_dir, video_folder, 'temporal_graphs.h5')
        return h5_path
    
    def _load_temporal_graphs(self, h5_path: str) -> Tuple[List[Data], torch.Tensor]:
        """
        Load all temporal graphs from an HDF5 file.
        
        Returns:
            List of Data objects (one per frame) and node mask tensor
        """
        graphs = []
        node_masks = []
        
        # Open file and immediately load all data, then close
        f = h5py.File(h5_path, 'r')
        try:
            # Get all frame keys and sort them
            frame_keys = sorted([k for k in f['graphs'].keys() if k.startswith('t_')],
                               key=lambda x: int(re.search(r'\d+', x).group()))
            
            if not frame_keys:
                raise ValueError(f"No temporal graphs found in {h5_path}")
            
            # Get the maximum number of nodes from the first frame
            first_frame_key = frame_keys[0]
            max_nodes = f[f'graphs/{first_frame_key}/node_features'].shape[0]
            
            # Load all data into memory first
            all_node_features = []
            all_adjacency_matrices = []
            all_node_masks = []
            all_edge_masks = []
            
            for frame_key in frame_keys:
                # Load frame data into memory (copy arrays)
                node_features_array = np.array(f[f'graphs/{frame_key}/node_features'][:], dtype=np.float32)
                adjacency_array = np.array(f[f'graphs/{frame_key}/adjacency_matrix'][:], dtype=np.float32)
                
                # Replace -inf values with -1e-40
                node_features_array = np.where(np.isinf(node_features_array) & (node_features_array < 0), -1e-40, node_features_array)
                adjacency_array = np.where(np.isinf(adjacency_array) & (adjacency_array < 0), -1e-40, adjacency_array)
                
                all_node_features.append(node_features_array)
                all_adjacency_matrices.append(adjacency_array)
                all_node_masks.append(
                    np.array(f[f'graphs/{frame_key}/node_mask'][:], dtype=np.float32)
                )
                all_edge_masks.append(
                    np.array(f[f'graphs/{frame_key}/edge_mask'][:], dtype=np.float32)
                )
        finally:
            # Always close the file
            f.close()
        
        # Process data after file is closed
        for i, frame_key in enumerate(frame_keys):
            # Convert to tensors
            node_features = torch.tensor(all_node_features[i], dtype=torch.float32)
            adjacency_matrix = torch.tensor(all_adjacency_matrices[i], dtype=torch.float32)
            node_mask = torch.tensor(all_node_masks[i], dtype=torch.float32)
            edge_mask = torch.tensor(all_edge_masks[i], dtype=torch.float32)
            
            # Normalize features
            if self.normalize_features:
                node_features = self._normalize_features(node_features, node_mask)
            
            # Create edge indices from adjacency matrix
            edge_index, edge_attr = self._adjacency_to_edges(
                adjacency_matrix, edge_mask, node_mask
            )
            
            # Create PyG Data object
            data = Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_attr,
                node_mask=node_mask,
                edge_mask=edge_mask,
                num_nodes=max_nodes,
            )
            
            graphs.append(data)
            node_masks.append(node_mask)
        
        # Stack node masks for easy access
        node_masks_tensor = torch.stack(node_masks, dim=0)  # [T, N]
        
        return graphs, node_masks_tensor
    
    def _normalize_features(self, features: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        """Normalize node features using precomputed statistics."""
        # Only normalize valid nodes
        normalized = features.clone()
        valid_mask = node_mask.bool()
        
        # Normalize each feature dimension
        for i in range(features.shape[1]):
            # Skip normalization for missing values (-1)
            valid_idx = valid_mask & (features[:, i] != -1)
            if valid_idx.any():
                normalized[valid_idx, i] = (
                    (features[valid_idx, i] - self.feature_mean[i]) / self.feature_std[i]
                )
        
        return normalized
    
    def _adjacency_to_edges(
        self,
        adjacency: torch.Tensor,
        edge_mask: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert adjacency matrix to edge indices and edge attributes.
        
        Args:
            adjacency: [N, N] adjacency matrix with distances
            edge_mask: [N, N] binary mask for valid edges
            node_mask: [N] binary mask for valid nodes
        
        Returns:
            edge_index: [2, E] edge indices
            edge_attr: [E] edge attributes (distances)
        """
        N = adjacency.shape[0]
        
        # Create edge indices for all possible edges
        row, col = torch.meshgrid(
            torch.arange(N, device=adjacency.device),
            torch.arange(N, device=adjacency.device),
            indexing='ij'
        )
        row = row.flatten()
        col = col.flatten()
        
        # Filter edges based on masks
        valid_nodes = node_mask.bool()
        valid_edges = edge_mask.bool()
        
        # Replace -inf values with -1e-40 in adjacency matrix
        adjacency = torch.where(
            torch.isinf(adjacency) & (adjacency < 0),
            torch.tensor(-1e-40, dtype=adjacency.dtype, device=adjacency.device),
            adjacency
        )
        
        # Edge is valid if:
        # 1. Both nodes are valid (node_mask)
        # 2. Edge mask indicates it's valid
        # 3. Adjacency value is not -1 (missing)
        valid_edge_mask = (
            valid_nodes[row] & valid_nodes[col] &
            valid_edges[row, col] &
            (adjacency[row, col] != -1) &
            torch.isfinite(adjacency[row, col])
        )
        
        # Apply threshold if specified
        if self.edge_threshold is not None:
            distances = adjacency[row, col]
            valid_edge_mask = valid_edge_mask & (distances <= self.edge_threshold)
        
        # Extract valid edges
        edge_index = torch.stack([row[valid_edge_mask], col[valid_edge_mask]], dim=0)
        edge_attr = adjacency[row[valid_edge_mask], col[valid_edge_mask]]
        
        # Add self-loops
        self_loops = torch.arange(N, device=adjacency.device)
        self_loop_edges = torch.stack([self_loops, self_loops], dim=0)
        self_loop_attr = torch.zeros(N, device=adjacency.device)
        
        # Combine with self-loops
        edge_index = torch.cat([edge_index, self_loop_edges], dim=1)
        edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)
        
        return edge_index, edge_attr
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[List[Data], torch.Tensor, int]:
        """
        Get a temporal graph sequence.
        
        Returns:
            graphs: List of Data objects (one per frame)
            node_masks: [T, N] tensor of node masks
            label: Binary label (0=Normal, 1=Anomalous)
        """
        h5_path = self._get_h5_path(idx)
        
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"HDF5 file not found: {h5_path}")
        
        graphs, node_masks = self._load_temporal_graphs(h5_path)
        label = self.df.iloc[idx]['label_binary']
        
        return graphs, node_masks, label


def collate_temporal_graphs(batch: List[Tuple[List[Data], torch.Tensor, int]]) -> Tuple[List[List[Data]], List[torch.Tensor], torch.Tensor]:
    """
    Custom collate function for temporal graph sequences.
    
    Args:
        batch: List of (graphs, node_masks, label) tuples
    
    Returns:
        batched_graphs: List of lists of Data objects (one list per sample)
        batched_masks: List of node mask tensors
        labels: Tensor of labels
    """
    graphs_list = [item[0] for item in batch]
    masks_list = [item[1] for item in batch]
    labels = torch.tensor([item[2] for item in batch], dtype=torch.long)
    
    return graphs_list, masks_list, labels


def create_dataloaders(
    root_dir: str,
    train_manifest: str,
    val_manifest: str,
    test_manifest: Optional[str] = None,
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
    normalize_features: bool = True,
    edge_threshold: Optional[float] = None,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create DataLoaders for train, validation, and test sets.
    
    Returns:
        train_loader, val_loader, test_loader (if test_manifest provided)
    """
    train_dataset = TemporalGraphDataset(
        manifest_path=train_manifest,
        root_dir=root_dir,
        normalize_features=normalize_features,
        edge_threshold=edge_threshold,
    )
    
    val_dataset = TemporalGraphDataset(
        manifest_path=val_manifest,
        root_dir=root_dir,
        normalize_features=normalize_features,
        edge_threshold=edge_threshold,
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
    if test_manifest and os.path.exists(os.path.join(root_dir, test_manifest)):
        test_dataset = TemporalGraphDataset(
            manifest_path=test_manifest,
            root_dir=root_dir,
            normalize_features=normalize_features,
            edge_threshold=edge_threshold,
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

