"""
TemporalGNN Model for dynamic graph sequence classification.
Combines spatial GCN layers with temporal GRU modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from typing import List, Optional


class TemporalGNN(nn.Module):
    """
    Temporal Graph Neural Network for video-level anomaly detection.
    
    Architecture:
    1. Per-frame spatial GCN processing
    2. Frame-level mean pooling
    3. Temporal GRU for sequence modeling
    4. Binary classification head
    
    Uses node_ids for identity alignment across frames.
    """
    
    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 64,
        temporal_dim: int = 128,
        num_gcn_layers: int = 2,
        num_gru_layers: int = 1,
        dropout: float = 0.3,
        num_classes: int = 2,
    ):
        """
        Args:
            input_dim: Dimension of input node features (default: 8)
            hidden_dim: Hidden dimension for GCN layers
            temporal_dim: Hidden dimension for GRU
            num_gcn_layers: Number of GCN layers
            num_gru_layers: Number of GRU layers
            dropout: Dropout probability
            num_classes: Number of output classes (2 for binary)
        """
        super(TemporalGNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.temporal_dim = temporal_dim
        self.num_gcn_layers = num_gcn_layers
        self.dropout = dropout
        
        # Spatial GCN layers
        self.gcn_layers = nn.ModuleList()
        
        # First layer: input_dim -> hidden_dim
        self.gcn_layers.append(GCNConv(input_dim, hidden_dim))
        
        # Additional layers: hidden_dim -> hidden_dim
        for _ in range(num_gcn_layers - 1):
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))
        
        # Batch normalization for each GCN layer
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_gcn_layers)
        ])
        
        # Temporal GRU
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=temporal_dim,
            num_layers=num_gru_layers,
            batch_first=False,  # [seq_len, batch, features]
            dropout=dropout if num_gru_layers > 1 else 0,
            bidirectional=False,
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(temporal_dim, temporal_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(temporal_dim // 2, num_classes),
        )
        
        # Dropout layer
        self.drop = nn.Dropout(dropout)
    
    def forward(self, batch) -> torch.Tensor:
        """
        Forward pass through the TemporalGNN using batched operations.
        
        Args:
            batch: PyG Batch object containing (B * T) flattened graphs.
                   Must have 't' attribute denoting sequence timestep.
        
        Returns:
            logits: [B, num_classes] classification logits
        """
        device = next(self.parameters()).device
        
        # Determine shapes
        # We need to know batch size (B) and sequence length (T)
        # Using the t attribute to find T
        if hasattr(batch, 't'):
            T = batch.t.max().item() + 1
        else:
            # Fallback if t is missing, assume T=50
            T = 50
            
        B = batch.num_graphs // T
        
        x = batch.x
        edge_index = batch.edge_index
        edge_weight = batch.edge_weight if hasattr(batch, 'edge_weight') else None
        
        # 1. Batched Spatial GCN Pass
        # We process all B*T graphs simultaneously
        if x is not None and x.shape[0] > 0:
            for i, gcn in enumerate(self.gcn_layers):
                x = gcn(x, edge_index, edge_weight)
                # Batch norm expects [N, C] or [N, C, L]
                if x.shape[0] > 1:
                    x = self.bn_layers[i](x)
                x = F.relu(x)
                x = self.drop(x)
                
            # 2. Frame-level mean pooling
            # Pool node features into graph-level features
            # x_pool: [B*T, hidden_dim]
            x_pool = global_mean_pool(x, batch.batch)
            
            # Handle empty graphs (global_mean_pool sets them to 0 automatically)
            # Ensure output shape is exactly B*T
            if x_pool.shape[0] < B * T:
                # This happens if trailing graphs are completely empty
                padded = torch.zeros((B * T, self.hidden_dim), device=device)
                padded[:x_pool.shape[0]] = x_pool
                x_pool = padded
        else:
            # Degenerate case: Entire batch has 0 nodes
            x_pool = torch.zeros((B * T, self.hidden_dim), device=device)
            
        # 3. Reshape for GRU
        # x_pool is ordered by graph index: [graph_0, graph_1, ..., graph_B*T-1]
        # Which is [batch_0_t0, batch_0_t1, ..., batch_1_t0, ...]
        # We reshape to [B, T, hidden_dim]
        H = x_pool.view(B, T, self.hidden_dim)
        
        # GRU expects [T, B, hidden_dim] when batch_first=False
        H = H.transpose(0, 1)
        
        # Apply GRU for temporal reasoning
        _, h_final = self.gru(H)  # h_final: [num_layers, B, temporal_dim]
        
        # Take the last layer's hidden state
        h_final = h_final[-1]  # [B, temporal_dim]
        
        # Classification
        logits = self.classifier(h_final)  # [B, num_classes]
        
        return logits


class TemporalGNNSimple(nn.Module):
    """
    Simplified TemporalGNN matching the exact architecture from docs.md.
    
    Architecture:
    - conv1: GCNConv(8, 64)
    - conv2: GCNConv(64, 64)
    - gru: GRU(64, 128)
    - fc: Linear(128, 1) for binary output
    """
    
    def __init__(self):
        super(TemporalGNNSimple, self).__init__()
        
        # Spatial GNN
        self.conv1 = GCNConv(8, 64)
        self.conv2 = GCNConv(64, 64)
        
        # Temporal model
        self.gru = nn.GRU(64, 128)
        
        # Final classifier (output 2 classes for consistency with loss function)
        self.fc = nn.Linear(128, 2)
    
    def _get_device(self) -> torch.device:
        """Get the device of model parameters."""
        return next(self.parameters()).device
    
    def forward(self, graphs_list: List[List[Data]]) -> torch.Tensor:
        """
        Forward pass matching docs.md implementation.
        
        Args:
            graphs_list: List of graph sequences [B, T=30]
        
        Returns:
            logits: [B, 2] classification logits
        """
        batch_size = len(graphs_list)
        device = self._get_device()
        
        batch_outputs = []
        
        for b in range(batch_size):
            graph_sequence = graphs_list[b]
            frame_embeddings = []
            
            for data in graph_sequence:
                data = data.to(device)
                
                # Handle empty graphs
                if data.x is None or data.x.shape[0] == 0:
                    frame_emb = torch.zeros(64, device=device)
                    frame_embeddings.append(frame_emb)
                    continue
                
                # Identity alignment using node_ids
                if hasattr(data, 'node_ids') and data.node_ids is not None:
                    sorted_idx = torch.argsort(data.node_ids)
                    x = data.x[sorted_idx]
                else:
                    x = data.x
                
                edge_index = data.edge_index
                edge_weight = data.edge_weight if hasattr(data, 'edge_weight') else None
                
                # Spatial GNN
                h = F.relu(self.conv1(x, edge_index, edge_weight))
                h = F.relu(self.conv2(h, edge_index, edge_weight))
                
                # Frame embedding (mean pooling)
                frame_emb = h.mean(dim=0)
                frame_embeddings.append(frame_emb)
            
            # Stack frames: [T, 64]
            H_sample = torch.stack(frame_embeddings, dim=0)
            batch_outputs.append(H_sample)
        
        # Stack batch: [T, B, 64]
        H = torch.stack(batch_outputs, dim=1)
        
        # Temporal reasoning
        _, h_final = self.gru(H)  # h_final: [1, B, 128]
        
        # Final prediction
        logits = self.fc(h_final.squeeze(0))  # [B, 2]
        
        return logits



