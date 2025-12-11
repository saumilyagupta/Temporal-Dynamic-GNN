"""
3D Dynamic Graph Neural Network for temporal graph sequence classification.
Combines spatial graph convolutions with temporal modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, LayerNorm
from torch_geometric.data import Data
from typing import List, Optional


class GraphConvLayer(nn.Module):
    """
    Graph Convolutional Layer with residual connection and layer normalization.
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        use_residual: bool = True,
        dropout: float = 0.0,
    ):
        super(GraphConvLayer, self).__init__()
        self.use_residual = use_residual and (in_dim == out_dim)
        self.conv = GCNConv(in_dim, out_dim)
        self.norm = LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [N, in_dim] node features
            edge_index: [2, E] edge indices
            edge_weight: [E] optional edge weights
        
        Returns:
            [N, out_dim] node features
        """
        out = self.conv(x, edge_index, edge_weight)
        out = self.norm(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        if self.use_residual:
            out = out + x
        
        return out


class TemporalModule(nn.Module):
    """
    Temporal modeling module using GRU to capture temporal dependencies.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        temporal_type: str = "gru",
    ):
        super(TemporalModule, self).__init__()
        self.temporal_type = temporal_type.lower()
        self.hidden_dim = hidden_dim
        
        if self.temporal_type == "gru":
            self.temporal = nn.GRU(
                input_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=False,  # [T, B*N, hidden_dim]
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=False,
            )
        elif self.temporal_type == "lstm":
            self.temporal = nn.LSTM(
                input_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=False,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=False,
            )
        else:
            raise ValueError(f"Unknown temporal_type: {temporal_type}. Must be 'gru' or 'lstm'")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process temporal sequence.
        
        Args:
            x: [T, B*N, input_dim] where T is time, B is batch, N is nodes
        
        Returns:
            [T, B*N, hidden_dim] temporal features
        """
        # x is already in [T, B*N, input_dim] format
        out, _ = self.temporal(x)
        return out


class GlobalPooling(nn.Module):
    """
    Global average pooling with masking support.
    Pools across nodes first, then across time.
    """
    
    def __init__(self):
        super(GlobalPooling, self).__init__()
    
    def forward(self, x: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        """
        Global average pooling with masking.
        
        Args:
            x: [T, N, hidden_dim] node features across time
            node_mask: [T, N] binary mask (1 = valid node, 0 = invalid)
        
        Returns:
            [hidden_dim] pooled feature vector
        """
        T, N, hidden_dim = x.shape
        
        # Expand mask to match feature dimensions
        mask = node_mask.unsqueeze(-1)  # [T, N, 1]
        
        # Mask out invalid nodes
        masked_x = x * mask
        
        # Pool across nodes (spatial pooling)
        # Sum valid features and divide by number of valid nodes per frame
        valid_nodes_per_frame = mask.sum(dim=1, keepdim=True)  # [T, 1, 1]
        valid_nodes_per_frame = torch.clamp(valid_nodes_per_frame, min=1.0)  # Avoid division by zero
        
        frame_features = masked_x.sum(dim=1) / valid_nodes_per_frame.squeeze(-1)  # [T, hidden_dim]
        
        # Pool across time (temporal pooling)
        # Simple average across all frames
        pooled = frame_features.mean(dim=0)  # [hidden_dim]
        
        return pooled


class DynamicGNN(nn.Module):
    """
    3D Dynamic Graph Neural Network for temporal graph sequence classification.
    
    Architecture:
    1. Per-frame GCN processing (spatial)
    2. Temporal modeling with GRU/LSTM
    3. Global average pooling
    4. Classification head
    """
    
    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 128,
        num_gcn_layers: int = 3,
        num_temporal_layers: int = 2,
        temporal_type: str = "gru",
        dropout: float = 0.3,
        use_residual: bool = True,
        num_classes: int = 2,
    ):
        """
        Args:
            input_dim: Dimension of input node features (default: 8)
            hidden_dim: Hidden dimension for GCN and temporal layers
            num_gcn_layers: Number of GCN layers per frame
            num_temporal_layers: Number of layers in temporal module
            temporal_type: Type of temporal module ('gru' or 'lstm')
            dropout: Dropout probability
            use_residual: Whether to use residual connections in GCN
            num_classes: Number of output classes (default: 2 for binary)
        """
        super(DynamicGNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_gcn_layers = num_gcn_layers
        
        # GCN layers for spatial processing
        self.gcn_layers = nn.ModuleList()
        
        # First layer: input_dim -> hidden_dim
        self.gcn_layers.append(
            GraphConvLayer(input_dim, hidden_dim, use_residual=False, dropout=dropout)
        )
        
        # Subsequent layers: hidden_dim -> hidden_dim
        for _ in range(num_gcn_layers - 1):
            self.gcn_layers.append(
                GraphConvLayer(hidden_dim, hidden_dim, use_residual=use_residual, dropout=dropout)
            )
        
        # Temporal modeling
        self.temporal_module = TemporalModule(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_temporal_layers,
            dropout=dropout,
            temporal_type=temporal_type,
        )
        
        # Global pooling
        self.pooling = GlobalPooling()
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
    
    def forward(
        self,
        graphs_list: List[List[Data]],
        node_masks_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass through the dynamic GNN.
        
        Args:
            graphs_list: List of lists of Data objects
                - Outer list: batch dimension
                - Inner list: temporal dimension (frames)
            node_masks_list: List of [T, N] node mask tensors
        
        Returns:
            [B, num_classes] logits
        """
        batch_size = len(graphs_list)
        batch_outputs = []
        
        # Process each sample in the batch
        for b in range(batch_size):
            graphs = graphs_list[b]  # List of Data objects (temporal frames)
            node_mask = node_masks_list[b]  # [T, N]
            
            T = len(graphs)
            if T == 0:
                # Handle empty sequence
                pooled = torch.zeros(self.hidden_dim, device=node_mask.device)
                batch_outputs.append(pooled)
                continue
            
            # Get number of nodes from first frame
            N = graphs[0].num_nodes
            
            # Process each frame with GCN layers
            frame_features = []
            for t in range(T):
                graph = graphs[t]
                x = graph.x  # [N, input_dim]
                edge_index = graph.edge_index
                edge_weight = graph.edge_attr if hasattr(graph, 'edge_attr') else None
                
                # Apply GCN layers
                for gcn_layer in self.gcn_layers:
                    x = gcn_layer(x, edge_index, edge_weight)
                
                frame_features.append(x)  # [N, hidden_dim]
            
            # Stack frames: [T, N, hidden_dim]
            temporal_features = torch.stack(frame_features, dim=0)
            
            # Reshape for temporal module: [T, N, hidden_dim] -> [T, N, hidden_dim]
            # Temporal module expects [T, batch*N, hidden_dim]
            T_seq, N_nodes, hidden = temporal_features.shape
            temporal_input = temporal_features.view(T_seq, N_nodes, hidden)
            
            # Apply temporal module
            temporal_output = self.temporal_module(temporal_input)  # [T, N, hidden_dim]
            
            # Global pooling with masking
            pooled = self.pooling(temporal_output, node_mask)  # [hidden_dim]
            
            batch_outputs.append(pooled)
        
        # Stack batch outputs: [B, hidden_dim]
        batch_features = torch.stack(batch_outputs, dim=0)
        
        # Classification
        logits = self.classifier(batch_features)  # [B, num_classes]
        
        return logits

