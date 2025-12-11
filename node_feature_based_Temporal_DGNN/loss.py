"""
Focal Loss implementation for binary classification.
Handles class imbalance by focusing on hard examples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification.
    
    Formula: FL(p_t) = -α * (1 - p_t)^γ * log(p_t)
    
    Where:
        - p_t is the predicted probability for the true class
        - α is the balancing factor (class weight)
        - γ is the focusing parameter
    """
    
    def __init__(self, alpha: float = 0.5, gamma: float = 1.0, reduction: str = 'mean'):
        """
        Args:
            alpha: Balancing factor for class weights
                   For anomaly detection: α = num_normal / (num_normal + num_anomalous)
                   Higher α gives more weight to anomalous class
            gamma: Focusing parameter (higher values focus more on hard examples)
            reduction: 'mean', 'sum', or 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Reduction must be 'mean', 'sum', or 'none', got {reduction}")
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: [B, 2] logits from model (before softmax)
            targets: [B] binary labels (0 or 1)
        
        Returns:
            Focal loss value
        """
        # Convert logits to probabilities using softmax
        probs = F.softmax(inputs, dim=1)
        
        # Get probability of the true class
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # [B]
        
        # Clamp p_t to avoid numerical instability
        p_t = torch.clamp(p_t, min=1e-8, max=1.0 - 1e-8)
        
        # Compute alpha for each sample
        # For class 0 (Normal): alpha = self.alpha
        # For class 1 (Anomalous): alpha = 1 - self.alpha
        alpha_t = torch.where(
            targets == 1,
            torch.tensor(1.0 - self.alpha, device=inputs.device, dtype=inputs.dtype),
            torch.tensor(self.alpha, device=inputs.device, dtype=inputs.dtype)
        )
        
        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Compute cross-entropy: -log(p_t)
        ce_loss = -torch.log(p_t)
        
        # Combine: alpha * focal_weight * ce_loss
        focal_loss = alpha_t * focal_weight * ce_loss
        
        # Check for NaN or Inf values
        if torch.isnan(focal_loss).any() or torch.isinf(focal_loss).any():
            # Fallback to standard cross-entropy
            focal_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class BCEWithLogitsLoss(nn.Module):
    """
    Binary Cross-Entropy with class weighting.
    Alternative to Focal Loss for simpler training.
    """
    
    def __init__(self, pos_weight: float = 1.0, reduction: str = 'mean'):
        """
        Args:
            pos_weight: Weight for positive (anomalous) class
            reduction: 'mean', 'sum', or 'none'
        """
        super(BCEWithLogitsLoss, self).__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted BCE loss.
        
        Args:
            inputs: [B, 2] logits from model
            targets: [B] binary labels
        
        Returns:
            Loss value
        """
        # Use class 1 (anomalous) probability
        logits = inputs[:, 1] - inputs[:, 0]  # log(p_anomalous / p_normal)
        
        pos_weight = torch.tensor(self.pos_weight, device=inputs.device, dtype=inputs.dtype)
        
        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets.float(),
            pos_weight=pos_weight,
            reduction=self.reduction,
        )
        
        return loss


def calculate_alpha_from_dataset(dataset) -> float:
    """
    Calculate alpha parameter for focal loss from dataset class distribution.
    
    Formula: α = num_normal / (num_normal + num_anomalous)
    
    Args:
        dataset: TemporalGraphDataset instance
    
    Returns:
        alpha: Balancing factor for focal loss
    """
    df = dataset.df
    num_normal = (df['label'] == 'Normal').sum()
    num_anomalous = (df['label'] == 'Anomalous').sum()
    
    if num_normal + num_anomalous == 0:
        return 0.5  # Default balanced
    
    alpha = num_normal / (num_normal + num_anomalous)
    
    print(f"Class distribution - Normal: {num_normal}, Anomalous: {num_anomalous}")
    print(f"Calculated alpha for focal loss: {alpha:.4f}")
    
    return alpha


def calculate_pos_weight(dataset) -> float:
    """
    Calculate positive class weight for BCE loss.
    
    Formula: pos_weight = num_normal / num_anomalous
    
    Args:
        dataset: TemporalGraphDataset instance
    
    Returns:
        pos_weight: Weight for positive (anomalous) class
    """
    df = dataset.df
    num_normal = (df['label'] == 'Normal').sum()
    num_anomalous = (df['label'] == 'Anomalous').sum()
    
    if num_anomalous == 0:
        return 1.0  # Default
    
    pos_weight = num_normal / num_anomalous
    
    print(f"Class distribution - Normal: {num_normal}, Anomalous: {num_anomalous}")
    print(f"Calculated pos_weight for BCE: {pos_weight:.4f}")
    
    return pos_weight



