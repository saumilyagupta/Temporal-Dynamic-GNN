"""
PyTorch Lightning module for Dynamic GNN training.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Dict, Any, Optional

from model import DynamicGNN
from loss import FocalLoss
from utils import compute_metrics


class DynamicGNNLightning(pl.LightningModule):
    """
    PyTorch Lightning module for Dynamic GNN.
    """
    
    def __init__(
        self,
        model: DynamicGNN,
        focal_loss_alpha: float = 0.5,
        focal_loss_gamma: float = 1.0,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        monitor_metric: str = "val_loss",
        monitor_mode: str = "min",
    ):
        """
        Args:
            model: DynamicGNN model instance
            focal_loss_alpha: Alpha parameter for focal loss
            focal_loss_gamma: Gamma parameter for focal loss
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
            monitor_metric: Metric to monitor for checkpointing
            monitor_mode: 'min' or 'max'
        """
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        
        self.model = model
        self.criterion = FocalLoss(alpha=focal_loss_alpha, gamma=focal_loss_gamma)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.monitor_metric = monitor_metric
        self.monitor_mode = monitor_mode
        
        # Initialize metrics storage
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
    
    def forward(
        self,
        graphs_list: list,
        node_masks_list: list,
    ) -> torch.Tensor:
        """Forward pass."""
        return self.model(graphs_list, node_masks_list)
    
    def training_step(self, batch: tuple, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Training step."""
        graphs_list, node_masks_list, labels = batch
        
        # Forward pass
        logits = self.forward(graphs_list, node_masks_list)
        
        # Compute loss
        loss = self.criterion(logits, labels)
        
        # Compute predictions
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        
        # Store outputs for epoch-end metrics
        self.training_step_outputs.append({
            'loss': loss.detach(),
            'preds': preds.detach().cpu(),
            'labels': labels.detach().cpu(),
            'probs': probs.detach().cpu(),
        })
        
        # Log metrics
        acc = (preds == labels).float().mean()
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: tuple, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step."""
        graphs_list, node_masks_list, labels = batch
        
        # Forward pass
        logits = self.forward(graphs_list, node_masks_list)
        
        # Compute loss
        loss = self.criterion(logits, labels)
        
        # Compute predictions
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        
        # Store outputs for epoch-end metrics
        self.validation_step_outputs.append({
            'loss': loss.detach(),
            'preds': preds.detach().cpu(),
            'labels': labels.detach().cpu(),
            'probs': probs.detach().cpu(),
        })
        
        # Log metrics
        acc = (preds == labels).float().mean()
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch: tuple, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Test step."""
        graphs_list, node_masks_list, labels = batch
        
        # Forward pass
        logits = self.forward(graphs_list, node_masks_list)
        
        # Compute loss
        loss = self.criterion(logits, labels)
        
        # Compute predictions
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        
        # Store outputs for epoch-end metrics
        self.test_step_outputs.append({
            'loss': loss.detach(),
            'preds': preds.detach().cpu(),
            'labels': labels.detach().cpu(),
            'probs': probs.detach().cpu(),
        })
        
        # Log metrics
        acc = (preds == labels).float().mean()
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True)
        
        return loss
    
    def on_train_epoch_end(self):
        """Compute and log training metrics at end of epoch."""
        if not self.training_step_outputs:
            return
        
        # Aggregate outputs
        all_preds = torch.cat([out['preds'] for out in self.training_step_outputs])
        all_labels = torch.cat([out['labels'] for out in self.training_step_outputs])
        all_probs = torch.cat([out['probs'] for out in self.training_step_outputs])
        
        # Compute metrics
        metrics = compute_metrics(
            all_labels.numpy(),
            all_preds.numpy(),
            all_probs.numpy(),
        )
        
        # Log metrics
        for key, value in metrics.items():
            self.log(f'train_{key}', value, on_epoch=True)
        
        # Clear outputs
        self.training_step_outputs.clear()
    
    def on_validation_epoch_end(self):
        """Compute and log validation metrics at end of epoch."""
        if not self.validation_step_outputs:
            return
        
        # Aggregate outputs
        all_preds = torch.cat([out['preds'] for out in self.validation_step_outputs])
        all_labels = torch.cat([out['labels'] for out in self.validation_step_outputs])
        all_probs = torch.cat([out['probs'] for out in self.validation_step_outputs])
        
        # Compute metrics
        metrics = compute_metrics(
            all_labels.numpy(),
            all_preds.numpy(),
            all_probs.numpy(),
        )
        
        # Log metrics
        for key, value in metrics.items():
            self.log(f'val_{key}', value, on_epoch=True)
        
        # Clear outputs
        self.validation_step_outputs.clear()
    
    def on_test_epoch_end(self):
        """Compute and log test metrics at end of epoch."""
        if not self.test_step_outputs:
            return
        
        # Aggregate outputs
        all_preds = torch.cat([out['preds'] for out in self.test_step_outputs])
        all_labels = torch.cat([out['labels'] for out in self.test_step_outputs])
        all_probs = torch.cat([out['probs'] for out in self.test_step_outputs])
        
        # Compute metrics
        metrics = compute_metrics(
            all_labels.numpy(),
            all_preds.numpy(),
            all_probs.numpy(),
        )
        
        # Log metrics
        for key, value in metrics.items():
            self.log(f'test_{key}', value, on_epoch=True)
        
        # Store for external access
        self.test_metrics = metrics
        self.test_predictions = all_preds.numpy()
        self.test_labels = all_labels.numpy()
        self.test_probabilities = all_probs.numpy()
        
        # Clear outputs
        self.test_step_outputs.clear()
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = Adam(
            self.parameters(),
            lr=float(self.learning_rate),
            weight_decay=float(self.weight_decay),
        )
        
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=self.monitor_mode,
            factor=0.5,
            patience=5,
            verbose=True,
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': self.monitor_metric,
            },
        }

