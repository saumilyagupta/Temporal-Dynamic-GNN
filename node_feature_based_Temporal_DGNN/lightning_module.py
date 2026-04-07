"""
PyTorch Lightning module for TemporalGNN training.
Handles training, validation, and testing with proper metric tracking.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from typing import Dict, Any, List

from model import TemporalGNN, TemporalGNNSimple
from loss import FocalLoss
from utils import compute_metrics


class TemporalGNNLightning(pl.LightningModule):
    """
    PyTorch Lightning module for TemporalGNN.
    
    Handles:
    - Training with focal loss for class imbalance
    - Validation and testing with comprehensive metrics
    - Learning rate scheduling
    """
    
    def __init__(
        self,
        model: nn.Module,
        focal_loss_alpha: float = 0.5,
        focal_loss_gamma: float = 1.0,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        optimizer: str = "adam",
        scheduler: str = "plateau",
        monitor_metric: str = "val_loss",
        monitor_mode: str = "min",
    ):
        """
        Args:
            model: TemporalGNN or TemporalGNNSimple model instance
            focal_loss_alpha: Alpha parameter for focal loss (class weight)
            focal_loss_gamma: Gamma parameter for focal loss (focusing)
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
            optimizer: Optimizer type ('adam' or 'adamw')
            scheduler: LR scheduler type ('plateau' or 'cosine')
            monitor_metric: Metric to monitor for checkpointing
            monitor_mode: 'min' or 'max'
        """
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        
        self.model = model
        self.criterion = FocalLoss(alpha=focal_loss_alpha, gamma=focal_loss_gamma)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer
        self.scheduler_name = scheduler
        self.monitor_metric = monitor_metric
        self.monitor_mode = monitor_mode
        
        # Metrics storage
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
        # Test results storage
        self.test_metrics = {}
        self.test_predictions = None
        self.test_labels = None
        self.test_probabilities = None
    
    def forward(self, batch) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(batch)
    
    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Training step."""
        graphs_batch, labels = batch
        # Find B by using PyG batch attributes: batch.batch.max() + 1 gives total graphs = B * T
        # Assuming T=50 for logging batch size
        T = graphs_batch.t.max().item() + 1 if hasattr(graphs_batch, 't') else 50
        batch_size = (graphs_batch.batch.max().item() + 1) // T
        
        # Forward pass
        logits = self.forward(graphs_batch)
        
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
        
        # Log step metrics (batch_size for proper averaging)
        acc = (preds == labels).float().mean()
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        
        return loss
    
    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Validation step."""
        graphs_batch, labels = batch
        T = graphs_batch.t.max().item() + 1 if hasattr(graphs_batch, 't') else 50
        batch_size = (graphs_batch.batch.max().item() + 1) // T
        
        # Forward pass
        logits = self.forward(graphs_batch)
        
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
        
        # Log step metrics (batch_size for proper averaging)
        acc = (preds == labels).float().mean()
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        
        return loss
    
    def test_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Test step."""
        graphs_batch, labels = batch
        T = graphs_batch.t.max().item() + 1 if hasattr(graphs_batch, 't') else 50
        batch_size = (graphs_batch.batch.max().item() + 1) // T
        
        # Forward pass
        logits = self.forward(graphs_batch)
        
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
        
        # Log step metrics (batch_size for proper averaging)
        acc = (preds == labels).float().mean()
        self.log('test_loss', loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('test_acc', acc, on_step=False, on_epoch=True, batch_size=batch_size)
        
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
            all_probs[:, 1].numpy(),  # Probability of anomalous class
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
            all_probs[:, 1].numpy(),
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
            all_probs[:, 1].numpy(),
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
        # Select optimizer
        if self.optimizer_name.lower() == 'adamw':
            optimizer = AdamW(
                self.parameters(),
                lr=float(self.learning_rate),
                weight_decay=float(self.weight_decay),
            )
        else:
            optimizer = Adam(
                self.parameters(),
                lr=float(self.learning_rate),
                weight_decay=float(self.weight_decay),
            )
        
        # Select scheduler
        if self.scheduler_name.lower() == 'cosine':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=50,  # Will be updated if trainer has different max_epochs
                eta_min=1e-7,
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                },
            }
        else:
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode=self.monitor_mode,
                factor=0.5,
                patience=5,
                min_lr=1e-7,
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': self.monitor_metric,
                },
            }

