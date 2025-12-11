"""
Reusable training function for Optuna trials.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import GCN3D
from data import VideoGraphDataset, collate_fn
from config import TRAIN_MANIFEST, VAL_MANIFEST
from utils import calculate_metrics


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch in dataloader:
        batch = batch.to(device)
        labels = batch.y.float()
        
        optimizer.zero_grad()
        logits = model(batch).squeeze(-1)
        
        if logits.dim() == 0:
            logits = logits.unsqueeze(0)
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)
        
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        
        if probs.ndim == 0:
            probs = probs.reshape(1)
        if preds.ndim == 0:
            preds = preds.reshape(1)
        labels_np = labels.cpu().numpy()
        if labels_np.ndim == 0:
            labels_np = labels_np.reshape(1)
        
        all_preds.extend(preds.tolist())
        all_labels.extend(labels_np.tolist())
    
    avg_loss = total_loss / len(dataloader)
    metrics = calculate_metrics(all_labels, all_preds)
    return avg_loss, metrics['accuracy']


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            labels = batch.y.float()
            
            logits = model(batch).squeeze(-1)
            
            if logits.dim() == 0:
                logits = logits.unsqueeze(0)
            if labels.dim() == 0:
                labels = labels.unsqueeze(0)
            
            loss = criterion(logits, labels)
            total_loss += loss.item()
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            
            if probs.ndim == 0:
                probs = probs.reshape(1)
            if preds.ndim == 0:
                preds = preds.reshape(1)
            labels_np = labels.cpu().numpy()
            if labels_np.ndim == 0:
                labels_np = labels_np.reshape(1)
            
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())
            all_labels.extend(labels_np.tolist())
    
    avg_loss = total_loss / len(dataloader)
    metrics = calculate_metrics(all_labels, all_preds, all_probs)
    return avg_loss, metrics


def train_with_hyperparameters(model_config, train_config, trial_dir=None, verbose=False):
    """
    Train model with given hyperparameters.
    
    Args:
        model_config: Dictionary with model hyperparameters
        train_config: Dictionary with training hyperparameters
        trial_dir: Optional directory to save trial checkpoints/logs
        verbose: Whether to print progress
        
    Returns:
        float: Best validation accuracy achieved
    """
    device = torch.device(train_config['device'])
    
    # Create datasets
    train_dataset = VideoGraphDataset(
        TRAIN_MANIFEST,
        max_frames=train_config['max_frames'],
        normalize_coords=train_config['normalize_coords']
    )
    
    val_dataset = VideoGraphDataset(
        VAL_MANIFEST,
        max_frames=train_config['max_frames'],
        normalize_coords=train_config['normalize_coords']
    )
    
    if len(val_dataset) == 0:
        raise ValueError("Validation dataset is empty! Cannot optimize on validation accuracy.")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config['batch_size'],
        shuffle=True,
        num_workers=train_config['num_workers'],
        pin_memory=train_config['pin_memory'],
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config['batch_size'],
        shuffle=False,
        num_workers=train_config['num_workers'],
        pin_memory=train_config['pin_memory'],
        collate_fn=collate_fn
    )
    
    # Create model
    model = GCN3D(**model_config).to(device)
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=train_config['learning_rate'],
        weight_decay=train_config['weight_decay']
    )
    
    # Training loop with early stopping
    best_val_acc = 0.0
    patience_counter = 0
    best_epoch = 0
    
    if trial_dir:
        os.makedirs(trial_dir, exist_ok=True)
        checkpoint_path = os.path.join(trial_dir, 'checkpoint.pth')
        best_checkpoint_path = os.path.join(trial_dir, 'checkpoint_best.pth')
    
    for epoch in range(1, train_config['num_epochs'] + 1):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        val_acc = val_metrics['accuracy']
        
        if verbose:
            print(f"Epoch {epoch}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
        
        # Early stopping logic
        is_best = val_acc > (best_val_acc + train_config['min_delta'])
        if is_best:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            
            # Save best checkpoint
            if trial_dir:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'model_config': model_config,
                    'train_config': train_config,
                }
                torch.save(checkpoint, best_checkpoint_path)
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= train_config['patience']:
            if verbose:
                print(f"Early stopping at epoch {epoch}. Best val acc: {best_val_acc:.4f} at epoch {best_epoch}")
            break
    
    return best_val_acc


