"""
Training script for 3D Graph Convolutional Network for video anomaly detection.
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import GCN3D
from data import VideoGraphDataset, collate_fn
from config import (
    TRAIN_MANIFEST, VAL_MANIFEST, MODEL_CONFIG, TRAIN_CONFIG, 
    CHECKPOINT_DIR, LOG_DIR, LOG_CONFIG
)
from utils import (
    save_checkpoint, load_checkpoint, plot_training_curves,
    calculate_metrics, print_metrics
)


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """
    Train for one epoch.
    
    Returns:
        avg_loss: Average loss
        accuracy: Accuracy
    """
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch_idx, batch in enumerate(dataloader):
        batch = batch.to(device)
        labels = batch.y.float()
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(batch).squeeze(-1)  # Squeeze last dimension only
        
        # Ensure shapes match
        if logits.dim() == 0:
            logits = logits.unsqueeze(0)
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)
        
        # Calculate loss
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        
        # Ensure arrays are flat for extending
        if probs.ndim == 0:
            probs = probs.reshape(1)
        if preds.ndim == 0:
            preds = preds.reshape(1)
        labels_np = labels.cpu().numpy()
        if labels_np.ndim == 0:
            labels_np = labels_np.reshape(1)
        
        all_preds.extend(preds.tolist())
        all_labels.extend(labels_np.tolist())
        
        # Logging
        if batch_idx % LOG_CONFIG['log_interval'] == 0:
            current_loss = total_loss / (batch_idx + 1)
            print(f'Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, '
                  f'Loss: {current_loss:.4f}')
    
    avg_loss = total_loss / len(dataloader)
    metrics = calculate_metrics(all_labels, all_preds)
    
    return avg_loss, metrics['accuracy']


def validate(model, dataloader, criterion, device):
    """
    Validate the model.
    
    Returns:
        avg_loss: Average loss
        metrics: Dictionary of metrics
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            labels = batch.y.float()
            
            # Forward pass
            logits = model(batch).squeeze(-1)  # Squeeze last dimension only
            
            # Ensure shapes match
            if logits.dim() == 0:
                logits = logits.unsqueeze(0)
            if labels.dim() == 0:
                labels = labels.unsqueeze(0)
            
            loss = criterion(logits, labels)
            
            # Metrics
            total_loss += loss.item()
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            
            # Ensure arrays are flat for extending
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


def main():
    """Main training function."""
    print("=" * 80)
    print("3D GCN Video Anomaly Detection - Training")
    print("=" * 80)
    
    # Configuration
    device = torch.device(TRAIN_CONFIG['device'])
    print(f"\nUsing device: {device}")
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = VideoGraphDataset(
        TRAIN_MANIFEST,
        max_frames=TRAIN_CONFIG['max_frames'],
        normalize_coords=TRAIN_CONFIG['normalize_coords']
    )
    
    val_dataset = VideoGraphDataset(
        VAL_MANIFEST,
        max_frames=TRAIN_CONFIG['max_frames'],
        normalize_coords=TRAIN_CONFIG['normalize_coords']
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Warn if no validation data
    if len(val_dataset) == 0:
        print("\nWARNING: No validation samples found!")
        print("Training will continue but validation metrics will be skipped.")
        print("Consider checking your data split or using a portion of training data for validation.")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_CONFIG['batch_size'],
        shuffle=True,
        num_workers=TRAIN_CONFIG['num_workers'],
        pin_memory=TRAIN_CONFIG['pin_memory'],
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAIN_CONFIG['batch_size'],
        shuffle=False,
        num_workers=TRAIN_CONFIG['num_workers'],
        pin_memory=TRAIN_CONFIG['pin_memory'],
        collate_fn=collate_fn
    ) if len(val_dataset) > 0 else None
    
    # Create model
    print("\nCreating model...")
    model = GCN3D(**MODEL_CONFIG).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Loss and optimizer (weighted for class imbalance)
    pos_weight = torch.tensor([TRAIN_CONFIG['pos_weight']]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(
        model.parameters(),
        lr=TRAIN_CONFIG['learning_rate'],
        weight_decay=TRAIN_CONFIG['weight_decay']
    )
    
    # Training history
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # Early stopping
    best_val_acc = 0.0
    patience_counter = 0
    best_epoch = 0
    
    # Checkpoint paths
    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'checkpoint.pth')
    best_checkpoint_path = os.path.join(CHECKPOINT_DIR, 'checkpoint_best.pth')
    
    # Training loop
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)
    
    start_time = time.time()
    
    for epoch in range(1, TRAIN_CONFIG['num_epochs'] + 1):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        if val_loader is not None:
            val_loss, val_metrics = validate(model, val_loader, criterion, device)
            val_acc = val_metrics['accuracy']
        else:
            # No validation data - use training metrics as placeholder
            val_loss = train_loss
            val_metrics = {'accuracy': train_acc}
            val_acc = train_acc
            print("  [No validation data - using training metrics]")
        
        # Record history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        epoch_time = time.time() - epoch_start
        
        # Print metrics
        print(f"\nEpoch {epoch}/{TRAIN_CONFIG['num_epochs']} ({epoch_time:.2f}s)")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print_metrics(val_metrics, 'Val')
        
        # Save checkpoint
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Save checkpoint
        if epoch % LOG_CONFIG['save_interval'] == 0 or is_best:
            save_checkpoint(
                model, optimizer, epoch, val_loss, checkpoint_path, is_best=is_best
            )
        
        # Plot curves
        if epoch % LOG_CONFIG['plot_interval'] == 0:
            plot_path = os.path.join(LOG_DIR, 'training_curves.png')
            plot_training_curves(train_losses, val_losses, train_accs, val_accs, plot_path)
        
        # Save training logs
        log_df = pd.DataFrame({
            'epoch': range(1, len(train_losses) + 1),
            'train_loss': train_losses,
            'val_loss': val_losses,
            'train_acc': train_accs,
            'val_acc': val_accs,
        })
        log_df.to_csv(os.path.join(LOG_DIR, 'training_log.csv'), index=False)
        
        # Early stopping (only if we have validation data)
        if val_loader is not None:
            if patience_counter >= TRAIN_CONFIG['patience']:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {TRAIN_CONFIG['patience']} epochs)")
                print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
                break
        
        print("-" * 80)
    
    total_time = time.time() - start_time
    
    # Final summary
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Total training time: {total_time/60:.2f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
    print(f"\nCheckpoints saved to: {CHECKPOINT_DIR}")
    print(f"Training logs saved to: {LOG_DIR}")
    print("=" * 80)
    
    # Load best model for final evaluation
    if os.path.exists(best_checkpoint_path):
        print(f"\nLoading best model from epoch {best_epoch}...")
        load_checkpoint(best_checkpoint_path, model, optimizer)
        print("Best model loaded successfully!")


if __name__ == '__main__':
    main()

