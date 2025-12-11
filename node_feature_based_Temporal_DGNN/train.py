"""
Training script for TemporalGNN using PyTorch Lightning.
"""

import os
import yaml
import argparse
import shutil
from datetime import datetime
import torch
import torch.multiprocessing
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

# Fix for "too many open files" error with multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from dataset import create_dataloaders, TemporalGraphDataset
from model import TemporalGNN, TemporalGNNSimple
from loss import calculate_alpha_from_dataset
from lightning_module import TemporalGNNLightning
from utils import set_seed, get_device, count_parameters


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Train TemporalGNN')
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility',
    )
    parser.add_argument(
        '--exp_name',
        type=str,
        default=None,
        help='Custom experiment name (optional)',
    )
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create experiment folder with timestamp
    exp_start_time = datetime.now()
    exp_timestamp = exp_start_time.strftime("%Y%m%d_%H%M%S")
    
    if args.exp_name:
        exp_dir = os.path.join("experiments", f"{args.exp_name}_{exp_timestamp}")
    else:
        exp_dir = os.path.join("experiments", f"exp_{exp_timestamp}")
    
    os.makedirs(exp_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("TemporalGNN Training")
    print("=" * 60)
    print(f"Experiment directory: {exp_dir}")
    print(f"Experiment start time: {exp_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60 + "\n")
    
    # Load configuration
    config = load_config(args.config)
    
    # Set GPU device if specified
    gpu_id = config['training'].get('gpu_id', 0)
    if gpu_id >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        print(f"Using GPU: {gpu_id}")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        print("Using CPU (gpu_id=-1)")
    
    # Save config to experiment folder
    config_save_path = os.path.join(exp_dir, "config.yaml")
    shutil.copy(args.config, config_save_path)
    
    # Print configuration
    print("Configuration:")
    print("-" * 40)
    print(yaml.dump(config, default_flow_style=False))
    print("-" * 40 + "\n")
    
    # Get device
    device = get_device()
    
    # Create datasets and dataloaders
    print("\nCreating datasets and dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_root=config['data']['root_dir'],
        train_manifest=config['data']['train_manifest'],
        val_manifest=config['data']['val_manifest'],
        test_manifest=config['data'].get('test_manifest'),
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        normalize_features=config['data']['normalize_features'],
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Val batches: {len(val_loader)}")
    if test_loader:
        print(f"Test samples: {len(test_loader.dataset)}")
        print(f"Test batches: {len(test_loader)}")
    
    # Calculate alpha for focal loss from training set
    print("\nCalculating focal loss alpha...")
    train_manifest_path = os.path.join(
        config['data']['root_dir'],
        config['data']['train_manifest']
    )
    temp_dataset = TemporalGraphDataset(
        manifest_path=train_manifest_path,
        normalize_features=False,  # Don't need normalization for counting
    )
    alpha = calculate_alpha_from_dataset(temp_dataset)
    
    # Create model
    print("\nCreating model...")
    if config['model']['type'].lower() == 'simple':
        model = TemporalGNNSimple()
        print("Using TemporalGNNSimple (matches docs.md architecture)")
    else:
        model = TemporalGNN(
            input_dim=config['model']['input_dim'],
            hidden_dim=config['model']['hidden_dim'],
            temporal_dim=config['model']['temporal_dim'],
            num_gcn_layers=config['model']['num_gcn_layers'],
            num_gru_layers=config['model']['num_gru_layers'],
            dropout=config['model']['dropout'],
            num_classes=config['model']['num_classes'],
        )
        print("Using configurable TemporalGNN")
    
    # Print model info
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")
    
    # Save model architecture
    model_arch_path = os.path.join(exp_dir, "model_architecture.txt")
    with open(model_arch_path, 'w') as f:
        f.write(str(model))
        f.write(f"\n\nTotal parameters: {num_params:,}")
    
    # Create Lightning module
    lightning_module = TemporalGNNLightning(
        model=model,
        focal_loss_alpha=alpha,
        focal_loss_gamma=float(config['loss']['gamma']),
        learning_rate=float(config['training']['learning_rate']),
        weight_decay=float(config['training']['weight_decay']),
        optimizer=config['training']['optimizer'],
        scheduler=config['training']['scheduler'],
        monitor_metric=config['training']['monitor_metric'],
        monitor_mode=config['training']['monitor_mode'],
    )
    
    # Setup callbacks
    callbacks = []
    
    # Create checkpoints directory in experiment folder
    checkpoint_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='{epoch:02d}-{val_loss:.4f}-{val_f1:.4f}',
        monitor=config['training']['monitor_metric'],
        mode=config['training']['monitor_mode'],
        save_top_k=config['callbacks']['checkpoint']['save_top_k'],
        save_last=config['callbacks']['checkpoint']['save_last'],
        every_n_epochs=config['callbacks']['checkpoint']['every_n_epochs'],
        verbose=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor=config['training']['monitor_metric'],
        mode=config['training']['monitor_mode'],
        patience=config['callbacks']['early_stopping']['patience'],
        min_delta=config['callbacks']['early_stopping']['min_delta'],
        verbose=True,
    )
    callbacks.append(early_stopping)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    # Setup logger
    log_dir = os.path.join(exp_dir, "logs")
    logger = TensorBoardLogger(
        save_dir=log_dir,
        name="training",
        version=None,
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator='gpu' if device.type == 'cuda' else 'cpu',
        devices=1 if device.type == 'cuda' else 'auto',
        gradient_clip_val=config['training']['gradient_clip_val'],
        accumulate_grad_batches=config['training']['accumulate_grad_batches'],
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=config['logging']['log_every_n_steps'],
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
        precision='16-mixed' if device.type == 'cuda' else '32',
    )
    
    # Train model
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")
    
    trainer.fit(
        lightning_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")
    print(f"Best {config['training']['monitor_metric']}: {checkpoint_callback.best_model_score:.4f}")
    
    # Copy best model to experiment folder with readable name
    if checkpoint_callback.best_model_path:
        best_model_dest = os.path.join(exp_dir, "best_model.ckpt")
        shutil.copy(checkpoint_callback.best_model_path, best_model_dest)
        print(f"Best model copied to: {best_model_dest}")
    
    # Test on test set if available
    if test_loader:
        print("\n" + "=" * 60)
        print("Evaluating on test set...")
        print("=" * 60 + "\n")
        
        trainer.test(lightning_module, dataloaders=test_loader)
        
        print("\nTest metrics:")
        for key, value in lightning_module.test_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # Save test results
        results_dir = os.path.join(exp_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Save metrics
        metrics_file = os.path.join(results_dir, "test_metrics.txt")
        with open(metrics_file, 'w') as f:
            f.write(f"Experiment Start Time: {exp_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Experiment End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Best Model: {checkpoint_callback.best_model_path}\n")
            f.write(f"Best {config['training']['monitor_metric']}: {checkpoint_callback.best_model_score:.4f}\n\n")
            f.write("Test Metrics:\n")
            f.write("=" * 40 + "\n")
            for key, value in lightning_module.test_metrics.items():
                f.write(f"{key}: {value:.4f}\n")
        
        print(f"\nResults saved to: {results_dir}")
    
    # Save experiment summary
    summary_file = os.path.join(exp_dir, "experiment_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("TemporalGNN Training Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Experiment Start Time: {exp_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Experiment End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Config File: {args.config}\n")
        f.write(f"Random Seed: {args.seed}\n")
        f.write(f"Best Model: {checkpoint_callback.best_model_path}\n")
        f.write(f"Best {config['training']['monitor_metric']}: {checkpoint_callback.best_model_score:.4f}\n")
        f.write(f"\nModel Parameters: {num_params:,}\n")
        f.write(f"Focal Loss Alpha: {alpha:.4f}\n")
        f.write(f"\nDataset Info:\n")
        f.write(f"  Train samples: {len(train_loader.dataset)}\n")
        f.write(f"  Val samples: {len(val_loader.dataset)}\n")
        if test_loader:
            f.write(f"  Test samples: {len(test_loader.dataset)}\n")
    
    print(f"\nExperiment summary saved to: {summary_file}")
    print(f"All experiment files saved in: {exp_dir}")
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()

