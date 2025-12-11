"""
Training script for Dynamic GNN using PyTorch Lightning.
"""

import os
import yaml
import argparse
import shutil
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from dataset import create_dataloaders
from model import DynamicGNN
from loss import calculate_alpha_from_dataset
from lightning_module import DynamicGNNLightning
from utils import set_seed, get_device, count_parameters


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Train Dynamic GNN')
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
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create experiment folder with timestamp
    exp_start_time = datetime.now()
    exp_timestamp = exp_start_time.strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join("experiments", f"exp_{exp_timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    print(f"\nExperiment directory: {exp_dir}")
    print(f"Experiment start time: {exp_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Load configuration
    config = load_config(args.config)
    
    # Save config to experiment folder
    config_save_path = os.path.join(exp_dir, "config.yaml")
    shutil.copy(args.config, config_save_path)
    
    # Print configuration
    print("=" * 50)
    print("Configuration:")
    print("=" * 50)
    print(yaml.dump(config, default_flow_style=False))
    print("=" * 50)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create datasets and dataloaders
    print("\nCreating datasets and dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        root_dir=config['data']['root_dir'],
        train_manifest=config['data']['train_manifest'],
        val_manifest=config['data']['val_manifest'],
        test_manifest=config['data'].get('test_manifest'),
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        normalize_features=True,
        edge_threshold=None,
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    if test_loader:
        print(f"Test batches: {len(test_loader)}")
    
    # Calculate alpha for focal loss from training set
    print("\nCalculating focal loss alpha...")
    from dataset import TemporalGraphDataset
    train_dataset = TemporalGraphDataset(
        manifest_path=config['data']['train_manifest'],
        root_dir=config['data']['root_dir'],
        normalize_features=True,
    )
    alpha = calculate_alpha_from_dataset(train_dataset)
    
    # Create model
    print("\nCreating model...")
    model = DynamicGNN(
        input_dim=config['model']['input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_gcn_layers=config['model']['num_gcn_layers'],
        num_temporal_layers=config['model']['num_temporal_layers'],
        temporal_type=config['model']['temporal_type'],
        dropout=config['model']['dropout'],
        use_residual=config['model']['use_residual'],
        num_classes=2,
    )
    
    # Print model info
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")
    
    # Create Lightning module
    lightning_module = DynamicGNNLightning(
        model=model,
        focal_loss_alpha=alpha,
        focal_loss_gamma=float(config['loss']['gamma']),
        learning_rate=float(config['training']['learning_rate']),
        weight_decay=float(config['training']['weight_decay']),
        monitor_metric=config['training']['monitor_metric'],
        monitor_mode=config['training']['monitor_mode'],
    )
    
    # Setup callbacks
    callbacks = []
    
    # Create checkpoints directory in experiment folder
    checkpoint_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Model checkpoint - save in experiment folder
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='{epoch:02d}-{val_loss:.4f}',
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
    
    # Setup logger - save logs in experiment folder
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
        devices=1 if device.type == 'cuda' else None,
        gradient_clip_val=config['training']['gradient_clip_val'],
        accumulate_grad_batches=config['training']['accumulate_grad_batches'],
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=config['logging']['log_every_n_steps'],
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
        precision='16-mixed' if device.type == 'cuda' else '32',  # Use mixed precision on GPU
    )
    
    # Train model
    print("\n" + "=" * 50)
    print("Starting training...")
    print("=" * 50)
    
    trainer.fit(
        lightning_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    
    print("\n" + "=" * 50)
    print("Training completed!")
    print("=" * 50)
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")
    print(f"Best {config['training']['monitor_metric']}: {checkpoint_callback.best_model_score:.4f}")
    
    # Copy best model to experiment folder
    if checkpoint_callback.best_model_path:
        best_model_name = os.path.basename(checkpoint_callback.best_model_path)
        best_model_dest = os.path.join(exp_dir, "best_model.ckpt")
        shutil.copy(checkpoint_callback.best_model_path, best_model_dest)
        print(f"Best model copied to: {best_model_dest}")
    
    # Test on test set if available
    if test_loader:
        print("\n" + "=" * 50)
        print("Evaluating on test set...")
        print("=" * 50)
        
        # Load best model
        best_model = DynamicGNNLightning.load_from_checkpoint(
            best_model_path=checkpoint_callback.best_model_path,
            model=model,
        )
        
        trainer.test(best_model, dataloaders=test_loader)
        
        print("\nTest metrics:")
        for key, value in best_model.test_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # Save test results to experiment folder
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
            f.write("=" * 50 + "\n")
            for key, value in best_model.test_metrics.items():
                f.write(f"{key}: {value:.4f}\n")
        
        print(f"\nResults saved to: {results_dir}")
    
    # Save experiment summary
    summary_file = os.path.join(exp_dir, "experiment_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Experiment Start Time: {exp_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Experiment End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Config File: {args.config}\n")
        f.write(f"Random Seed: {args.seed}\n")
        f.write(f"Best Model: {checkpoint_callback.best_model_path}\n")
        f.write(f"Best {config['training']['monitor_metric']}: {checkpoint_callback.best_model_score:.4f}\n")
        f.write(f"\nModel Parameters: {num_params:,}\n")
        f.write(f"Focal Loss Alpha: {alpha:.4f}\n")
    
    print(f"\nExperiment summary saved to: {summary_file}")
    print(f"All experiment files saved in: {exp_dir}")


if __name__ == '__main__':
    main()

