"""
Evaluation script for trained Dynamic GNN model.
"""

import os
import yaml
import argparse
import torch
import pytorch_lightning as pl
from pathlib import Path
from datetime import datetime

from dataset import create_dataloaders
from model import DynamicGNN
from lightning_module import DynamicGNNLightning
from utils import (
    compute_metrics,
    plot_confusion_matrix,
    print_classification_report,
    save_predictions,
    format_metrics,
    get_device,
)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Evaluate Dynamic GNN')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint (.ckpt file)',
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file',
    )
    parser.add_argument(
        '--split',
        type=str,
        choices=['train', 'val', 'test'],
        default='test',
        help='Dataset split to evaluate on',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for results (default: from config)',
    )
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Determine output directory - try to use experiment folder if checkpoint is in one
    checkpoint_dir = os.path.dirname(os.path.dirname(args.checkpoint)) if os.path.dirname(args.checkpoint) else None
    if checkpoint_dir and 'exp_' in checkpoint_dir:
        # Checkpoint is in an experiment folder, use that for results
        exp_base = os.path.dirname(checkpoint_dir) if os.path.basename(checkpoint_dir).startswith('exp_') else checkpoint_dir
        exp_name = os.path.basename(checkpoint_dir) if os.path.basename(checkpoint_dir).startswith('exp_') else None
        if exp_name:
            output_dir = os.path.join(exp_base, exp_name, "results", args.split)
        else:
            output_dir = args.output_dir or config['evaluation']['output_dir']
    else:
        # Use default or specified output directory
        output_dir = args.output_dir or config['evaluation']['output_dir']
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create model (same architecture as training)
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
    
    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}...")
    lightning_module = DynamicGNNLightning.load_from_checkpoint(
        args.checkpoint,
        model=model,
        map_location=device,
    )
    lightning_module.eval()
    lightning_module.to(device)
    
    # Create dataloader for specified split
    print(f"\nCreating {args.split} dataloader...")
    from dataset import TemporalGraphDataset
    from torch.utils.data import DataLoader
    from dataset import collate_temporal_graphs
    
    if args.split == 'train':
        manifest = config['data']['train_manifest']
    elif args.split == 'val':
        manifest = config['data']['val_manifest']
    else:
        manifest = config['data'].get('test_manifest', config['data']['val_manifest'])
    
    dataset = TemporalGraphDataset(
        manifest_path=manifest,
        root_dir=config['data']['root_dir'],
        normalize_features=True,
    )
    
    eval_loader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        collate_fn=collate_temporal_graphs,
    )
    
    print(f"Evaluating on {args.split} set ({len(eval_loader)} batches)...")
    
    # Run evaluation
    trainer = pl.Trainer(
        accelerator='gpu' if device.type == 'cuda' else 'cpu',
        devices=1,
        logger=False,
    )
    
    trainer.test(lightning_module, dataloaders=eval_loader)
    
    # Get results
    y_true = lightning_module.test_labels
    y_pred = lightning_module.test_predictions
    y_proba = lightning_module.test_probabilities
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_proba[:, 1])
    
    # Print results
    print("\n" + "=" * 50)
    print(f"Evaluation Results ({args.split} set)")
    print("=" * 50)
    print(format_metrics(metrics))
    print("=" * 50)
    
    # Print classification report
    print("\nClassification Report:")
    print("-" * 50)
    print_classification_report(y_true, y_pred)
    
    # Save results
    print(f"\nSaving results to {output_dir}...")
    
    # Save predictions
    if config['evaluation']['save_predictions']:
        predictions_path = os.path.join(output_dir, f'{args.split}_predictions.csv')
        save_predictions(y_pred, y_proba, y_true, predictions_path)
    
    # Save confusion matrix
    if config['evaluation']['save_confusion_matrix']:
        cm_path = os.path.join(output_dir, f'{args.split}_confusion_matrix.png')
        plot_confusion_matrix(y_true, y_pred, save_path=cm_path)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, f'{args.split}_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"Evaluation Results ({args.split} set)\n")
        f.write("=" * 50 + "\n")
        f.write(format_metrics(metrics))
        f.write("\n" + "=" * 50 + "\n")
        f.write("\nClassification Report:\n")
        f.write("-" * 50 + "\n")
        f.write(print_classification_report(y_true, y_pred))
    
    print(f"Results saved to {output_dir}")
    print("\nEvaluation completed!")


if __name__ == '__main__':
    main()

