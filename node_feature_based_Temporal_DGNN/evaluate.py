"""
Evaluation script for TemporalGNN.
Loads a trained checkpoint and evaluates on test set.
"""

import os
import yaml
import argparse
import torch
import torch.multiprocessing
import numpy as np
from datetime import datetime

# Fix for "too many open files" error with multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from dataset import create_dataloaders, TemporalGraphDataset
from model import TemporalGNN, TemporalGNNSimple
from lightning_module import TemporalGNNLightning
from loss import calculate_alpha_from_dataset
from utils import (
    set_seed,
    get_device,
    compute_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    print_classification_report,
    save_predictions,
    format_metrics,
)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def evaluate(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
) -> dict:
    """
    Evaluate model on a dataset.
    
    Args:
        model: Trained model
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
    
    Returns:
        Dictionary with predictions, probabilities, labels, and metrics
    """
    model.eval()
    model.to(device)
    
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (graphs_list, labels) in enumerate(dataloader):
            # Forward pass
            logits = model(graphs_list)
            
            # Compute predictions
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(dataloader)} batches...")
    
    # Concatenate results
    all_preds = torch.cat(all_preds).numpy()
    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    # Compute metrics
    metrics = compute_metrics(
        all_labels,
        all_preds,
        all_probs[:, 1],  # Probability of anomalous class
    )
    
    return {
        'predictions': all_preds,
        'probabilities': all_probs,
        'labels': all_labels,
        'metrics': metrics,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate TemporalGNN')
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
        '--output_dir',
        type=str,
        default=None,
        help='Directory to save evaluation results',
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['val', 'test'],
        help='Dataset split to evaluate on',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed',
    )
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup output directory
    if args.output_dir is None:
        # Create output dir next to checkpoint
        checkpoint_dir = os.path.dirname(args.checkpoint)
        args.output_dir = os.path.join(checkpoint_dir, f'eval_{args.split}')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("TemporalGNN Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.config}")
    print(f"Split: {args.split}")
    print(f"Output directory: {args.output_dir}")
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
    
    # Get device
    device = get_device()
    
    # Create dataloaders
    print("\nLoading dataset...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_root=config['data']['root_dir'],
        train_manifest=config['data']['train_manifest'],
        val_manifest=config['data']['val_manifest'],
        test_manifest=config['data'].get('test_manifest'),
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        pin_memory=False,  # Don't need pinned memory for evaluation
        normalize_features=config['data']['normalize_features'],
    )
    
    # Select dataloader based on split
    if args.split == 'test' and test_loader is not None:
        eval_loader = test_loader
        print(f"Evaluating on test set: {len(eval_loader.dataset)} samples")
    elif args.split == 'val':
        eval_loader = val_loader
        print(f"Evaluating on validation set: {len(eval_loader.dataset)} samples")
    else:
        print("Error: Test set not available. Using validation set.")
        eval_loader = val_loader
        args.split = 'val'
    
    # Create model
    print("\nCreating model...")
    if config['model']['type'].lower() == 'simple':
        model = TemporalGNNSimple()
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
    
    # Calculate alpha for creating the lightning module
    train_manifest_path = os.path.join(
        config['data']['root_dir'],
        config['data']['train_manifest']
    )
    temp_dataset = TemporalGraphDataset(
        manifest_path=train_manifest_path,
        normalize_features=False,
    )
    alpha = calculate_alpha_from_dataset(temp_dataset)
    
    # Load checkpoint
    print(f"\nLoading checkpoint from: {args.checkpoint}")
    lightning_module = TemporalGNNLightning.load_from_checkpoint(
        args.checkpoint,
        model=model,
        focal_loss_alpha=alpha,
        focal_loss_gamma=float(config['loss']['gamma']),
        weights_only=False,
    )
    
    # Evaluate
    print(f"\nRunning evaluation on {args.split} set...")
    results = evaluate(lightning_module.model, eval_loader, device)
    
    # Print metrics
    print("\n" + "=" * 60)
    print(f"Evaluation Results ({args.split} set)")
    print("=" * 60)
    print(format_metrics(results['metrics']))
    print("=" * 60)
    
    # Print classification report
    print("\nClassification Report:")
    print_classification_report(results['labels'], results['predictions'])
    
    # Save results
    print(f"\nSaving results to: {args.output_dir}")
    
    # Save metrics
    metrics_file = os.path.join(args.output_dir, 'metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write(f"TemporalGNN Evaluation Results\n")
        f.write(f"=" * 40 + "\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Split: {args.split}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"=" * 40 + "\n\n")
        f.write("Metrics:\n")
        f.write(format_metrics(results['metrics']))
    print(f"  Metrics saved to: {metrics_file}")
    
    # Save predictions
    predictions_file = os.path.join(args.output_dir, 'predictions.csv')
    save_predictions(
        results['predictions'],
        results['probabilities'],
        results['labels'],
        predictions_file,
    )
    
    # Plot confusion matrix
    cm_file = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(
        results['labels'],
        results['predictions'],
        save_path=cm_file,
    )
    
    # Plot normalized confusion matrix
    cm_norm_file = os.path.join(args.output_dir, 'confusion_matrix_normalized.png')
    plot_confusion_matrix(
        results['labels'],
        results['predictions'],
        save_path=cm_norm_file,
        normalize=True,
    )
    
    # Plot ROC curve (if we have both classes)
    if len(np.unique(results['labels'])) > 1:
        roc_file = os.path.join(args.output_dir, 'roc_curve.png')
        plot_roc_curve(
            results['labels'],
            results['probabilities'][:, 1],
            save_path=roc_file,
        )
        
        # Plot PR curve
        pr_file = os.path.join(args.output_dir, 'precision_recall_curve.png')
        plot_precision_recall_curve(
            results['labels'],
            results['probabilities'][:, 1],
            save_path=pr_file,
        )
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()

