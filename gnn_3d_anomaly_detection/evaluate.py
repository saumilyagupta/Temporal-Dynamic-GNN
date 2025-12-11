"""
Evaluation script for 3D Graph Convolutional Network for video anomaly detection.
"""

import os
import sys
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import GCN3D
from data import VideoGraphDataset, collate_fn
from config import (
    TEST_MANIFEST, MODEL_CONFIG, EVAL_CONFIG, TRAIN_CONFIG,
    CHECKPOINT_DIR, PREDICTION_DIR
)
from utils import (
    load_checkpoint, calculate_metrics, print_metrics,
    plot_confusion_matrix, plot_roc_curve, plot_pr_curve
)


def evaluate(model, dataloader, device, threshold=0.5):
    """
    Evaluate the model on a dataset.
    
    Returns:
        metrics: Dictionary of metrics
        all_preds: All predictions
        all_probs: All probabilities
        all_labels: All true labels
        video_paths: Video paths for each sample
    """
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    video_paths = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Extract video paths before moving to device
            batch_paths_list = []
            if hasattr(batch, 'video_path'):
                # PyG Batch stores lists/tuples as is
                if isinstance(batch.video_path, (list, tuple)):
                    batch_paths_list = batch.video_path
                else:
                    batch_paths_list = [batch.video_path]
            
            batch = batch.to(device)
            labels = batch.y.float()
            
            # Forward pass
            logits = model(batch).squeeze(-1)  # Squeeze last dimension only
            
            # Ensure shapes match for batch_size=1 case
            if logits.dim() == 0:
                logits = logits.unsqueeze(0)
            
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= threshold).astype(int)
            
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
            
            # Collect video paths
            if batch_paths_list:
                video_paths.extend(batch_paths_list)
            else:
                # Fallback: use batch size to create placeholder paths
                batch_size = len(preds) if hasattr(preds, '__len__') else 1
                video_paths.extend([f'video_{len(video_paths) + i}' for i in range(batch_size)])
    
    metrics = calculate_metrics(all_labels, all_preds, all_probs)
    
    return metrics, all_preds, all_probs, all_labels, video_paths


def main():
    """Main evaluation function."""
    print("=" * 80)
    print("3D GCN Video Anomaly Detection - Evaluation")
    print("=" * 80)
    
    # Configuration
    device = torch.device(TRAIN_CONFIG['device'])
    print(f"\nUsing device: {device}")
    
    # Load model checkpoint
    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'checkpoint_best.pth')
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(CHECKPOINT_DIR, 'checkpoint.pth')
        if not os.path.exists(checkpoint_path):
            print(f"\nError: No checkpoint found in {CHECKPOINT_DIR}")
            print("Please train the model first using train.py")
            return
    
    print(f"\nLoading checkpoint: {checkpoint_path}")
    
    # Create model
    model = GCN3D(**MODEL_CONFIG).to(device)
    epoch, loss = load_checkpoint(checkpoint_path, model)
    print(f"Loaded model from epoch {epoch}, loss: {loss:.4f}")
    
    # Create test dataset
    print("\nLoading test dataset...")
    test_dataset = VideoGraphDataset(
        TEST_MANIFEST,
        max_frames=TRAIN_CONFIG['max_frames'],
        normalize_coords=TRAIN_CONFIG['normalize_coords']
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=EVAL_CONFIG['batch_size'],
        shuffle=False,
        num_workers=TRAIN_CONFIG['num_workers'],
        pin_memory=TRAIN_CONFIG['pin_memory'],
        collate_fn=collate_fn
    )
    
    # Evaluate
    print("\n" + "=" * 80)
    print("Evaluating on test set...")
    print("=" * 80)
    
    metrics, all_preds, all_probs, all_labels, video_paths = evaluate(
        model, test_loader, device, threshold=EVAL_CONFIG['threshold']
    )
    
    # Print metrics
    print("\n" + "=" * 80)
    print("Test Set Results")
    print("=" * 80)
    print_metrics(metrics, 'Test')
    
    # Ensure video_paths matches length of labels
    if len(video_paths) != len(all_labels):
        video_paths = [f'video_{i}' for i in range(len(all_labels))]
    
    # Create predictions DataFrame
    predictions_df = pd.DataFrame({
        'video_path': video_paths,
        'true_label': ['Anomalous' if label == 1 else 'Normal' for label in all_labels],
        'predicted_label': ['Anomalous' if pred == 1 else 'Normal' for pred in all_preds],
        'probability': all_probs,
        'correct': [all_labels[i] == all_preds[i] for i in range(len(all_labels))]
    })
    
    # Save predictions
    if EVAL_CONFIG['save_predictions']:
        pred_path = os.path.join(PREDICTION_DIR, 'test_predictions.csv')
        predictions_df.to_csv(pred_path, index=False)
        print(f"\nPredictions saved to: {pred_path}")
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_path = os.path.join(PREDICTION_DIR, 'test_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to: {metrics_path}")
    
    # Generate visualizations
    if EVAL_CONFIG['save_plots']:
        print("\nGenerating visualizations...")
        
        # Confusion matrix
        cm_path = os.path.join(PREDICTION_DIR, 'confusion_matrix.png')
        plot_confusion_matrix(all_labels, all_preds, cm_path)
        print(f"Confusion matrix saved to: {cm_path}")
        
        # ROC curve
        roc_path = os.path.join(PREDICTION_DIR, 'roc_curve.png')
        roc_auc = plot_roc_curve(all_labels, all_probs, roc_path)
        print(f"ROC curve saved to: {roc_path} (AUC: {roc_auc:.4f})")
        
        # Precision-Recall curve
        pr_path = os.path.join(PREDICTION_DIR, 'pr_curve.png')
        pr_auc = plot_pr_curve(all_labels, all_probs, pr_path)
        print(f"PR curve saved to: {pr_path} (AUC: {pr_auc:.4f})")
    
    # Classification report
    print("\n" + "=" * 80)
    print("Classification Report")
    print("=" * 80)
    
    # Per-class metrics
    from sklearn.metrics import classification_report
    class_names = ['Normal', 'Anomalous']
    report = classification_report(
        all_labels, all_preds, 
        target_names=class_names,
        output_dict=True
    )
    
    print("\nPer-class metrics:")
    for class_name in class_names:
        if class_name in report:
            print(f"\n{class_name}:")
            print(f"  Precision: {report[class_name]['precision']:.4f}")
            print(f"  Recall:    {report[class_name]['recall']:.4f}")
            print(f"  F1-Score:  {report[class_name]['f1-score']:.4f}")
            print(f"  Support:   {report[class_name]['support']}")
    
    # Save classification report
    report_df = pd.DataFrame(report).transpose()
    report_path = os.path.join(PREDICTION_DIR, 'classification_report.csv')
    report_df.to_csv(report_path)
    print(f"\nClassification report saved to: {report_path}")
    
    # Summary
    print("\n" + "=" * 80)
    print("Evaluation Complete!")
    print("=" * 80)
    print(f"\nResults saved to: {PREDICTION_DIR}")
    print("\nFiles generated:")
    print(f"  - test_predictions.csv: Detailed predictions for each video")
    print(f"  - test_metrics.csv: Overall metrics")
    print(f"  - classification_report.csv: Per-class metrics")
    if EVAL_CONFIG['save_plots']:
        print(f"  - confusion_matrix.png: Confusion matrix visualization")
        print(f"  - roc_curve.png: ROC curve")
        print(f"  - pr_curve.png: Precision-Recall curve")
    print("=" * 80)


if __name__ == '__main__':
    main()

