"""
Utility functions for metrics, visualization, and data processing.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve,
)
from typing import Dict, List, Tuple, Optional
import os
import random


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        y_proba: Predicted probabilities for positive class (optional, for AUC)
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    
    # Calculate AUC if probabilities provided
    if y_proba is not None:
        try:
            # Check if we have both classes in y_true
            if len(np.unique(y_true)) > 1:
                metrics['auc_roc'] = roc_auc_score(y_true, y_proba)
            else:
                metrics['auc_roc'] = 0.0
        except ValueError:
            metrics['auc_roc'] = 0.0
    
    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = ['Normal', 'Anomalous'],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
    normalize: bool = False,
):
    """
    Plot and optionally save confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        save_path: Path to save the figure
        figsize: Figure size
        normalize: Whether to normalize confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
    else:
        fmt = 'd'
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        square=True,
    )
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
):
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities for positive class
        save_path: Path to save the figure
        figsize: Figure size
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
):
    """
    Plot Precision-Recall curve.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities for positive class
        save_path: Path to save the figure
        figsize: Figure size
    """
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    
    plt.figure(figsize=figsize)
    plt.plot(recall, precision, 'b-', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Precision-Recall curve saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = ['Normal', 'Anomalous'],
) -> str:
    """
    Print detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
    
    Returns:
        Classification report string
    """
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=False,
    )
    print(report)
    return report


def save_predictions(
    predictions: np.ndarray,
    probabilities: np.ndarray,
    true_labels: np.ndarray,
    save_path: str,
):
    """
    Save predictions and probabilities to a CSV file.
    
    Args:
        predictions: Predicted labels
        probabilities: Predicted probabilities [N, 2] or [N]
        true_labels: True labels
        save_path: Path to save the results
    """
    import pandas as pd
    
    if probabilities.ndim > 1:
        prob_normal = probabilities[:, 0]
        prob_anomalous = probabilities[:, 1]
    else:
        prob_anomalous = probabilities
        prob_normal = 1 - probabilities
    
    results = pd.DataFrame({
        'true_label': true_labels,
        'predicted_label': predictions,
        'prob_normal': prob_normal,
        'prob_anomalous': prob_anomalous,
    })
    
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    results.to_csv(save_path, index=False)
    print(f"Predictions saved to {save_path}")


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_metrics(metrics: Dict[str, float]) -> str:
    """
    Format metrics dictionary as a readable string.
    
    Args:
        metrics: Dictionary of metric names and values
    
    Returns:
        Formatted string
    """
    lines = []
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"{key}: {value:.4f}")
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


def get_device() -> torch.device:
    """
    Get the appropriate device (CUDA if available, else CPU).
    
    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


def save_model_summary(model: torch.nn.Module, save_path: str):
    """
    Save model architecture summary to a text file.
    
    Args:
        model: PyTorch model
        save_path: Path to save the summary
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    with open(save_path, 'w') as f:
        f.write("Model Architecture\n")
        f.write("=" * 50 + "\n\n")
        f.write(str(model) + "\n\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total Parameters: {count_parameters(model):,}\n")
    
    print(f"Model summary saved to {save_path}")



