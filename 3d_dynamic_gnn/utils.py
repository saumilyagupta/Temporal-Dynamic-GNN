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
)
from typing import Dict, List, Tuple, Optional
import os


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
        y_proba: Predicted probabilities (optional, for AUC)
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    
    if y_proba is not None:
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics['auc_roc'] = 0.0
    
    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = ['Normal', 'Anomalous'],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
):
    """
    Plot and optionally save confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        save_path: Path to save the figure
        figsize: Figure size
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
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
    Save predictions and probabilities to a file.
    
    Args:
        predictions: Predicted labels
        probabilities: Predicted probabilities
        true_labels: True labels
        save_path: Path to save the results
    """
    import pandas as pd
    
    results = pd.DataFrame({
        'true_label': true_labels,
        'predicted_label': predictions,
        'prob_normal': probabilities[:, 0] if probabilities.ndim > 1 else 1 - probabilities,
        'prob_anomalous': probabilities[:, 1] if probabilities.ndim > 1 else probabilities,
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
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

