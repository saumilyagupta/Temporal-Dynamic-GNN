"""
Configuration parameters for 3D GCN training and evaluation.
"""

import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Data paths
DATA_ROOT = os.path.join(PROJECT_ROOT, 'NiAD_Large_Videos_processed_graphs')
TRAIN_MANIFEST = os.path.join(DATA_ROOT, 'train_manifest.csv')
VAL_MANIFEST = os.path.join(DATA_ROOT, 'val_manifest.csv')
TEST_MANIFEST = os.path.join(DATA_ROOT, 'test_manifest.csv')

# Results paths
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, 'checkpoints')
LOG_DIR = os.path.join(RESULTS_DIR, 'logs')
PREDICTION_DIR = os.path.join(RESULTS_DIR, 'predictions')

# Create directories if they don't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PREDICTION_DIR, exist_ok=True)

# Model architecture
MODEL_CONFIG = {
    'node_feat_dim': 3,  # Additional node features beyond 3D coords (width, height, class_id)
    'edge_feat_dim': 1,  # Edge features (distance only)
    'hidden_dim': 64,
    'num_layers': 3,
    'dropout': 0.5,
    'pooling_type': 'all',  # 'mean', 'max', 'sum', 'meanmax', 'all'
}

# Training configuration
TRAIN_CONFIG = {
    'batch_size': 1,
    'num_epochs': 100,
    'learning_rate': 0.0001,  # Increased from 1e-6 to 1e-4
    'weight_decay': 1e-6,
    'patience': 20,  # Early stopping patience
    'min_delta': 0.0001,  # Minimum change to qualify as improvement
    'pos_weight': 3.13,  # Weight for anomalous class (638 Normal / 204 Anomalous)
    # 'device': 'cuda' if __import__('torch').cuda.is_available() else 'cpu',
    'device': 'cuda:3',
    'num_workers': 4,  # DataLoader workers
    'pin_memory': True,
    'max_frames': None,  # None for all frames, or int to limit
    'normalize_coords': True,  # Normalize node coordinates
}

# Evaluation configuration
EVAL_CONFIG = {
    'batch_size': 1,
    'threshold': 0.3,  # Lowered threshold to improve anomaly recall
    'save_predictions': True,
    'save_plots': True,
}

# Logging
LOG_CONFIG = {
    'log_interval': 10,  # Log every N batches
    'save_interval': 5,  # Save checkpoint every N epochs
    'plot_interval': 1,  # Plot metrics every N epochs
}


