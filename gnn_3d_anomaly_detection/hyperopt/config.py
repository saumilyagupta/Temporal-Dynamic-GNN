"""
Hyperparameter optimization configuration for 3D GCN.
Defines search spaces for Optuna trials.
"""

import os
import sys

# Import base config from parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRAIN_MANIFEST, VAL_MANIFEST, TEST_MANIFEST

# Hyperopt-specific paths
HYPEROPT_DIR = os.path.dirname(os.path.abspath(__file__))
STUDY_DB_DIR = os.path.join(HYPEROPT_DIR, 'study_db')
TRIALS_DIR = os.path.join(HYPEROPT_DIR, 'trials')
STUDY_DB_PATH = os.path.join(STUDY_DB_DIR, 'optuna_study.db')
STUDY_DB_URL = f'sqlite:///{STUDY_DB_PATH}'

# Create directories
os.makedirs(STUDY_DB_DIR, exist_ok=True)
os.makedirs(TRIALS_DIR, exist_ok=True)

# System-specific parameters (not optimized, use from base config)
SYSTEM_CONFIG = {
    'device': 'cuda:3',
    'num_workers': 4,
    'pin_memory': True,
    'max_frames': None,
    'normalize_coords': True,
    'num_epochs': 100,
    'log_interval': 10,
    'save_interval': 5,
    'plot_interval': 1,
}


def suggest_hyperparameters(trial):
    """
    Suggest hyperparameters for Optuna trial.
    
    Args:
        trial: Optuna trial object
        
    Returns:
        dict: Dictionary with model_config and train_config
    """
    # Model architecture hyperparameters
    hidden_dim = trial.suggest_int('hidden_dim', 32, 128, step=16)
    num_layers = trial.suggest_int('num_layers', 2, 5)
    dropout = trial.suggest_float('dropout', 0.1, 0.7, step=0.1)
    pooling_type = trial.suggest_categorical('pooling_type', 
                                             ['mean', 'max', 'sum', 'meanmax', 'all'])
    
    # Training hyperparameters
    batch_size = trial.suggest_int('batch_size', 1, 8, step=1)
    learning_rate = trial.suggest_float('learning_rate', 1e-7, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-7, 1e-4, log=True)
    patience = trial.suggest_int('patience', 5, 30, step=5)
    min_delta = trial.suggest_float('min_delta', 1e-5, 1e-2, log=True)
    
    # Build config dictionaries
    model_config = {
        'node_feat_dim': 3,  # Fixed by data structure
        'edge_feat_dim': 3,  # Fixed by data structure
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'dropout': dropout,
        'pooling_type': pooling_type,
    }
    
    train_config = {
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'patience': patience,
        'min_delta': min_delta,
        **SYSTEM_CONFIG  # Add system-specific config
    }
    
    return {
        'model_config': model_config,
        'train_config': train_config
    }


