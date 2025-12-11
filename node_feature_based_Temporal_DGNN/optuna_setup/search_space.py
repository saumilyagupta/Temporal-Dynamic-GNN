"""
Search space definitions for Optuna hyperparameter optimization.
Provides functions to sample hyperparameters from defined ranges.
"""

import optuna
from typing import Dict, Any
import yaml


def load_search_space_config(config_path: str = "optuna_config.yaml") -> Dict:
    """Load search space configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config.get('search_space', {})


def suggest_model_params(trial: optuna.Trial, config: Dict = None) -> Dict[str, Any]:
    """
    Suggest model architecture hyperparameters.
    
    Args:
        trial: Optuna trial object
        config: Optional config dict (if None, uses defaults)
    
    Returns:
        Dictionary of model hyperparameters
    """
    if config is None:
        config = {}
    
    model_config = config.get('model', {})
    
    # Hidden dimension for GCN layers
    hidden_dim_cfg = model_config.get('hidden_dim', {'choices': [32, 64, 128, 256]})
    hidden_dim = trial.suggest_categorical('hidden_dim', hidden_dim_cfg['choices'])
    
    # Temporal dimension for GRU
    temporal_dim_cfg = model_config.get('temporal_dim', {'choices': [64, 128, 256]})
    temporal_dim = trial.suggest_categorical('temporal_dim', temporal_dim_cfg['choices'])
    
    # Number of GCN layers
    gcn_cfg = model_config.get('num_gcn_layers', {'low': 1, 'high': 4})
    num_gcn_layers = trial.suggest_int('num_gcn_layers', int(gcn_cfg['low']), int(gcn_cfg['high']))
    
    # Number of GRU layers
    gru_cfg = model_config.get('num_gru_layers', {'low': 1, 'high': 3})
    num_gru_layers = trial.suggest_int('num_gru_layers', int(gru_cfg['low']), int(gru_cfg['high']))
    
    # Dropout rate
    dropout_cfg = model_config.get('dropout', {'low': 0.1, 'high': 0.5, 'step': 0.05})
    dropout = trial.suggest_float(
        'dropout',
        float(dropout_cfg['low']),
        float(dropout_cfg['high']),
        step=float(dropout_cfg.get('step', 0.05))
    )
    
    return {
        'hidden_dim': hidden_dim,
        'temporal_dim': temporal_dim,
        'num_gcn_layers': num_gcn_layers,
        'num_gru_layers': num_gru_layers,
        'dropout': dropout,
        'input_dim': 8,  # Fixed: node feature dimension
        'num_classes': 2,  # Fixed: binary classification
    }


def suggest_training_params(trial: optuna.Trial, config: Dict = None) -> Dict[str, Any]:
    """
    Suggest training hyperparameters.
    
    Args:
        trial: Optuna trial object
        config: Optional config dict
    
    Returns:
        Dictionary of training hyperparameters
    """
    if config is None:
        config = {}
    
    training_config = config.get('training', {})
    
    # Learning rate (log-uniform)
    lr_cfg = training_config.get('learning_rate', {'low': 1e-5, 'high': 1e-2})
    lr_low = float(lr_cfg['low'])
    lr_high = float(lr_cfg['high'])
    learning_rate = trial.suggest_float(
        'learning_rate',
        lr_low,
        lr_high,
        log=True
    )
    
    # Weight decay (log-uniform)
    wd_cfg = training_config.get('weight_decay', {'low': 1e-6, 'high': 1e-3})
    wd_low = float(wd_cfg['low'])
    wd_high = float(wd_cfg['high'])
    weight_decay = trial.suggest_float(
        'weight_decay',
        wd_low,
        wd_high,
        log=True
    )
    
    # Batch size
    bs_cfg = training_config.get('batch_size', {'choices': [4, 8, 16]})
    batch_size = trial.suggest_categorical('batch_size', bs_cfg['choices'])
    
    # Optimizer
    opt_cfg = training_config.get('optimizer', {'choices': ['adam', 'adamw']})
    optimizer = trial.suggest_categorical('optimizer', opt_cfg['choices'])
    
    return {
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'batch_size': batch_size,
        'optimizer': optimizer,
    }


def suggest_loss_params(trial: optuna.Trial, base_alpha: float, config: Dict = None) -> Dict[str, Any]:
    """
    Suggest loss function hyperparameters.
    
    Args:
        trial: Optuna trial object
        base_alpha: Base alpha computed from class distribution
        config: Optional config dict
    
    Returns:
        Dictionary of loss hyperparameters
    """
    if config is None:
        config = {}
    
    loss_config = config.get('loss', {})
    
    # Focal loss gamma
    gamma_cfg = loss_config.get('focal_gamma', {'low': 0.5, 'high': 3.0, 'step': 0.25})
    focal_gamma = trial.suggest_float(
        'focal_gamma',
        float(gamma_cfg['low']),
        float(gamma_cfg['high']),
        step=float(gamma_cfg.get('step', 0.25))
    )
    
    # Alpha multiplier (to tune around the class-balanced alpha)
    alpha_mult_cfg = loss_config.get('alpha_multiplier', {'low': 0.8, 'high': 1.2, 'step': 0.05})
    alpha_multiplier = trial.suggest_float(
        'alpha_multiplier',
        float(alpha_mult_cfg['low']),
        float(alpha_mult_cfg['high']),
        step=float(alpha_mult_cfg.get('step', 0.05))
    )
    
    # Compute final alpha
    focal_alpha = base_alpha * alpha_multiplier
    # Clamp to valid range
    focal_alpha = max(0.1, min(0.9, focal_alpha))
    
    return {
        'focal_gamma': focal_gamma,
        'focal_alpha': focal_alpha,
        'alpha_multiplier': alpha_multiplier,
    }


def suggest_threshold(trial: optuna.Trial, config: Dict = None) -> float:
    """
    Suggest classification threshold for precision/recall trade-off.
    
    Args:
        trial: Optuna trial object
        config: Optional config dict
    
    Returns:
        Classification threshold
    """
    if config is None:
        config = {}
    
    threshold_cfg = config.get('threshold', {'low': 0.3, 'high': 0.7, 'step': 0.05})
    
    threshold = trial.suggest_float(
        'threshold',
        float(threshold_cfg['low']),
        float(threshold_cfg['high']),
        step=float(threshold_cfg.get('step', 0.05))
    )
    
    return threshold


def suggest_all_hyperparameters(
    trial: optuna.Trial,
    base_alpha: float,
    config: Dict = None
) -> Dict[str, Any]:
    """
    Suggest all hyperparameters for a trial.
    
    Args:
        trial: Optuna trial object
        base_alpha: Base alpha from class distribution
        config: Search space configuration dict
    
    Returns:
        Complete hyperparameter dictionary
    """
    model_params = suggest_model_params(trial, config)
    training_params = suggest_training_params(trial, config)
    loss_params = suggest_loss_params(trial, base_alpha, config)
    threshold = suggest_threshold(trial, config)
    
    return {
        'model': model_params,
        'training': training_params,
        'loss': loss_params,
        'threshold': threshold,
    }


def get_default_hyperparameters() -> Dict[str, Any]:
    """
    Get default hyperparameters (baseline configuration).
    
    Returns:
        Default hyperparameter dictionary
    """
    return {
        'model': {
            'input_dim': 8,
            'hidden_dim': 64,
            'temporal_dim': 128,
            'num_gcn_layers': 2,
            'num_gru_layers': 1,
            'dropout': 0.3,
            'num_classes': 2,
        },
        'training': {
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'batch_size': 8,
            'optimizer': 'adam',
        },
        'loss': {
            'focal_gamma': 1.0,
            'focal_alpha': 0.76,
            'alpha_multiplier': 1.0,
        },
        'threshold': 0.5,
    }


def hyperparams_to_flat_dict(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten nested hyperparameter dict for logging.
    
    Args:
        params: Nested hyperparameter dictionary
    
    Returns:
        Flattened dictionary with prefixed keys
    """
    flat = {}
    for category, values in params.items():
        if isinstance(values, dict):
            for key, value in values.items():
                flat[f"{category}_{key}"] = value
        else:
            flat[category] = values
    return flat

