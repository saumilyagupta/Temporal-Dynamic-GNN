#!/usr/bin/env python3
"""
Utility to export best Optuna hyperparameters to a config.yaml file
compatible with the main training script.
"""

import os
import sys
import argparse
import json
import yaml
import optuna


def load_best_trial_from_study(study_name: str, storage: str) -> dict:
    """
    Load the best trial from an Optuna study.
    
    Args:
        study_name: Name of the study
        storage: Storage URL (e.g., sqlite:///study.db)
    
    Returns:
        Dictionary with best trial info
    """
    study = optuna.load_study(
        study_name=study_name,
        storage=storage,
    )
    
    if study.best_trial is None:
        raise ValueError(f"No completed trials found in study: {study_name}")
    
    return {
        'trial_number': study.best_trial.number,
        'value': study.best_trial.value,
        'params': study.best_trial.params,
        'user_attrs': study.best_trial.user_attrs,
    }


def load_best_trial_from_json(json_path: str) -> dict:
    """
    Load best trial info from a JSON file.
    
    Args:
        json_path: Path to trial JSON file
    
    Returns:
        Dictionary with trial info
    """
    with open(json_path, 'r') as f:
        return json.load(f)


def convert_params_to_config(params: dict, user_attrs: dict = None) -> dict:
    """
    Convert Optuna trial parameters to training config format.
    
    Args:
        params: Trial parameters from Optuna
        user_attrs: User attributes from trial
    
    Returns:
        Config dictionary compatible with train.py
    """
    # Extract parameters
    hidden_dim = params.get('hidden_dim', 64)
    temporal_dim = params.get('temporal_dim', 128)
    num_gcn_layers = params.get('num_gcn_layers', 2)
    num_gru_layers = params.get('num_gru_layers', 1)
    dropout = params.get('dropout', 0.3)
    
    learning_rate = params.get('learning_rate', 0.001)
    weight_decay = params.get('weight_decay', 1e-5)
    batch_size = params.get('batch_size', 8)
    optimizer = params.get('optimizer', 'adam')
    
    focal_gamma = params.get('focal_gamma', 1.0)
    threshold = params.get('threshold', 0.5)
    
    # Build config
    config = {
        'data': {
            'root_dir': '/workspace/saumilya/GNN-Research/NiAD_graphs_node_feature_based',
            'train_manifest': 'train_manifest.csv',
            'val_manifest': 'val_manifest.csv',
            'test_manifest': 'test_manifest.csv',
            'num_workers': 0,
            'pin_memory': True,
            'normalize_features': True,
        },
        'model': {
            'type': 'full',  # Use configurable TemporalGNN
            'input_dim': 8,
            'hidden_dim': hidden_dim,
            'temporal_dim': temporal_dim,
            'num_gcn_layers': num_gcn_layers,
            'num_gru_layers': num_gru_layers,
            'dropout': dropout,
            'num_classes': 2,
        },
        'training': {
            'gpu_id': 4,
            'batch_size': batch_size,
            'max_epochs': 100,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'gradient_clip_val': 1.0,
            'accumulate_grad_batches': 1,
            'optimizer': optimizer,
            'scheduler': 'plateau',
            'monitor_metric': 'val_f1',
            'monitor_mode': 'max',
        },
        'loss': {
            'type': 'focal',
            'gamma': focal_gamma,
        },
        'callbacks': {
            'checkpoint': {
                'save_top_k': 3,
                'save_last': True,
                'every_n_epochs': 1,
            },
            'early_stopping': {
                'patience': 15,
                'min_delta': 0.0001,
            },
        },
        'logging': {
            'log_every_n_steps': 10,
            'experiment_name': 'temporal_gnn_optimized',
        },
        # Store optimized threshold for evaluation
        '_optuna_info': {
            'threshold': threshold,
            'source': 'optuna_optimization',
        },
    }
    
    # Add performance info if available
    if user_attrs:
        config['_optuna_info']['best_f1'] = user_attrs.get('f1')
        config['_optuna_info']['best_precision'] = user_attrs.get('precision')
        config['_optuna_info']['best_recall'] = user_attrs.get('recall')
    
    return config


def export_config(
    config: dict,
    output_path: str,
    include_comments: bool = True,
):
    """
    Export configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        output_path: Output file path
        include_comments: Whether to add header comments
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Add header comment
    header = """# TemporalGNN Configuration - Optimized by Optuna
# This configuration was generated from the best Optuna trial
# for improved F1 score and precision on anomaly detection.
#
# To use: python train.py --config <this_file>
#
"""
    
    with open(output_path, 'w') as f:
        if include_comments:
            f.write(header)
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Configuration exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Export best Optuna hyperparameters to config.yaml'
    )
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Source: either a JSON file path or "study:<study_name>:<storage>"',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='best_config.yaml',
        help='Output config file path',
    )
    parser.add_argument(
        '--gpu_id',
        type=int,
        default=4,
        help='GPU ID to set in config',
    )
    args = parser.parse_args()
    
    # Load best trial
    if args.source.startswith('study:'):
        # Format: study:<study_name>:<storage>
        parts = args.source.split(':')
        if len(parts) < 3:
            raise ValueError("Study source format: study:<study_name>:<storage>")
        study_name = parts[1]
        storage = ':'.join(parts[2:])  # Handle sqlite:// prefix
        
        print(f"Loading from study: {study_name}")
        trial_info = load_best_trial_from_study(study_name, storage)
    else:
        # Load from JSON file
        print(f"Loading from file: {args.source}")
        trial_info = load_best_trial_from_json(args.source)
    
    # Print trial info
    print("\n" + "=" * 50)
    print("Best Trial Information")
    print("=" * 50)
    print(f"Trial number: {trial_info.get('trial_number', 'N/A')}")
    print(f"Objective value: {trial_info.get('value', 'N/A')}")
    
    if 'user_attrs' in trial_info:
        attrs = trial_info['user_attrs']
        if 'f1' in attrs:
            print(f"F1 Score: {attrs['f1']:.4f}")
        if 'precision' in attrs:
            print(f"Precision: {attrs['precision']:.4f}")
        if 'recall' in attrs:
            print(f"Recall: {attrs['recall']:.4f}")
    
    print("\nHyperparameters:")
    for key, value in trial_info.get('params', {}).items():
        print(f"  {key}: {value}")
    print("=" * 50 + "\n")
    
    # Convert to config
    config = convert_params_to_config(
        trial_info.get('params', {}),
        trial_info.get('user_attrs', {}),
    )
    
    # Override GPU ID
    config['training']['gpu_id'] = args.gpu_id
    
    # Export
    export_config(config, args.output)
    
    print("\nTo train with optimized config:")
    print(f"  python train.py --config {args.output}")


if __name__ == '__main__':
    main()


