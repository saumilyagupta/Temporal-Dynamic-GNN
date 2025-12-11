#!/usr/bin/env python3
"""
Main script to run Optuna hyperparameter optimization for TemporalGNN.
Optimizes for F1 score and precision on anomaly detection task.
"""

import os
import sys
import argparse
import yaml
import json
from datetime import datetime
import optuna
from optuna.samplers import TPESampler, RandomSampler
from optuna.pruners import MedianPruner, HyperbandPruner

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from objective import create_objective, run_single_trial
from search_space import get_default_hyperparameters


def create_sampler(config: dict) -> optuna.samplers.BaseSampler:
    """Create sampler based on configuration."""
    sampler_config = config.get('sampler', {})
    sampler_type = sampler_config.get('type', 'tpe').lower()
    seed = sampler_config.get('seed', 42)
    
    if sampler_type == 'tpe':
        return TPESampler(
            seed=seed,
            n_startup_trials=sampler_config.get('n_startup_trials', 10),
        )
    elif sampler_type == 'random':
        return RandomSampler(seed=seed)
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")


def create_pruner(config: dict) -> optuna.pruners.BasePruner:
    """Create pruner based on configuration."""
    pruner_config = config.get('pruner', {})
    pruner_type = pruner_config.get('type', 'median').lower()
    
    if pruner_type == 'median':
        return MedianPruner(
            n_startup_trials=pruner_config.get('n_startup_trials', 5),
            n_warmup_steps=pruner_config.get('n_warmup_steps', 10),
            interval_steps=pruner_config.get('interval_steps', 1),
        )
    elif pruner_type == 'hyperband':
        return HyperbandPruner(
            min_resource=pruner_config.get('min_resource', 1),
            max_resource=pruner_config.get('max_resource', 50),
            reduction_factor=pruner_config.get('reduction_factor', 3),
        )
    elif pruner_type == 'none':
        return optuna.pruners.NopPruner()
    else:
        raise ValueError(f"Unknown pruner type: {pruner_type}")


def save_trial_result(trial: optuna.Trial, output_dir: str):
    """Save a single trial's result to file (called after each trial)."""
    os.makedirs(output_dir, exist_ok=True)
    
    trial_info = {
        'trial_number': trial.number,
        'state': str(trial.state),
        'value': trial.value,
        'params': trial.params,
        'user_attrs': trial.user_attrs,
        'datetime_start': str(trial.datetime_start) if trial.datetime_start else None,
        'datetime_complete': str(trial.datetime_complete) if trial.datetime_complete else None,
    }
    
    filepath = os.path.join(output_dir, f'trial_{trial.number:04d}.json')
    with open(filepath, 'w') as f:
        json.dump(trial_info, f, indent=2)


def save_best_trials(study: optuna.Study, output_dir: str, n_best: int = 5):
    """Save the best trials to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get best trials
    trials = study.best_trials if hasattr(study, 'best_trials') else [study.best_trial]
    
    # Sort by value (descending for maximization)
    sorted_trials = sorted(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE],
        key=lambda t: t.value if t.value is not None else float('-inf'),
        reverse=True
    )[:n_best]
    
    # Save each trial
    for i, trial in enumerate(sorted_trials):
        trial_info = {
            'rank': i + 1,
            'trial_number': trial.number,
            'value': trial.value,
            'params': trial.params,
            'user_attrs': trial.user_attrs,
            'datetime_start': str(trial.datetime_start) if trial.datetime_start else None,
            'datetime_complete': str(trial.datetime_complete) if trial.datetime_complete else None,
        }
        
        filepath = os.path.join(output_dir, f'trial_rank_{i+1}.json')
        with open(filepath, 'w') as f:
            json.dump(trial_info, f, indent=2)
    
    print(f"Saved top {len(sorted_trials)} trials to {output_dir}")


def evaluate_best_trial_on_test(
    study: optuna.Study,
    config: dict,
    output_dir: str,
    gpu_id: int = None,
):
    """
    Evaluate the best trial on the test set.
    
    Args:
        study: Completed Optuna study
        config: Configuration dictionary
        output_dir: Directory to save results
        gpu_id: GPU ID to use
    """
    if study.best_trial is None:
        print("No completed trials to evaluate.")
        return None
    
    print("\n" + "=" * 60)
    print("EVALUATING BEST TRIAL ON TEST SET")
    print("=" * 60)
    
    # Get best trial parameters
    best_params = study.best_trial.params
    
    # Convert flat params to nested structure
    params = {
        'model': {
            'input_dim': 8,
            'hidden_dim': best_params.get('hidden_dim', 64),
            'temporal_dim': best_params.get('temporal_dim', 128),
            'num_gcn_layers': best_params.get('num_gcn_layers', 2),
            'num_gru_layers': best_params.get('num_gru_layers', 1),
            'dropout': best_params.get('dropout', 0.3),
            'num_classes': 2,
        },
        'training': {
            'learning_rate': best_params.get('learning_rate', 0.001),
            'weight_decay': best_params.get('weight_decay', 1e-5),
            'batch_size': best_params.get('batch_size', 8),
            'optimizer': best_params.get('optimizer', 'adam'),
        },
        'loss': {
            'focal_gamma': best_params.get('focal_gamma', 1.0),
            'focal_alpha': best_params.get('focal_alpha', 0.76),
        },
        'threshold': best_params.get('threshold', 0.5),
    }
    
    # Calculate focal_alpha if we have alpha_multiplier
    if 'alpha_multiplier' in best_params:
        base_alpha = 0.76  # Default from class distribution
        params['loss']['focal_alpha'] = base_alpha * best_params['alpha_multiplier']
    
    print(f"Best trial #{study.best_trial.number}")
    print(f"Validation F1: {study.best_trial.value:.4f}")
    print(f"Parameters: {best_params}")
    
    # Run evaluation with full training and test set evaluation
    try:
        save_path = os.path.join(output_dir, 'best_model.ckpt')
        test_metrics = run_single_trial(
            params=params,
            config_path='optuna_config.yaml',
            save_path=save_path,
        )
        
        # Save test results
        test_results = {
            'best_trial_number': study.best_trial.number,
            'validation_f1': study.best_trial.value,
            'validation_metrics': study.best_trial.user_attrs,
            'test_metrics': test_metrics,
            'best_params': best_params,
            'threshold': params['threshold'],
        }
        
        results_path = os.path.join(output_dir, 'best_trial_test_results.json')
        with open(results_path, 'w') as f:
            json.dump(test_metrics, f, indent=2)
        
        print("\n" + "-" * 40)
        print("TEST SET RESULTS")
        print("-" * 40)
        for key, value in test_metrics.items():
            print(f"  {key}: {value:.4f}")
        print("-" * 40)
        print(f"Results saved to: {results_path}")
        
        return test_metrics
        
    except Exception as e:
        print(f"Error evaluating best trial: {e}")
        return None


def print_study_summary(study: optuna.Study):
    """Print summary of the optimization study."""
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    
    print(f"\nStudy name: {study.study_name}")
    print(f"Total trials: {len(study.trials)}")
    
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    
    print(f"Completed: {len(completed)}")
    print(f"Pruned: {len(pruned)}")
    print(f"Failed: {len(failed)}")
    
    if study.best_trial:
        print("\n" + "-" * 40)
        print("BEST TRIAL")
        print("-" * 40)
        print(f"Trial number: {study.best_trial.number}")
        print(f"Value (F1): {study.best_trial.value:.4f}")
        
        if 'precision' in study.best_trial.user_attrs:
            print(f"Precision: {study.best_trial.user_attrs['precision']:.4f}")
        if 'recall' in study.best_trial.user_attrs:
            print(f"Recall: {study.best_trial.user_attrs['recall']:.4f}")
        
        print("\nBest hyperparameters:")
        for key, value in study.best_trial.params.items():
            print(f"  {key}: {value}")
    
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Run Optuna HPO for TemporalGNN')
    parser.add_argument(
        '--config',
        type=str,
        default='optuna_config.yaml',
        help='Path to Optuna configuration file',
    )
    parser.add_argument(
        '--n_trials',
        type=int,
        default=None,
        help='Number of trials (overrides config)',
    )
    parser.add_argument(
        '--timeout',
        type=float,
        default=None,
        help='Timeout in hours (overrides config)',
    )
    parser.add_argument(
        '--gpu_id',
        type=int,
        default=None,
        help='GPU ID to use (overrides config)',
    )
    parser.add_argument(
        '--study_name',
        type=str,
        default=None,
        help='Custom study name (overrides config)',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='optuna_results',
        help='Directory to save results',
    )
    parser.add_argument(
        '--n_jobs',
        type=int,
        default=1,
        help='Number of parallel jobs (for distributed optimization)',
    )
    parser.add_argument(
        '--skip_test_eval',
        action='store_true',
        help='Skip final evaluation on test set',
    )
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    study_config = config['study']
    
    # Override settings from command line
    n_trials = args.n_trials or study_config.get('n_trials', 100)
    timeout = args.timeout * 3600 if args.timeout else study_config.get('timeout')
    study_name = args.study_name or study_config.get('name', 'temporal_gnn_hpo')
    
    # Add timestamp to study name if creating new
    if not study_config.get('load_if_exists', False):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        study_name = f"{study_name}_{timestamp}"
    
    # Setup storage
    storage = study_config.get('storage', f'sqlite:///{study_name}.db')
    
    print("\n" + "=" * 60)
    print("OPTUNA HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)
    print(f"Study name: {study_name}")
    print(f"Storage: {storage}")
    print(f"Number of trials: {n_trials}")
    print(f"Timeout: {timeout/3600:.1f}h" if timeout else "Timeout: None")
    print(f"Direction: {study_config.get('direction', 'maximize')}")
    print("=" * 60 + "\n")
    
    # Create sampler and pruner
    sampler = create_sampler(study_config)
    pruner = create_pruner(study_config)
    
    # Create or load study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=study_config.get('direction', 'maximize'),
        sampler=sampler,
        pruner=pruner,
        load_if_exists=study_config.get('load_if_exists', True),
    )
    
    # Create objective function
    objective = create_objective(
        config_path=args.config,
        gpu_id=args.gpu_id,
    )
    
    # Setup output directory for trial results
    output_dir = os.path.join(args.output_dir, study_name)
    trials_dir = os.path.join(output_dir, 'all_trials')
    os.makedirs(trials_dir, exist_ok=True)
    
    # Callback to save each trial result
    def trial_callback(study: optuna.Study, trial: optuna.Trial):
        save_trial_result(trial, trials_dir)
        print(f"Trial {trial.number} saved. Value: {trial.value}")
    
    # Run optimization
    try:
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=args.n_jobs,
            show_progress_bar=True,
            gc_after_trial=True,  # Garbage collect to free memory
            callbacks=[trial_callback],  # Save each trial
        )
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
    
    # Print summary
    print_study_summary(study)
    
    # Save best trials
    save_best_trials(study, output_dir, n_best=config['logging'].get('save_best_n', 5))
    
    # Save study summary
    summary_path = os.path.join(output_dir, 'study_summary.json')
    summary = {
        'study_name': study_name,
        'direction': study_config.get('direction', 'maximize'),
        'n_trials_completed': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        'n_trials_pruned': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        'best_value': study.best_trial.value if study.best_trial else None,
        'best_params': study.best_trial.params if study.best_trial else None,
        'best_user_attrs': study.best_trial.user_attrs if study.best_trial else None,
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Evaluate best trial on test set (unless skipped)
    test_metrics = None
    if not args.skip_test_eval:
        test_metrics = evaluate_best_trial_on_test(
            study=study,
            config=config,
            output_dir=output_dir,
            gpu_id=args.gpu_id,
        )
        
        # Update summary with test results
        if test_metrics:
            summary['test_metrics'] = test_metrics
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
    else:
        print("\nSkipping test set evaluation (--skip_test_eval flag set)")
    
    print(f"\nAll results saved to: {output_dir}")
    
    return study


if __name__ == '__main__':
    main()

