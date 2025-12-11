"""
Optuna hyperparameter optimization for 3D GCN.
"""

import os
import sys
import json
import optuna
from datetime import datetime
from optuna.trial import TrialState

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hyperopt.config import (
    suggest_hyperparameters, STUDY_DB_URL, TRIALS_DIR, 
    SYSTEM_CONFIG
)
from hyperopt.train_trial import train_with_hyperparameters

# Study configuration
STUDY_NAME = 'gcn3d_hyperopt'
N_TRIALS = 100
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'optimization_log.txt')


def log_message(message, verbose=True):
    """Log message to file and optionally print."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {message}\n"
    
    with open(LOG_FILE, 'a') as f:
        f.write(log_entry)
    
    if verbose:
        print(message)


def objective(trial):
    """
    Optuna objective function.
    
    Args:
        trial: Optuna trial object
        
    Returns:
        float: Validation accuracy (to maximize)
    """
    trial_num = trial.number
    
    # Suggest hyperparameters
    configs = suggest_hyperparameters(trial)
    model_config = configs['model_config']
    train_config = configs['train_config']
    
    # Create trial directory
    trial_dir = os.path.join(TRIALS_DIR, f'trial_{trial_num}')
    os.makedirs(trial_dir, exist_ok=True)
    
    # Log trial start
    log_message(f"Trial {trial_num} started")
    log_message(f"  Parameters: {json.dumps(trial.params, indent=2)}")
    
    try:
        # Train model with suggested hyperparameters
        val_accuracy = train_with_hyperparameters(
            model_config=model_config,
            train_config=train_config,
            trial_dir=trial_dir,
            verbose=False
        )
        
        # Log trial completion
        log_message(f"Trial {trial_num} completed: val_acc={val_accuracy:.4f}")
        
        # Save trial info to JSON
        trial_info = {
            'trial_number': trial_num,
            'params': trial.params,
            'value': val_accuracy,
            'state': 'COMPLETE',
            'datetime': datetime.now().isoformat(),
            'model_config': model_config,
            'train_config': {k: v for k, v in train_config.items() if k not in SYSTEM_CONFIG}
        }
        
        trial_info_path = os.path.join(trial_dir, 'trial_info.json')
        with open(trial_info_path, 'w') as f:
            json.dump(trial_info, f, indent=2)
        
        return val_accuracy
        
    except Exception as e:
        # Log trial failure
        error_msg = f"Trial {trial_num} failed: {str(e)}"
        log_message(error_msg)
        
        # Save failure info
        trial_info = {
            'trial_number': trial_num,
            'params': trial.params,
            'state': 'FAIL',
            'error': str(e),
            'datetime': datetime.now().isoformat()
        }
        
        trial_info_path = os.path.join(trial_dir, 'trial_info.json')
        with open(trial_info_path, 'w') as f:
            json.dump(trial_info, f, indent=2)
        
        # Re-raise to mark trial as failed in Optuna
        raise


def load_or_create_study():
    """Load existing study or create new one."""
    try:
        study = optuna.load_study(
            study_name=STUDY_NAME,
            storage=STUDY_DB_URL
        )
        log_message(f"Loaded existing study '{STUDY_NAME}' with {len(study.trials)} completed trials")
        return study
    except (ValueError, KeyError):
        # Study doesn't exist, create new one
        study = optuna.create_study(
            study_name=STUDY_NAME,
            storage=STUDY_DB_URL,
            direction='maximize',  # Maximize validation accuracy
            load_if_exists=False
        )
        log_message(f"Created new study '{STUDY_NAME}'")
        return study


def save_best_config(study):
    """Save best trial configuration to JSON file."""
    if len(study.trials) == 0:
        log_message("No trials completed yet. Cannot save best config.")
        return
    
    # Get best trial
    best_trial = study.best_trial
    
    # Get best trial info if available
    best_trial_dir = os.path.join(TRIALS_DIR, f'trial_{best_trial.number}')
    best_trial_info_path = os.path.join(best_trial_dir, 'trial_info.json')
    
    model_config = None
    train_config = None
    
    if os.path.exists(best_trial_info_path):
        with open(best_trial_info_path, 'r') as f:
            trial_info = json.load(f)
            model_config = trial_info.get('model_config')
            train_config = trial_info.get('train_config')
    
    # If not found in trial info, reconstruct from params
    if model_config is None or train_config is None:
        # Create configs from best_trial.params
        params = best_trial.params
        
        model_config = {
            'node_feat_dim': 3,
            'edge_feat_dim': 3,
            'hidden_dim': params['hidden_dim'],
            'num_layers': params['num_layers'],
            'dropout': params['dropout'],
            'pooling_type': params['pooling_type'],
        }
        
        train_config = {
            'batch_size': params['batch_size'],
            'learning_rate': params['learning_rate'],
            'weight_decay': params['weight_decay'],
            'patience': params['patience'],
            'min_delta': params['min_delta'],
            **SYSTEM_CONFIG
        }
    
    # Create best config dictionary
    best_config = {
        'best_trial_number': best_trial.number,
        'best_validation_accuracy': best_trial.value,
        'datetime': datetime.now().isoformat(),
        'total_trials': len(study.trials),
        'completed_trials': len([t for t in study.trials if t.state == TrialState.COMPLETE]),
        'model_config': model_config,
        'train_config': {k: v for k, v in train_config.items() if k not in SYSTEM_CONFIG},
        'system_config': SYSTEM_CONFIG,
        'all_params': best_trial.params
    }
    
    # Save to JSON
    best_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_config.json')
    with open(best_config_path, 'w') as f:
        json.dump(best_config, f, indent=2)
    
    log_message(f"Best config saved to {best_config_path}")
    log_message(f"Best trial: {best_trial.number}, Best val_acc: {best_trial.value:.4f}")


def generate_summary_report(study):
    """Generate summary report of optimization."""
    if len(study.trials) == 0:
        return "No trials completed yet."
    
    completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    failed_trials = [t for t in study.trials if t.state == TrialState.FAIL]
    
    report = f"""
Optimization Summary Report
{'=' * 80}
Study Name: {STUDY_NAME}
Total Trials: {len(study.trials)}
Completed Trials: {len(completed_trials)}
Failed Trials: {len(failed_trials)}

Best Trial:
  Number: {study.best_trial.number}
  Validation Accuracy: {study.best_trial.value:.4f}
  Parameters:
"""
    
    for param, value in study.best_trial.params.items():
        report += f"    {param}: {value}\n"
    
    report += f"\nTop 5 Trials:\n"
    sorted_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)[:5]
    for i, trial in enumerate(sorted_trials, 1):
        report += f"  {i}. Trial {trial.number}: {trial.value:.4f}\n"
    
    return report


def main():
    """Main optimization function."""
    print("=" * 80)
    print("3D GCN Hyperparameter Optimization with Optuna")
    print("=" * 80)
    print(f"Study: {STUDY_NAME}")
    print(f"Trials: {N_TRIALS}")
    print(f"Database: {STUDY_DB_URL}")
    print("=" * 80)
    
    # Initialize log file
    if os.path.exists(LOG_FILE):
        log_message("\n" + "=" * 80)
        log_message("Resuming optimization...")
    else:
        log_message("Starting new optimization...")
    
    # Load or create study
    study = load_or_create_study()
    
    # Calculate remaining trials
    completed_count = len([t for t in study.trials if t.state == TrialState.COMPLETE])
    remaining = max(0, N_TRIALS - len(study.trials))
    
    log_message(f"Completed: {completed_count}, Total trials in DB: {len(study.trials)}, Remaining: {remaining}")
    
    if remaining > 0:
        log_message(f"Starting optimization for {remaining} more trials...")
        study.optimize(objective, n_trials=remaining, show_progress_bar=True)
        log_message("Optimization completed!")
    else:
        log_message(f"Target of {N_TRIALS} trials already reached or exceeded.")
    
    # Save best configuration
    save_best_config(study)
    
    # Print summary
    summary = generate_summary_report(study)
    print(summary)
    log_message(summary)
    
    print("\n" + "=" * 80)
    print("Optimization Complete!")
    print(f"Results saved to: {os.path.dirname(os.path.abspath(__file__))}")
    print(f"Best config: best_config.json")
    print(f"Optimization log: optimization_log.txt")
    print("=" * 80)


if __name__ == '__main__':
    main()

