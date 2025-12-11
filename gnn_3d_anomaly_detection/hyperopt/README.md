# Hyperparameter Optimization with Optuna

This module implements automated hyperparameter optimization for the 3D Graph Convolutional Network using Optuna.

## Overview

The optimization process:
1. Uses Optuna to search hyperparameter space over 100 trials
2. Stores all trial data in SQLite database for resume capability
3. Saves best configuration found
4. Logs all trials with detailed information

## Installation

Ensure Optuna is installed:
```bash
pip install optuna>=3.0.0 optuna-dashboard>=0.13.0
```

Or install all requirements:
```bash
pip install -r ../requirements.txt
```

## Usage

### Run Optimization

From the `gnn_3d_anomaly_detection` directory:

```bash
cd hyperopt
python optimize.py
```

### Resume Interrupted Optimization

If optimization is interrupted, simply run again:
```bash
python optimize.py
```

Optuna will automatically resume from the last completed trial using the database in `study_db/optuna_study.db`.

## Directory Structure

```
hyperopt/
├── __init__.py
├── config.py              # Hyperparameter search space definitions
├── optimize.py            # Main optimization script
├── train_trial.py         # Reusable training function for trials
├── README.md              # This file
├── study_db/              # Optuna database storage
│   └── optuna_study.db
├── trials/                # Per-trial outputs
│   ├── trial_0/
│   │   ├── trial_info.json
│   │   ├── checkpoint_best.pth
│   │   └── ...
│   ├── trial_1/
│   └── ...
├── best_config.json       # Best configuration found (auto-generated)
└── optimization_log.txt   # Optimization log file (auto-generated)
```

## Hyperparameters Optimized

### Model Architecture
- **hidden_dim**: Integer [32, 128] (step=16) - GCN hidden dimension
- **num_layers**: Integer [2, 5] - Number of GCN layers
- **dropout**: Float [0.1, 0.7] (step=0.1) - Dropout rate
- **pooling_type**: Categorical ['mean', 'max', 'sum', 'meanmax', 'all']

### Training Parameters
- **batch_size**: Integer [1, 8] (step=1) - Training batch size
- **learning_rate**: Float [1e-7, 1e-3] (log scale) - Learning rate
- **weight_decay**: Float [1e-7, 1e-4] (log scale) - Weight decay
- **patience**: Integer [5, 30] (step=5) - Early stopping patience
- **min_delta**: Float [1e-5, 1e-2] (log scale) - Early stopping minimum improvement

### Fixed Parameters (Not Optimized)
- device: CUDA device (from base config)
- num_workers: DataLoader workers
- pin_memory: Memory pinning
- max_frames: Maximum frames per video
- normalize_coords: Coordinate normalization flag
- num_epochs: Maximum epochs (100)

## Objective Metric

The optimization **maximizes validation accuracy** on the validation set.

## Output Files

### best_config.json

Contains the best hyperparameter configuration found:
```json
{
  "best_trial_number": 42,
  "best_validation_accuracy": 0.7500,
  "datetime": "2024-11-02T14:30:00",
  "total_trials": 100,
  "completed_trials": 95,
  "model_config": { ... },
  "train_config": { ... },
  "all_params": { ... }
}
```

### optimization_log.txt

Logs all trial starts, completions, and important events with timestamps.

### trials/trial_X/trial_info.json

Per-trial information including:
- Trial number
- Hyperparameters used
- Validation accuracy achieved
- Model and training configs
- Trial state (COMPLETE/FAIL)

### trials/trial_X/checkpoint_best.pth

Best checkpoint for each trial (saved at best epoch during training).

## Monitoring Progress

### View Optuna Dashboard

Launch Optuna dashboard to visualize optimization progress:

```bash
optuna-dashboard sqlite:///study_db/optuna_study.db
```

Then open http://localhost:8080 in your browser.

### Check Log File

```bash
tail -f optimization_log.txt
```

## Using Best Configuration

After optimization completes, use the best configuration:

```python
import json

# Load best config
with open('best_config.json', 'r') as f:
    best_config = json.load(f)

# Extract model and training configs
model_config = best_config['model_config']
train_config = best_config['train_config']

# Use in your training script
from models import GCN3D
model = GCN3D(**model_config)
```

## Configuration

To modify search spaces, edit `config.py`:

```python
def suggest_hyperparameters(trial):
    hidden_dim = trial.suggest_int('hidden_dim', 32, 128, step=16)
    # Modify ranges as needed
    ...
```

## Troubleshooting

### Database Locked
If you see database lock errors, ensure only one optimization process is running at a time.

### Out of Memory
If trials run out of memory, reduce the search space for `hidden_dim` or `batch_size` in `config.py`.

### Resume Not Working
Check that `study_db/optuna_study.db` exists and is readable. If corrupted, delete and restart.

## Notes

- Each trial trains until early stopping (based on patience parameter)
- Trials run sequentially (not in parallel) to avoid resource conflicts
- The database stores complete history allowing analysis of all trials
- Failed trials are logged but don't count toward the 100 trial limit


