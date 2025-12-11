"""
Optuna objective function for TemporalGNN hyperparameter optimization.
Trains the model and returns validation F1 score for optimization.
"""

import os
import sys
import torch
import torch.multiprocessing
import numpy as np
from typing import Dict, Any, Optional
import optuna
from optuna.trial import Trial
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import f1_score, precision_score, recall_score

# Fix for shared memory issues
torch.multiprocessing.set_sharing_strategy('file_system')

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import create_dataloaders, TemporalGraphDataset
from model import TemporalGNN
from loss import FocalLoss, calculate_alpha_from_dataset
from lightning_module import TemporalGNNLightning
from search_space import suggest_all_hyperparameters, load_search_space_config


class OptunaPruningCallback(pl.Callback):
    """
    PyTorch Lightning callback for Optuna pruning.
    Reports intermediate values and handles pruning.
    """
    
    def __init__(self, trial: Trial, monitor: str = "val_f1"):
        super().__init__()
        self.trial = trial
        self.monitor = monitor
    
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Report and check for pruning after each validation epoch."""
        epoch = trainer.current_epoch
        
        # Get the monitored metric
        current_score = trainer.callback_metrics.get(self.monitor)
        
        if current_score is not None:
            # Report to Optuna
            self.trial.report(float(current_score), epoch)
            
            # Check if trial should be pruned
            if self.trial.should_prune():
                raise optuna.TrialPruned()


def create_objective(
    config_path: str = "optuna_config.yaml",
    data_root: Optional[str] = None,
    gpu_id: Optional[int] = None,
):
    """
    Create an objective function for Optuna optimization.
    
    Args:
        config_path: Path to optuna configuration file
        data_root: Override data root directory
        gpu_id: Override GPU ID
    
    Returns:
        Objective function for Optuna
    """
    import yaml
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override settings if provided
    if data_root is not None:
        config['data']['root_dir'] = data_root
    if gpu_id is not None:
        config['trial_training']['gpu_id'] = gpu_id
    
    # Extract settings
    data_config = config['data']
    trial_config = config['trial_training']
    search_config = config['search_space']
    objective_config = config.get('objective', {})
    
    # Set GPU
    gpu = trial_config.get('gpu_id', 0)
    if gpu >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    
    def objective(trial: Trial) -> float:
        """
        Objective function for a single Optuna trial.
        
        Args:
            trial: Optuna trial object
        
        Returns:
            Validation F1 score (or combined metric)
        """
        # Create dataloaders
        train_loader, val_loader, _ = create_dataloaders(
            data_root=data_config['root_dir'],
            train_manifest=data_config['train_manifest'],
            val_manifest=data_config['val_manifest'],
            batch_size=4,  # Temporary, will be overridden
            num_workers=data_config.get('num_workers', 0),
            normalize_features=data_config.get('normalize_features', True),
        )
        
        # Calculate base alpha from class distribution
        train_manifest_path = os.path.join(
            data_config['root_dir'],
            data_config['train_manifest']
        )
        temp_dataset = TemporalGraphDataset(
            manifest_path=train_manifest_path,
            normalize_features=False,
        )
        base_alpha = calculate_alpha_from_dataset(temp_dataset)
        
        # Sample hyperparameters
        params = suggest_all_hyperparameters(trial, base_alpha, search_config)
        
        # Recreate dataloaders with suggested batch size
        train_loader, val_loader, _ = create_dataloaders(
            data_root=data_config['root_dir'],
            train_manifest=data_config['train_manifest'],
            val_manifest=data_config['val_manifest'],
            batch_size=params['training']['batch_size'],
            num_workers=data_config.get('num_workers', 0),
            normalize_features=data_config.get('normalize_features', True),
        )
        
        # Create model
        model = TemporalGNN(
            input_dim=params['model']['input_dim'],
            hidden_dim=params['model']['hidden_dim'],
            temporal_dim=params['model']['temporal_dim'],
            num_gcn_layers=params['model']['num_gcn_layers'],
            num_gru_layers=params['model']['num_gru_layers'],
            dropout=params['model']['dropout'],
            num_classes=params['model']['num_classes'],
        )
        
        # Create Lightning module
        lightning_module = TemporalGNNLightning(
            model=model,
            focal_loss_alpha=params['loss']['focal_alpha'],
            focal_loss_gamma=params['loss']['focal_gamma'],
            learning_rate=params['training']['learning_rate'],
            weight_decay=params['training']['weight_decay'],
            optimizer=params['training']['optimizer'],
            scheduler='plateau',
            monitor_metric='val_f1',
            monitor_mode='max',
        )
        
        # Callbacks
        callbacks = [
            OptunaPruningCallback(trial, monitor='val_f1'),
            EarlyStopping(
                monitor='val_f1',
                mode='max',
                patience=trial_config.get('patience', 10),
                verbose=False,
            ),
        ]
        
        # Trainer
        trainer = pl.Trainer(
            max_epochs=trial_config.get('max_epochs', 50),
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            callbacks=callbacks,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,  # Disable logging for HPO
            gradient_clip_val=trial_config.get('gradient_clip_val', 1.0),
        )
        
        # Train
        try:
            trainer.fit(lightning_module, train_loader, val_loader)
        except optuna.TrialPruned:
            raise
        except Exception as e:
            print(f"Trial {trial.number} failed with error: {e}")
            return 0.0  # Return worst score on failure
        
        # Get validation predictions with custom threshold
        lightning_module.eval()
        device = next(lightning_module.parameters()).device
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for graphs_list, labels in val_loader:
                logits = lightning_module(graphs_list)
                probs = torch.softmax(logits, dim=1)
                
                # Apply custom threshold
                threshold = params['threshold']
                preds = (probs[:, 1] >= threshold).long()
                
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
                all_probs.append(probs[:, 1].cpu())
        
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        all_probs = torch.cat(all_probs).numpy()
        
        # Calculate metrics
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        
        # Log additional metrics
        trial.set_user_attr('precision', precision)
        trial.set_user_attr('recall', recall)
        trial.set_user_attr('f1', f1)
        
        # Calculate objective value
        use_weighted = objective_config.get('use_weighted', False)
        
        if use_weighted:
            weights = objective_config.get('weights', {'f1': 0.5, 'precision': 0.3, 'recall': 0.2})
            objective_value = (
                weights.get('f1', 0.5) * f1 +
                weights.get('precision', 0.3) * precision +
                weights.get('recall', 0.2) * recall
            )
        else:
            objective_value = f1
        
        print(f"Trial {trial.number}: F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
        
        return objective_value
    
    return objective


def run_single_trial(
    params: Dict[str, Any],
    config_path: str = "optuna_config.yaml",
    save_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    Run a single trial with specified hyperparameters.
    Useful for evaluating best parameters found by Optuna.
    
    Args:
        params: Hyperparameter dictionary
        config_path: Path to configuration file
        save_path: Optional path to save the trained model
    
    Returns:
        Dictionary of metrics
    """
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_config = config['data']
    trial_config = config['trial_training']
    
    # Set GPU
    gpu = trial_config.get('gpu_id', 0)
    if gpu >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_root=data_config['root_dir'],
        train_manifest=data_config['train_manifest'],
        val_manifest=data_config['val_manifest'],
        test_manifest=data_config.get('test_manifest'),
        batch_size=params['training']['batch_size'],
        num_workers=data_config.get('num_workers', 0),
        normalize_features=data_config.get('normalize_features', True),
    )
    
    # Create model
    model = TemporalGNN(
        input_dim=params['model']['input_dim'],
        hidden_dim=params['model']['hidden_dim'],
        temporal_dim=params['model']['temporal_dim'],
        num_gcn_layers=params['model']['num_gcn_layers'],
        num_gru_layers=params['model']['num_gru_layers'],
        dropout=params['model']['dropout'],
        num_classes=params['model']['num_classes'],
    )
    
    # Create Lightning module
    lightning_module = TemporalGNNLightning(
        model=model,
        focal_loss_alpha=params['loss']['focal_alpha'],
        focal_loss_gamma=params['loss']['focal_gamma'],
        learning_rate=params['training']['learning_rate'],
        weight_decay=params['training']['weight_decay'],
        optimizer=params['training']['optimizer'],
        scheduler='plateau',
        monitor_metric='val_f1',
        monitor_mode='max',
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_f1',
            mode='max',
            patience=trial_config.get('patience', 10),
            verbose=True,
        ),
    ]
    
    if save_path:
        callbacks.append(
            ModelCheckpoint(
                dirpath=os.path.dirname(save_path),
                filename=os.path.basename(save_path).replace('.ckpt', ''),
                monitor='val_f1',
                mode='max',
                save_top_k=1,
            )
        )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=trial_config.get('max_epochs', 50),
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=callbacks,
        enable_progress_bar=True,
        gradient_clip_val=trial_config.get('gradient_clip_val', 1.0),
    )
    
    # Train
    trainer.fit(lightning_module, train_loader, val_loader)
    
    # Evaluate on test set
    if test_loader:
        trainer.test(lightning_module, test_loader)
        return lightning_module.test_metrics
    
    return {}


