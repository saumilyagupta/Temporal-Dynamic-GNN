"""
Train TemporalGNN on CarCrash VGG16 bbox embeddings; then run evaluate_carcrash.
"""

import os
import shutil
import argparse
from datetime import datetime

import torch
import torch.multiprocessing
import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger

torch.multiprocessing.set_sharing_strategy("file_system")

from dataset_carcrash import (
    create_carcrash_dataloaders,
    create_carcrash_precomputed_dataloaders,
)
from evaluate_carcrash import evaluate_and_save
from loss import calculate_alpha_from_dataset
from lightning_module import TemporalGNNLightning
from model import TemporalGNN
from utils import count_parameters, get_device, set_seed


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train TemporalGNN on CarCrash")
    parser.add_argument(
        "--config",
        type=str,
        default="config_carcrash.yaml",
        help="Path to configuration file",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp_name", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)

    exp_start = datetime.now()
    ts = exp_start.strftime("%Y%m%d_%H%M%S")
    base_name = args.exp_name or "carcrash_temporal_gnn"
    exp_dir = os.path.join("experiments", f"{base_name}_{ts}")
    os.makedirs(exp_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("CarCrash TemporalGNN Training")
    print("=" * 60)
    print(f"Experiment directory: {exp_dir}")
    print("=" * 60 + "\n")

    config = load_config(args.config)

    gpu_id = config["training"].get("gpu_id", 0)
    if gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"Using GPU: {gpu_id}")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("Using CPU (gpu_id=-1)")

    shutil.copy(args.config, os.path.join(exp_dir, "config.yaml"))
    print("Configuration:")
    print("-" * 40)
    print(yaml.dump(config, default_flow_style=False))
    print("-" * 40 + "\n")

    device = get_device()

    data_cfg = config["data"]
    train_cfg = config["training"]
    num_frames = int(data_cfg.get("num_frames", 50))

    if data_cfg.get("source", "npz") == "precomputed":
        train_loader, val_loader, test_loader, train_dataset = (
            create_carcrash_precomputed_dataloaders(
                graph_export_root=data_cfg["graph_export_root"],
                train_manifest=data_cfg.get("train_manifest", "train_manifest.csv"),
                test_manifest=data_cfg.get("test_manifest", "test_manifest.csv"),
                batch_size=train_cfg["batch_size"],
                val_split=float(data_cfg.get("val_split", 0.2)),
                seed=args.seed,
                num_workers=data_cfg["num_workers"],
                pin_memory=data_cfg.get("pin_memory", True),
                normalize_features=data_cfg.get("normalize_features", True),
                num_frames=num_frames,
            )
        )
    else:
        train_loader, val_loader, test_loader, train_dataset = create_carcrash_dataloaders(
            data_root=data_cfg["root_dir"],
            train_txt=data_cfg["train_txt"],
            test_txt=data_cfg["test_txt"],
            batch_size=train_cfg["batch_size"],
            val_split=float(data_cfg.get("val_split", 0.2)),
            seed=args.seed,
            num_workers=data_cfg["num_workers"],
            pin_memory=data_cfg.get("pin_memory", True),
            normalize_features=data_cfg.get("normalize_features", True),
            num_frames=num_frames,
        )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    loss_cfg = config.get("loss") or {}
    if loss_cfg.get("alpha") is not None:
        alpha = float(loss_cfg["alpha"])
        print(f"\nUsing focal loss alpha from config: {alpha:.4f}")
    else:
        print("\nCalculating focal loss alpha from training set...")
        alpha = calculate_alpha_from_dataset(train_dataset)

    print("\nCreating model...")
    model = TemporalGNN(
        input_dim=config["model"]["input_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        temporal_dim=config["model"]["temporal_dim"],
        num_gcn_layers=config["model"]["num_gcn_layers"],
        num_gru_layers=config["model"]["num_gru_layers"],
        dropout=config["model"]["dropout"],
        num_classes=config["model"]["num_classes"],
    )
    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}")

    with open(os.path.join(exp_dir, "model_architecture.txt"), "w") as f:
        f.write(str(model))
        f.write(f"\n\nTotal parameters: {n_params:,}")

    lightning_module = TemporalGNNLightning(
        model=model,
        focal_loss_alpha=alpha,
        focal_loss_gamma=float(config["loss"]["gamma"]),
        learning_rate=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
        optimizer=train_cfg["optimizer"],
        scheduler=train_cfg["scheduler"],
        monitor_metric=train_cfg["monitor_metric"],
        monitor_mode=train_cfg["monitor_mode"],
    )

    checkpoint_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{epoch:02d}-{val_loss:.4f}-{val_f1:.4f}",
        monitor=train_cfg["monitor_metric"],
        mode=train_cfg["monitor_mode"],
        save_top_k=config["callbacks"]["checkpoint"]["save_top_k"],
        save_last=config["callbacks"]["checkpoint"]["save_last"],
        every_n_epochs=config["callbacks"]["checkpoint"]["every_n_epochs"],
        verbose=True,
    )

    early_stopping = EarlyStopping(
        monitor=train_cfg["monitor_metric"],
        mode=train_cfg["monitor_mode"],
        patience=config["callbacks"]["early_stopping"]["patience"],
        min_delta=config["callbacks"]["early_stopping"]["min_delta"],
        verbose=True,
    )

    callbacks = [checkpoint_callback, early_stopping, LearningRateMonitor(logging_interval="epoch")]

    log_dir = os.path.join(exp_dir, "logs")
    logger = TensorBoardLogger(save_dir=log_dir, name="training", version=None)

    trainer = pl.Trainer(
        max_epochs=train_cfg["max_epochs"],
        accelerator="gpu" if device.type == "cuda" else "cpu",
        devices=1 if device.type == "cuda" else "auto",
        gradient_clip_val=train_cfg["gradient_clip_val"],
        accumulate_grad_batches=train_cfg["accumulate_grad_batches"],
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=config["logging"]["log_every_n_steps"],
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
        precision="16-mixed" if device.type == "cuda" else "32",
    )

    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")

    trainer.fit(
        lightning_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    print("\nTraining completed.")
    best_path = checkpoint_callback.best_model_path
    print(f"Best model: {best_path}")
    print(f"Best {train_cfg['monitor_metric']}: {checkpoint_callback.best_model_score}")

    if best_path:
        best_dest = os.path.join(exp_dir, "best_model.ckpt")
        shutil.copy(best_path, best_dest)
        print(f"Copied to: {best_dest}")

        print("\nRunning CarCrash evaluation (combined + Day/Night)...")
        evaluate_and_save(
            checkpoint_path=best_dest,
            config=config,
            exp_dir=exp_dir,
            seed=args.seed,
        )

    summary_path = os.path.join(exp_dir, "experiment_summary.txt")
    with open(summary_path, "w") as f:
        f.write("CarCrash TemporalGNN — summary\n")
        f.write(f"Start: {exp_start}\n")
        f.write(f"End: {datetime.now()}\n")
        f.write(f"Config: {args.config}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Best ckpt: {best_path}\n")
        f.write(f"Params: {n_params:,}\n")
        f.write(f"Alpha: {alpha:.4f}\n")

    print(f"\nExperiment directory: {exp_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
