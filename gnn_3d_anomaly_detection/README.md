# 3D Graph Convolutional Network for Video Anomaly Detection

This project implements a 3D Graph Convolutional Network (3D-GCN) for detecting anomalies in video sequences. The model processes temporal graph sequences where nodes represent vehicles with 3D coordinates (x, y, timestamp) and predicts whether a video sequence is Normal or Anomalous.

## Architecture

The 3D-GCN operates on graphs where:
- **Nodes** have 3D spatial-temporal coordinates: (x, y, timestamp)
- **Additional node features**: width, height, vehicle_id
- **Edge features**: distance between vehicles, frame number, timestamp
- The model uses multiple GCN layers to process the graph structure, followed by graph-level pooling and a classifier

## Project Structure

```
gnn_3d_anomaly_detection/
├── models/
│   ├── __init__.py
│   └── gcn_3d.py              # 3D GCN model architecture
├── data/
│   ├── __init__.py
│   └── dataset.py             # PyTorch Dataset for temporal graphs
├── train.py                   # Training script
├── evaluate.py                # Evaluation/inference script
├── config.py                  # Configuration parameters
├── utils.py                   # Utility functions
├── requirements.txt           # Dependencies
├── README.md                  # This file
└── results/                   # Output folder
    ├── checkpoints/           # Model checkpoints
    ├── logs/                  # Training logs
    └── predictions/           # Predictions and metrics
```

## Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Ensure you have processed your video data using the data pipeline:
   - The pipeline should generate temporal graphs in HDF5 format
   - CSV manifests (train_manifest.csv, val_manifest.csv, test_manifest.csv) should be available
   - See `../data_pipeline/README.md` for details

## Configuration

Edit `config.py` to adjust hyperparameters:

- **Model architecture**: `MODEL_CONFIG`
  - `hidden_dim`: Hidden dimension for GCN layers (default: 64)
  - `num_layers`: Number of GCN layers (default: 3)
  - `dropout`: Dropout rate (default: 0.5)
  - `pooling_type`: Graph pooling type ('mean', 'max', 'sum', 'meanmax', 'all')

- **Training**: `TRAIN_CONFIG`
  - `batch_size`: Batch size (default: 8)
  - `num_epochs`: Number of epochs (default: 100)
  - `learning_rate`: Learning rate (default: 0.001)
  - `patience`: Early stopping patience (default: 15)
  - `max_frames`: Maximum frames per video (None for all)

## Usage

### Training

Train the model:

```bash
cd gnn_3d_anomaly_detection
python train.py
```

The training script will:
- Load data from `output_pipeline/train_manifest.csv` and `val_manifest.csv`
- Train the 3D-GCN model with early stopping
- Save checkpoints to `results/checkpoints/`
- Save training logs and curves to `results/logs/`

### Evaluation

Evaluate the trained model:

```bash
python evaluate.py
```

The evaluation script will:
- Load the best checkpoint from training
- Evaluate on `output_pipeline/test_manifest.csv`
- Generate comprehensive metrics and visualizations
- Save results to `results/predictions/`

## Output Files

### Training Outputs

- `results/checkpoints/checkpoint.pth`: Latest model checkpoint
- `results/checkpoints/checkpoint_best.pth`: Best model checkpoint
- `results/logs/training_log.csv`: Training history (losses, accuracies)
- `results/logs/training_curves.png`: Training/validation curves

### Evaluation Outputs

- `results/predictions/test_predictions.csv`: Detailed predictions for each video
- `results/predictions/test_metrics.csv`: Overall classification metrics
- `results/predictions/classification_report.csv`: Per-class metrics
- `results/predictions/confusion_matrix.png`: Confusion matrix visualization
- `results/predictions/roc_curve.png`: ROC curve
- `results/predictions/pr_curve.png`: Precision-Recall curve

## Metrics

The evaluation script computes:
- **Accuracy**: Overall classification accuracy
- **Precision**: Precision score
- **Recall**: Recall score
- **F1-Score**: F1 score
- **AUC-ROC**: Area under ROC curve
- **AUC-PR**: Area under Precision-Recall curve

## Data Format

The model expects temporal graphs stored in HDF5 format with the following structure:

```
/graphs/t_XXXX/
  ├── node_features [N, 6]: [x, y, width, height, timestamp, vehicle_id]
  ├── edge_index [2, E]: Edge connections
  └── edge_features [E, 3]: [distance, frame, timestamp]
```

The dataset loader combines all frames into a single graph where nodes have 3D coordinates (x, y, timestamp).

## Model Details

### Input
- Node features: [x, y, timestamp, width, height, vehicle_id]
- Edge indices: Graph connectivity
- Edge features: [distance, frame, timestamp]

### Architecture
1. **GCN Layers**: Multiple Graph Convolutional layers process node features
2. **Graph Pooling**: Global pooling (mean/max/sum) aggregates graph-level features
3. **Classifier**: Multi-layer perceptron for binary classification

### Output
- Logits for binary classification (Normal=0, Anomalous=1)

## Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` in `config.py`
- Set `max_frames` to limit frames per video
- Reduce `hidden_dim` or `num_layers`

### No Checkpoint Found
- Ensure training has been completed first
- Check that `results/checkpoints/` contains checkpoint files

### Empty Datasets
- Verify that CSV manifests exist and contain valid paths
- Check that HDF5 files exist in the specified video folders
- Ensure the data pipeline has been run successfully

## Dependencies

- PyTorch >= 2.0.0
- PyTorch Geometric >= 2.3.0
- h5py >= 3.8.0
- pandas >= 1.5.0
- numpy >= 1.23.0
- scikit-learn >= 1.2.0
- matplotlib >= 3.6.0
- seaborn >= 0.12.0

## Integration

This implementation integrates with the existing data pipeline:
- Uses temporal graphs from `output_pipeline/`
- Reads from CSV manifests generated by the data pipeline
- Compatible with the HDF5 format produced by `vehicle_processor.py`

## Citation

If you use this code, please cite:
```
3D Graph Convolutional Network for Video Anomaly Detection
Using spatial-temporal graph representations for video sequence classification
```

## License

This project is part of the GNN Research repository.










