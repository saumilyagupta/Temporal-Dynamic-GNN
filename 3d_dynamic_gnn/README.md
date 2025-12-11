# 3D Dynamic Graph Neural Network

Implementation of a 3D Dynamic Graph Neural Network for accident prediction using temporal graph sequences.

## Architecture

The model processes temporal graph sequences with the following architecture:

1. **Spatial Processing**: Graph Convolutional Network (GCN) layers applied per frame
2. **Temporal Modeling**: GRU/LSTM to capture temporal dependencies across frames
3. **Global Average Pooling**: Pools across nodes and time dimensions
4. **Classification**: Binary classification head (Normal vs Anomalous)

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Edit `config.yaml` to configure:
- Dataset paths and manifest files
- Model hyperparameters (hidden_dim, num_layers, dropout, etc.)
- Training hyperparameters (batch_size, learning_rate, epochs)
- Callback settings (checkpointing, early stopping)

## Usage

### Training

Train the model:

```bash
python train.py --config config.yaml --seed 42
```

The training script will:
- Load the dataset from HDF5 files
- Calculate focal loss alpha from class distribution
- Train the model with PyTorch Lightning
- Save checkpoints and logs to `./logs/`

### Evaluation

Evaluate a trained model:

```bash
python evaluate.py \
    --checkpoint path/to/checkpoint.ckpt \
    --config config.yaml \
    --split test
```

Options for `--split`:
- `train`: Evaluate on training set
- `val`: Evaluate on validation set
- `test`: Evaluate on test set (default)

Results will be saved to `./results/` including:
- Predictions CSV file
- Confusion matrix visualization
- Metrics text file

## File Structure

```
3d_dynamic_gnn/
├── dataset.py          # HDF5 dataset loader with masking support
├── model.py            # Dynamic GNN architecture
├── loss.py             # Focal loss implementation
├── lightning_module.py # PyTorch Lightning wrapper
├── config.yaml         # Configuration file
├── train.py            # Training script
├── evaluate.py         # Evaluation script
├── utils.py            # Helper functions (metrics, visualization)
├── requirements.txt    # Dependencies
├── docs.md             # Dataset documentation
└── README.md           # This file
```

## Model Details

### Input Format
- Node features: `[T, N, 8]` where T is frames, N is nodes
- Adjacency matrices: `[T, N, N]` with distance-based edges
- Node masks: `[T, N]` indicating valid nodes per frame
- Edge masks: `[T, N, N]` indicating valid edges per frame

### Loss Function
- **Focal Loss** with:
  - Gamma (γ) = 1.0
  - Alpha (α) = calculated from training set class distribution
  - Formula: `FL(p_t) = -α * (1 - p_t)^γ * log(p_t)`

### Key Features
- Handles variable-sized graphs using masking
- Processes complete temporal sequences
- Supports dynamic graphs (nodes/edges appear/disappear)
- Global average pooling across nodes and time

## Outputs

After training, you'll find:
- Model checkpoints in `./logs/dynamic_gnn/`
- TensorBoard logs for visualization
- Best model based on validation loss

After evaluation:
- Predictions CSV in `./results/`
- Confusion matrix PNG
- Metrics summary text file

## Notes

- The model uses PyTorch Geometric for graph operations
- PyTorch Lightning handles training infrastructure
- All temporal graphs are processed in sequence
- Variable graph sizes are handled via masking during pooling

