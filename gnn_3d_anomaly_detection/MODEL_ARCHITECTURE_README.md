# Model Architecture Diagrams

This directory contains LaTeX source files for generating visual diagrams of the 3D GCN model architecture.

## Files

1. **`model_architecture.tex`** - Main architecture diagram (recommended)
2. **`model_architecture_detailed.tex`** - More detailed version with annotations
3. **`model_architecture_simple.tex`** - Simplified compact version

## Compilation

### Using pdflatex (Recommended)

```bash
cd gnn_3d_anomaly_detection/
pdflatex model_architecture.tex
```

This will generate `model_architecture.pdf`.

### Using Overleaf (Online)

1. Go to [Overleaf.com](https://www.overleaf.com)
2. Create a new project
3. Copy the contents of `model_architecture.tex`
4. Click "Compile" to generate PDF

### Alternative: Using latexmk

```bash
latexmk -pdf model_architecture.tex
```

## Model Architecture Overview

### Input Layer
- **Node Features**: `[N, 6]` = `[x, y, timestamp, width, height, vehicle_id]`
- **Edge Index**: `[2, E]` - Graph connectivity
- **Edge Attributes**: `[E, 3]` = `[distance, frame, timestamp]`

### Feature Processing
1. Split node features into 3D coordinates and additional features
2. Concatenate back to `[N, 6]`

### GCN Layers (3 layers)
- **Layer 1**: `6 → 64` (with ReLU + Dropout)
- **Layer 2**: `64 → 64` (with ReLU + Dropout)
- **Layer 3**: `64 → 64` (no activation)

### Graph Pooling
- Mean pooling: `[N, 64] → [B, 64]`
- Max pooling: `[N, 64] → [B, 64]`
- Concatenate: `[B, 128]`

### Classifier Head
- **FC1**: `128 → 64` (ReLU + Dropout)
- **FC2**: `64 → 32` (ReLU + Dropout)
- **FC3**: `32 → 1` (Linear)

### Output
- Binary logits: `[B, 1]`
- Sigmoid activation for probabilities
- Classification: Normal (0) / Anomalous (1)

## Key Dimensions

- `N` = Number of nodes (vehicles) across all frames
- `E` = Number of edges (vehicle interactions)
- `B` = Batch size

## Notes

- The model uses **complete graphs** within each frame (all vehicles connected)
- All frames are **combined into a single graph** for processing
- Edge index is used in all GCN layers (shown as dashed arrows)
- Coordinates are normalized if `normalize_coords=True` in config

