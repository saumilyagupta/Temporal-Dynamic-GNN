# 3D Dynamic Graph Neural Network Documentation

## Dataset Overview

**Dataset Path:** `/workspace/saumilya/GNN-Research/NiAD_Large_Videos_processed_graphs`

### Directory Structure

```
NiAD_Large_Videos_processed_graphs/
├── train/
│   ├── Normal/
│   │   └── 000037_1_N/
│   │       ├── raw_video.mp4              # Original 30-frame clip
│   │       ├── visualization_video.mp4    # With bounding boxes & edges
│   │       └── temporal_graphs.h5         # Dynamic graph data
│   └── Anomalous/
├── val/
│   ├── Normal/
│   └── Anomalous/
├── test/
│   ├── Normal/
│   └── Anomalous/
├── train_manifest.csv
├── val_manifest.csv
└── test_manifest.csv
```

## CSV Manifest Format

Each manifest file (`train_manifest.csv`, `val_manifest.csv`, `test_manifest.csv`) contains the following columns:

- **`video_folder_path`**: Path to the video folder
- **`label`**: Class label (`Normal` or `Anomalous`)
- **`source_video`**: Original source video name
- **`num_vehicles`**: Number of unique vehicles detected in the clip
- **`num_frames`**: Number of frames in the clip

## HDF5 File Format

Each `temporal_graphs.h5` file contains temporal graph data for a video clip.

### File Structure

```
/graphs/
  /t_0000/                      # Frame 0
    /node_features [N, 8]      # Fixed-size feature matrix (missing=-1)
    /adjacency_matrix [N, N]   # Complete graph distances (missing=-1)
    /node_mask [N]             # 1 if node present in frame else 0
    /edge_mask [N, N]          # 1 if edge valid else 0
  /t_0001/                      # Frame 1
    ...
/summary/
  /vehicle_id_mapping
  /vehicle_class_ids
  /attributes (statistics)
```

### Node Features [N, 8]

The node feature matrix has 8 columns:

- **Column 0**: Centroid x-coordinate
- **Column 1**: Centroid y-coordinate
- **Column 2**: Bounding box x (top-left corner)
- **Column 3**: Bounding box y (top-left corner)
- **Column 4**: Bounding box width
- **Column 5**: Bounding box height
- **Column 6**: Vehicle class (YOLO ID: 2, 3, 5, 7)
- **Column 7**: Normalized timestamp (0 → start of clip, 1 → end of clip)

**Note:** Missing values are represented as `-1`.

### Adjacency Matrix [N, N]

- Complete graph representation of all vehicles that appear in the video
- Entries contain Euclidean distances between vehicle centroids
- Cells are set to `float('-inf')` when one or both vehicles are absent in that frame
- Missing values are represented as `-1`

### Masks

- **`node_mask [N]`**: Binary mask indicating which nodes are present in the frame (1 = present, 0 = absent)
- **`edge_mask [N, N]`**: Binary mask indicating which edges are valid (1 = valid, 0 = invalid)

## Loading Data for Training

Example code to load temporal graph data:

```python
import h5py
import torch
import pandas as pd

# Load manifest
df = pd.read_csv('output_folder/train_manifest.csv')

# Load a video's temporal graphs
video_folder = df.iloc[0]['video_folder_path']
h5_path = f"{video_folder}/temporal_graphs.h5"

with h5py.File(h5_path, 'r') as f:
    # Load specific frame
    nodes = torch.tensor(f['graphs/t_0010/node_features'][:])
    adjacency = torch.tensor(f['graphs/t_0010/adjacency_matrix'][:])
    node_mask = torch.tensor(f['graphs/t_0010/node_mask'][:])

```

## Model Requirements

### Architecture

We need to implement a **3D Dynamic Graph Neural Network** for dynamic graph temporal data.

**Key Requirements:**

1. **Input Format:**
   - The model must process the **complete sequence** of feature matrices and adjacency matrices
   - Feature matrix: `[T, N, 8]` where T is the number of frames and N is the number of nodes
   - Adjacency matrix: `[T, N, N]` where T is the number of frames and N is the number of nodes

2. **Variable Graph Size:**
   - For different sequences, the maximum number of nodes (N) in the graph sequence will be **variable**
   - The model must handle variable-sized graphs using masks (node_mask and edge_mask)

3. **Temporal Modeling:**
   - The model should capture temporal dependencies across frames
   - Must handle dynamic graphs where nodes and edges can appear/disappear over time

### Loss Function

**Focal Loss** will be used for training:

- **Gamma (γ)**: `1.0`
- **Alpha (α)**: Calculated based on the number of Normal and Anomalous sequences in the training set
  - Formula: `α = (num_normal_samples) / (num_normal_samples + num_anomalous_samples)`
  - This helps balance the class weights for imbalanced datasets

**Focal Loss Formula:**
```
FL(p_t) = -α * (1 - p_t)^γ * log(p_t)
```

Where:
- `p_t` is the predicted probability for the true class
- `α` is the balancing factor
- `γ` is the focusing parameter (set to 1.0)
