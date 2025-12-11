# Video Processing Pipeline with Train/Val/Test Split

Complete data processing pipeline for vehicle detection, tracking, and temporal graph generation with automatic train/validation/test splitting.

## Features

- **Pre-split Dataset Ingestion**: Reads NiAD_Large_Videos structure (Training/Validation/Testing) with class labels from folder names
- **Vehicle Detection & Tracking**: YOLOv8-based detection (car, motorcycle, bus, truck) with persistent IDs across frames
- **Dynamic Graph Generation**: Builds fixed-size node feature matrices (one row per vehicle in the entire clip) and complete adjacency matrices per frame
- **Normalization Support**: Provides normalized timestamps for every node feature vector
- **Organized Output**: Per-video folders with raw video, visualization, and HDF5 graph data under train/val/test splits
- **CSV Manifests**: Easy-to-use manifests listing processed videos and metadata

## Quick Start

### 1. Edit Configuration

Open `main.py` and edit the `CONFIG` dictionary at the top:

```python
CONFIG = {
    'input_folder': '/workspace/saumilya/GNN-Research/accident_prediction/NiAD_Large_Videos',
    'output_folder': '/workspace/saumilya/GNN-Research/final_dataset',
    'split_mapping': {'Training': 'train', 'Validation': 'val', 'Testing': 'test'},
    'class_mapping': {'Normal': 'Normal', 'Accident': 'Anomalous'},
    'yolo_model': 'yolov8n.pt',
    'detection_conf': 0.3,
    'vehicle_classes': [2, 3, 5, 7],
    'frames_per_video': 30,
    'missing_value': -1
}
```

### 2. Run Pipeline

```bash
cd /workspace/saumilya/GNN-Research/final_dataset/data_pipeline
python main.py
```

## Input Format

The pipeline expects the NiAD_Large_Videos directory structure:

```
NiAD_Large_Videos/
├── Training/
│   ├── Normal/
│   └── Accident/
├── Validation/
│   ├── Normal/
│   └── Accident/
└── Testing/
    ├── Normal/
    └── Accident/
```

Splits are inferred from the parent directory (Training → `train`, Validation → `val`, Testing → `test`). Class labels are inferred from the class folder name (`Normal` or `Accident`, mapped to `Normal` / `Anomalous`).

## Output Structure

```
output_folder/
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

Each manifest contains:
- `video_folder_path`: Path to video folder
- `label`: Normal or Anomalous
- `source_video`: Original source video name
- `num_vehicles`: Number of unique vehicles detected
- `num_frames`: Number of frames in clip

## HDF5 File Format

Each `temporal_graphs.h5` file contains:

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
- Column 0: centroid x
- Column 1: centroid y
- Column 2: bounding box x (top-left)
- Column 3: bounding box y (top-left)
- Column 4: bounding box width
- Column 5: bounding box height
- Column 6: vehicle_class (YOLO ID: 2, 3, 5, 7)
- Column 7: normalized timestamp (0 → start, 1 → end)

### Adjacency Matrix
- Complete graph of all vehicles that appear in the video
- Entries are Euclidean distances between vehicle centroids
- Cells are `-1` when one or both vehicles are absent in that frame

## Loading Data for Training

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

## Module Descriptions

- **main.py**: Main orchestration script with configuration
- **video_cropper.py**: Crops videos based on JSON timestamps
- **vehicle_processor.py**: YOLO detection, tracking, graph generation
- **data_splitter.py**: Train/val/test splitting and organization
- **utils.py**: Shared utility functions

## Requirements

- Python 3.8+
- OpenCV (cv2)
- NumPy
- NetworkX
- h5py
- ultralytics (YOLO)
- scipy

## Notes

- All clips from the same source video stay in the same split (train/val/test)
- Split is performed randomly with a seed for reproducibility
- Complete graphs are generated (all vehicles connected to all others)
- Visualization videos show bounding boxes, vehicle IDs, trajectories, and edges

