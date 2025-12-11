# Graph Data Production Guide

This document explains how temporal graph data is generated from video sequences for Dynamic Graph Networks.

## Overview

The pipeline converts 30-frame video clips into temporal graph sequences where:
- **Nodes** represent vehicles detected throughout the entire video
- **Edges** represent spatial relationships (Euclidean distances) between vehicles
- **Features** capture vehicle properties (position, bounding box, class, timestamp)
- **Graphs** are complete (all vehicles connected) with fixed-size matrices

## Pipeline Steps

### 1. Video Input
- Videos are 30 frames each
- Class labels come from parent folder (Normal/Accident)
- Pre-split into Training/Validation/Testing sets

### 2. Vehicle Detection (First Pass)
For each video:
1. **YOLO Detection**: Run YOLOv8 on all 30 frames to detect vehicles
   - Vehicle classes: car (2), motorcycle (3), bus (5), truck (7)
   - Confidence threshold: 0.3
   - Output: bounding boxes, class IDs, confidence scores

2. **Vehicle Tracking**: Assign consistent IDs across frames
   - Uses centroid-based tracking with maximum distance threshold
   - Maintains vehicle identity even when temporarily occluded
   - Creates a registry of ALL unique vehicles that appear at any point

3. **Vehicle Registry**: Build complete list of vehicles
   - `N_total` = total unique vehicles in entire video sequence
   - Each vehicle gets a unique ID that persists across frames
   - Store vehicle class (YOLO class ID) for each vehicle

### 3. Graph Construction (Second Pass)
For each of the 30 frames:

#### 3.1 Node Feature Matrix [N_total × 8]
Initialize matrix with `-1` (missing value) for all vehicles:

```python
node_features = np.full((N_total, 8), -1.0)
```

For vehicles **present** in the current frame:
- Fill row with actual values:
  1. `centroid_x`: X-coordinate of bounding box center
  2. `centroid_y`: Y-coordinate of bounding box center
  3. `bbox_x`: Top-left X coordinate
  4. `bbox_y`: Top-left Y coordinate
  5. `bbox_w`: Bounding box width
  6. `bbox_h`: Bounding box height
  7. `vehicle_class`: YOLO class ID (2, 3, 5, or 7)
  8. `normalized_timestamp`: Frame index / (total_frames - 1), range [0, 1]

For vehicles **absent** in the current frame:
- Keep all values as `-1` (including class and timestamp)

#### 3.2 Adjacency Matrix [N_total × N_total]
Initialize matrix with `-1` (missing value):

```python
adjacency_matrix = np.full((N_total, N_total), -1.0)
```

For each pair of vehicles:
- If **both vehicles present** in frame:
  - Compute Euclidean distance between their centroids
  - Store distance in both `[i, j]` and `[j, i]` (symmetric matrix)
  - Set diagonal `[i, i] = 0.0` (self-connection)
  
- If **either vehicle absent** in frame:
  - Keep value as `-1`

This creates a **complete graph** structure where:
- All vehicles that ever appear are nodes
- All pairs are potentially connected
- Missing vehicles/edges are marked with `-1`

#### 3.3 Node and Edge Masks
- **Node Mask** [N_total]: Binary array, `1` if vehicle present, `0` if absent
- **Edge Mask** [N_total × N_total]: Binary matrix, `1` if edge valid (both vehicles present), `0` otherwise

### 4. Data Storage (HDF5 Format)

Each video produces a `temporal_graphs.h5` file with structure:

```
/graphs/
  /t_0000/                      # Frame 0
    /node_features [N_total, 8]
    /adjacency_matrix [N_total, N_total]
    /node_mask [N_total]
    /edge_mask [N_total, N_total]
  /t_0001/                      # Frame 1
    ...
  /t_0029/                      # Frame 29 (last frame)
    ...
/summary/
  /vehicle_id_mapping           # Maps vehicle ID → matrix index
  /vehicle_class_ids [N_total]  # Class ID for each vehicle
  /attributes                   # Statistics (total_frames, total_vehicles, etc.)
```

## Key Concepts

### Fixed-Size Matrices
- All frames have the **same matrix dimensions** (N_total × N_total)
- This enables efficient batching for neural networks
- Missing vehicles are represented by `-1` values, not by changing matrix size

### Complete Graph Structure
- Every vehicle that appears in the video is a node in every frame
- All vehicle pairs have an edge (distance) when both are present
- This captures all possible spatial relationships

### Temporal Consistency
- Vehicle IDs remain constant across frames
- Same vehicle always occupies the same row/column index
- Enables tracking vehicle evolution over time

### Missing Value Handling
- `-1` indicates:
  - Vehicle not present in frame (node features)
  - Edge not valid (adjacency matrix)
- Masks provide binary indicators for valid nodes/edges
- Models can use masks to ignore missing data during training

## Example

Consider a video with 3 unique vehicles (A, B, C) across 30 frames:

**Frame 0**: Vehicles A and B present
- `node_features[0]` = [x_A, y_A, ...] (actual values)
- `node_features[1]` = [x_B, y_B, ...] (actual values)
- `node_features[2]` = [-1, -1, ...] (vehicle C absent)
- `adjacency_matrix[0,1]` = distance(A, B)
- `adjacency_matrix[0,2]` = -1 (C absent)
- `adjacency_matrix[1,2]` = -1 (C absent)

**Frame 15**: All vehicles present
- All node features have actual values
- All adjacency entries have distances (complete graph)

**Frame 29**: Only vehicle C present
- `node_features[0]` = [-1, -1, ...] (A absent)
- `node_features[1]` = [-1, -1, ...] (B absent)
- `node_features[2]` = [x_C, y_C, ...] (actual values)
- Only `adjacency_matrix[2,2]` = 0.0 (self-connection)

## Normalization

**Timestamp Normalization**:
- Formula: `normalized_timestamp = frame_index / (total_frames - 1)`
- Frame 0 → 0.0
- Frame 29 → 1.0
- Provides temporal position as a normalized feature

**No Spatial Normalization**:
- Coordinates are in pixel space
- Distances are in pixels
- Models can learn to handle raw pixel values or normalize during training

## Visualization

The pipeline also generates `visualization_video.mp4` showing:
- Green bounding boxes around detected vehicles
- Vehicle IDs and class labels
- Red lines connecting vehicles (edges)
- Distance labels on edges
- Blue trajectory lines showing vehicle movement

## Usage in Training

When loading data for Dynamic Graph Networks:

```python
import h5py
import torch

with h5py.File('temporal_graphs.h5', 'r') as f:
    # Load all 30 frames
    for frame_idx in range(30):
        frame_key = f't_{frame_idx:04d}'
        
        # Node features [N_total, 8]
        nodes = torch.tensor(f[f'graphs/{frame_key}/node_features'][:])
        
        # Adjacency matrix [N_total, N_total]
        adj = torch.tensor(f[f'graphs/{frame_key}/adjacency_matrix'][:])
        
        # Masks for filtering
        node_mask = torch.tensor(f[f'graphs/{frame_key}/node_mask'][:])
        edge_mask = torch.tensor(f[f'graphs/{frame_key}/edge_mask'][:])
        
        # Filter out missing values using masks
        valid_nodes = nodes[node_mask == 1]
        valid_adj = adj[edge_mask == 1]
```

## Summary

The graph data production process:
1. ✅ Detects all vehicles across entire video sequence
2. ✅ Tracks vehicles with consistent IDs
3. ✅ Creates fixed-size feature matrices (all vehicles, -1 for missing)
4. ✅ Builds complete adjacency matrices (all pairs, -1 for missing)
5. ✅ Stores temporal sequence in HDF5 format
6. ✅ Provides masks for filtering valid nodes/edges

This format is optimized for Dynamic Graph Networks that process temporal sequences of graphs with varying node presence.

