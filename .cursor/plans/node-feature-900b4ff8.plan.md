<!-- 900b4ff8-d75c-456a-9d0e-cd041b62c4e4 926808e0-399c-4ddc-9176-939ce842d8ca -->
# Node Feature-Based Data Pipeline Implementation

## Goal

Convert videos from `accident_prediction/NiAD_Large_Videos` into PyG-compatible temporal graph sequences where each frame is a separate graph with only present vehicles as nodes.

## Key Changes from Old Pipeline

- **Storage**: Individual `.pt` files per frame instead of single HDF5
- **Nodes**: Only vehicles present in frame (no padding)
- **Edges**: `edge_index` + `edge_weight` (COO sparse format)
- **Output**: 30 `graph_XXX.pt` files + `metadata.json` per video

## Files to Create in `data_pipeline_node_feature_based/`

### 1. `requirements.txt`

Dependencies: torch, torch-geometric, ultralytics, opencv-python, numpy, scipy

### 2. `utils.py`

Reuse from existing pipeline with minor adaptations:

- `discover_niaad_videos()` - discover videos from NiAD structure
- `ensure_dir()`, `ProgressLogger`, `format_time()` - utility functions

### 3. `vehicle_processor.py`

Core processing module (adapted from [data_pipeline/vehicle_processor.py](data_pipeline/vehicle_processor.py)):

- `VehicleTracker` class - centroid-based tracking (reuse as-is)
- `process_video_to_graphs()` - main processing function
  - Run YOLO detection per frame
  - Track vehicles across frames
  - For each frame, create PyG `Data` object:
    - `x`: [num_nodes_t, 8] node features
    - `edge_index`: [2, num_edges_t] fully connected
    - `edge_weight`: [num_edges_t] Euclidean distances
    - `node_ids`: [num_nodes_t] tracker IDs
    - `t`: frame index
  - Save each frame as `graph_XXX.pt`
  - Save `metadata.json` with summary

### 4. `data_splitter.py`

Organize processed data:

- `organize_processed_clip()` - move graphs to final location
- `create_manifest_csv()` - create split manifests
- Updated to handle new folder structure

### 5. `main.py`

Main pipeline orchestrator:

- Configuration for paths, YOLO params, device
- Discover videos -> Process -> Organize -> Create manifests
- Output structure:
```
output_folder/
  train/Normal/video_name/graph_000.pt...graph_029.pt, metadata.json
  train/Anomalous/video_name/...
  val/...
  test/...
  train_manifest.csv, val_manifest.csv, test_manifest.csv
```


### 6. `visualizer.py` (Optional)

Visualization utilities for debugging:

- Draw bounding boxes, edges, trajectories on frames
- Generate visualization video

## Output Graph Format

```python
Data(
    x: [num_nodes_t, 8],       # centroid_x/y, bbox_x/y/w/h, class_id, normalized_t
    edge_index: [2, num_edges],# fully connected edges
    edge_weight: [num_edges],  # Euclidean distance between centers
    node_ids: [num_nodes_t],   # tracker IDs for temporal consistency
    t: [1]                     # frame index
)
```

### To-dos

- [ ] Create requirements.txt with dependencies
- [ ] Create utils.py with discovery and helper functions
- [ ] Create vehicle_processor.py with VehicleTracker and PyG graph generation
- [ ] Create data_splitter.py for organizing outputs and manifests
- [ ] Create main.py pipeline orchestrator
- [ ] Create visualizer.py for optional debug visualization