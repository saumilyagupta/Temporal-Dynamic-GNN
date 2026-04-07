# BBox-only CarCrash graph pipeline

Exports **PyTorch Geometric** per-frame graphs from CarCrash `.npz` files using only the **`det`** bounding-box tensor (no VGG16, no inference).

## NPZ schema

Each clip (e.g. `positive/000001.npz`) contains:

| Key | Shape | Description |
|-----|-------|-------------|
| `det` | `(50, 19, 6)` | `x1, y1, x2, y2, conf, cls` per slot |
| `labels` | one-hot | Optional; training labels come from `train.txt` / `test.txt` |

The pipeline **does not** read `data` (VGG features).

## Node features (13-D)

Per valid box (non-degenerate `x1,y1,x2,y2`):

`cx, cy, w, h, area, aspect_ratio, x1, y1, x2, y2, conf, cls, normalized_t`

## Edges

Fully connected **directed** edges (no self-loops).

- `edge_strategy: centroid_dist` — `edge_weight` = Euclidean distance between centroids (default).
- `edge_strategy: iou` — `edge_weight` = `1 - IoU`.
- `edge_strategy: both` — `edge_weight` = centroid distance; `edge_attr` = IoU per edge (`[E, 1]`).

## Splits

- **`test.txt`** → exported under `test/` (unchanged).
- **`train.txt`** → stratified **80% / 20%** (default) into `train/` and `val/` using `sklearn.model_selection.train_test_split` with `random_state=42`. Stratification is used when each class appears at least twice in `train.txt`; otherwise an unstratified split is used.

## Output layout

```
{output_root}/
├── train/{Normal|Anomalous}/{video_id}/graph_000.pt … graph_049.pt
├── val/{Normal|Anomalous}/{video_id}/…
├── test/{Normal|Anomalous}/{video_id}/…
├── train_manifest.csv
├── val_manifest.csv
└── test_manifest.csv
```

Manifest columns: `video_folder_path`, `label`, `source_video`, `num_frames`, `max_nodes_per_frame`, `node_feature_dim`.

## Usage

From this directory (so imports resolve):

```bash
cd Temporal-Dynamic-GNN/data_pipeline_bbox_only

# Defaults (see config.yaml)
python export_pipeline.py

# Custom paths and 8 parallel workers
python export_pipeline.py \
  --npz_root /path/to/CarCrash/vgg16_features \
  --output_root /path/to/carcrash_bbox_graphs \
  --num_frames 50 \
  --val_split 0.2 \
  --seed 42 \
  --workers 8 \
  --edge_strategy centroid_dist

# YAML overrides
python export_pipeline.py --config config.yaml
```

Verify a completed export:

```bash
python verify_export.py --export_root /path/to/carcrash_bbox_graphs --sample_size 5
```

## Dependencies

- `numpy`, `torch`, `torch-geometric`
- `scikit-learn` (train/val split)
- `tqdm`
- `PyYAML` (optional, for `--config`)

## Training note

The Temporal DGNN model in `node_feature_based_Temporal_DGNN` defaults to `input_dim=8` for NiAD-style features. For this pipeline’s **13-D** nodes, set `input_dim: 13` (or add a projection layer) in your training config when using these exports.
