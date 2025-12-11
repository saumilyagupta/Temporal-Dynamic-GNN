# GNN Research Experiments Technical Report

A comprehensive LaTeX report documenting Graph Neural Networks for Video Anomaly Detection.

## Report Structure

```
report/
├── main.tex                    # Main document (compile this)
├── chapters/
│   ├── introduction.tex        # Chapter 1: Introduction
│   ├── background.tex          # Chapter 2: Mathematical Background
│   ├── data_pipeline.tex       # Chapter 3: Data Pipeline
│   ├── model_gcn3d.tex         # Chapter 4: 3D GCN Model
│   ├── model_dynamic_gnn.tex   # Chapter 5: Dynamic GNN Model
│   ├── experiments.tex         # Chapter 6: Experiments & Results
│   └── conclusion.tex          # Chapter 7: Conclusion
├── figures/
│   └── [24 TikZ diagrams]      # All diagrams in TikZ format
├── tables/                     # (empty, tables are inline)
└── README.md                   # This file
```

## Compilation

### Using pdflatex (Recommended)

```bash
cd /workspace/saumilya/GNN-Research/report
pdflatex main.tex
pdflatex main.tex  # Run twice for TOC and references
```

### Using latexmk

```bash
cd /workspace/saumilya/GNN-Research/report
latexmk -pdf main.tex
```

### Using Overleaf

1. Upload all files maintaining the folder structure
2. Set `main.tex` as the main document
3. Compile

## Required LaTeX Packages

The report uses the following packages (typically included in TeX Live Full):

- `amsmath`, `amssymb`, `amsfonts` - Math
- `tikz`, `pgfplots` - Diagrams
- `graphicx` - Graphics
- `booktabs`, `longtable`, `multirow` - Tables
- `algorithm`, `algpseudocode` - Algorithms
- `listings` - Code listings
- `hyperref` - Links
- `geometry`, `fancyhdr` - Page layout

## Content Overview

### Chapter 1: Introduction
- Research problem and motivation
- Two datasets overview (53 vs 1,179 samples)
- Key contributions

### Chapter 2: Mathematical Background
- Graph representation: $G = (V, A, X)$
- GCN formulation with normalized adjacency
- Dynamic graph types (DTDG, CTDG)
- Focal Loss for class imbalance

### Chapter 3: Data Pipeline
- YOLO vehicle detection
- Centroid-based tracking
- Graph construction with concrete samples
- Missing value handling ($-\infty$ → $-10^{-40}$)
- HDF5 storage format
- Data augmentation (4×)

### Chapter 4: 3D GCN Model
- Static spatial-temporal graph approach
- Architecture: 3 GCN layers + pooling + MLP
- Hyperparameter optimization (100 Optuna trials)
- Results: 66.67% → 83.33% validation accuracy

### Chapter 5: Dynamic GNN Model
- Per-frame GCN + GRU temporal modeling
- Masked pooling for missing vehicles
- Focal Loss implementation
- Results: 87.38% test accuracy

### Chapter 6: Experiments & Results
- Comprehensive metrics comparison
- Confusion matrices
- Ablation studies
- Error analysis

### Chapter 7: Conclusion
- Key findings summary
- Limitations
- Future work directions

## TikZ Diagrams (24 Total)

### System-Level
1. `system_architecture.tex` - End-to-end pipeline

### Mathematical
2. `message_passing.tex` - GNN message passing
3. `gcn_operation.tex` - GCN convolution
4. `dynamic_graph_types.tex` - DTDG vs CTDG
5. `focal_loss_curve.tex` - Loss comparison

### Data Pipeline
6. `pipeline_flow.tex` - Processing flowchart
7. `vehicle_detection.tex` - YOLO detection
8. `vehicle_tracking.tex` - Tracking across frames
9. `data_augmentation.tex` - 4× augmentation
10. `graph_construction.tex` - Node/edge creation
11. `sample_graph.tex` - Example graph
12. `missing_value_handling.tex` - $-\infty$ handling
13. `dataset_comparison.tex` - Dataset sizes

### Data Storage
14. `hdf5_structure.tex` - HDF5 file tree
15. `pooling_types.tex` - Mean/Max/Sum pooling
16. `temporal_sequence.tex` - 30-frame sequence

### Model Architecture
17. `gcn3d_architecture.tex` - 3D GCN model
18. `dynamic_gnn_architecture.tex` - Dynamic GNN
19. `gru_module.tex` - GRU temporal module

### Results
20. `training_curves_gcn3d.tex` - Training history
21. `confusion_matrices.tex` - CM comparison
22. `hyperopt_results.tex` - Optuna trials
23. `hyperopt_distribution.tex` - Accuracy distribution
24. `results_comparison.tex` - Metrics bar chart

## Troubleshooting

### Missing packages
Install TeX Live Full or add packages via `tlmgr`:
```bash
tlmgr install pgfplots algorithm2e
```

### Compilation errors
- Run pdflatex twice to resolve references
- Check for missing TikZ libraries
- Verify all chapter files are in `chapters/` folder

### Large file size
The report uses vector TikZ graphics. To reduce size, compile with:
```bash
pdflatex -interaction=nonstopmode main.tex
```




