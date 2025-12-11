# Graph Neural Networks for Video Anomaly Detection

A comprehensive research project on using Temporal Dynamic Graph Neural Networks (Temporal DGNN) for detecting anomalies in traffic surveillance videos.

## 📋 Project Overview

This project implements a Temporal Graph Neural Network that processes sequences of per-frame graphs using spatial GCN layers followed by temporal GRU modeling. The system converts 30-frame surveillance videos into dynamic graph sequences where nodes represent vehicles and edges represent spatial relationships.

**Key Results:**
- Test Accuracy: **89.32%**
- AUC-ROC: **0.81**
- Dataset: NiAD_large_graphs (1,176 videos)

## 🏗️ Project Structure

```
GNN-Research/
├── data_pipeline_node_feature_based/    # Node-feature-based data processing pipeline
│   ├── main.py                          # Main pipeline script
│   ├── vehicle_detector.py              # YOLOv8 vehicle detection
│   ├── vehicle_tracker.py               # Centroid-based tracking
│   ├── graph_builder.py                 # PyG graph construction
│   └── README.md                        # Pipeline documentation
│
├── node_feature_based_Temporal_DGNN/    # Temporal DGNN model implementation
│   ├── model.py                         # Temporal DGNN architecture
│   ├── dataset.py                       # PyG dataset loader
│   ├── train.py                         # Training script
│   ├── config.yaml                      # Training configuration
│   └── experiments/                     # Experiment results
│
├── report/                              # LaTeX technical report
│   ├── main.tex                         # Main report document
│   ├── chapters/                        # Report chapters
│   │   ├── introduction.tex
│   │   ├── background.tex
│   │   ├── data_pipeline.tex
│   │   ├── model_dynamic_gnn.tex
│   │   ├── experiments.tex
│   │   └── conclusion.tex
│   └── figures/                         # TikZ diagrams
│
├── NiAD_graphs_node_feature_based/     # Processed graph dataset
│   ├── train/                          # Training graphs
│   ├── val/                            # Validation graphs
│   ├── test/                           # Test graphs
│   └── *_manifest.csv                  # Dataset manifests
│
└── README.md                           # This file
```

## 🚀 Quick Start

### 1. Data Pipeline

Process raw videos into PyTorch Geometric graph sequences:

```bash
cd data_pipeline_node_feature_based/
python main.py
```

**Configuration:**
- Input: Videos organized by split (Training/Validation/Testing) and class (Normal/Accident)
- Output: 30 `.pt` files per video (one per frame) + CSV manifests
- Detection: YOLOv8n with confidence threshold 0.3
- Vehicle classes: Car (2), Motorcycle (3), Bus (5), Truck (7)

### 2. Model Training

Train the Temporal DGNN model:

```bash
cd node_feature_based_Temporal_DGNN/
python train.py --config config.yaml
```

**Model Architecture:**
- Spatial: GCNConv(8, 64) → GCNConv(64, 64)
- Temporal: GRU(64, 128)
- Classifier: Linear(128, 2)
- Loss: Focal Loss (α=0.758, γ=1.0)

### 3. Evaluation

Evaluate on test set:

```bash
python evaluate.py --checkpoint experiments/exp_*/best_model.ckpt
```

## 📊 Dataset: NiAD_large_graphs

**Statistics:**
- **Total videos:** 1,176
- **Training:** 842 (638 Normal + 204 Anomalous)
- **Validation:** 128 (112 Normal + 16 Anomalous)
- **Test:** 206 (184 Normal + 22 Anomalous)
- **Frames per video:** 30 (fixed)
- **Node features:** 8 dimensions per vehicle

**Node Features:**
1. centroid_x, centroid_y
2. bbox_x, bbox_y, bbox_w, bbox_h
3. class_id (YOLO class)
4. timestamp (normalized frame index)

**Data Augmentation:**
- Applied only to anomalous training samples
- Strategies: Brightness, Horizontal Flip, Spatial Shift
- 4× augmentation: 51 original → 204 total anomalous videos

## 🔬 Model Details

### Temporal DGNN Architecture

1. **Per-frame Processing:** Each frame's graph is processed by 2 GCN layers
2. **Node Alignment:** Nodes sorted by `node_ids` for temporal consistency
3. **Mean Pooling:** Frame-level embeddings via mean pooling
4. **Temporal Modeling:** GRU processes sequence of 30 frame embeddings
5. **Classification:** Final hidden state → binary classifier

### Training Configuration

- **Optimizer:** Adam (lr=1e-3, weight_decay=1e-5)
- **Batch Size:** 8
- **Max Epochs:** 100
- **Gradient Clipping:** 1.0
- **Early Stopping:** Patience=15 epochs
- **Loss Function:** Focal Loss (handles class imbalance)

## 📈 Results

**Test Set Performance:**
- Accuracy: 89.32%
- Precision (Anomalous): 50.00%
- Recall (Anomalous): 22.73%
- F1-Score (Anomalous): 31.25%
- AUC-ROC: 0.8116
- Specificity: 97.28%

**Confusion Matrix:**
```
                Predicted
              Normal  Anomalous
Actual Normal   179       5
      Anomalous  17       5
```

## 📝 Technical Report

A comprehensive 30-page LaTeX report is available in `report/`:

```bash
cd report/
pdflatex main.tex
```

The report includes:
- Mathematical foundations of GNNs
- Complete data pipeline documentation
- Model architecture details
- Experimental results and analysis
- Future work directions

## 🛠️ Requirements

### Python Packages
```
torch>=2.0.0
torch-geometric
torchvision
pytorch-lightning
ultralytics  # YOLOv8
opencv-python
numpy
pandas
scipy
scikit-learn
```

### LaTeX (for report)
```
texlive-latex-base
texlive-latex-extra
texlive-fonts-recommended
```

## 📚 Key Features

- **Dynamic Graph Representation:** Variable node counts per frame (no padding)
- **Clean Data Format:** PyTorch Geometric `.pt` files (30 per video)
- **Temporal Consistency:** Node ID tracking across frames
- **Class Imbalance Handling:** Focal Loss + data augmentation
- **Efficient Storage:** Sparse graph representation

## 🔍 Data Pipeline Flow

1. **Video Input:** 30-frame video clips
2. **Vehicle Detection:** YOLOv8 detects vehicles in each frame
3. **Vehicle Tracking:** Centroid-based tracking maintains IDs
4. **Graph Construction:** Per-frame graphs with:
   - Nodes: Detected vehicles (8 features each)
   - Edges: Fully connected (Euclidean distance weights)
5. **Storage:** PyG Data objects saved as `.pt` files

## 📖 Citation

If you use this work, please cite:

```bibtex
@techreport{gnn_video_anomaly_2025,
  title={Graph Neural Networks for Video Anomaly Detection},
  author={Gupta, Saumilya},
  institution={The LNM Institute of Information Technology, Jaipur},
  year={2025},
  note={Mini Project Report}
}
```

## 👥 Authors

- **Saumilya Gupta** (23UEC618)
- **Supervisor:** Dr. Preety Singh
- **Institution:** The LNM Institute of Information Technology, Jaipur
- **Department:** Computer Science and Engineering

## 📄 License

Copyright © The LNMIIT 2025. All rights reserved.

## 🔗 References

- Kipf & Welling (2017): Graph Convolutional Networks
- Lin et al. (2017): Focal Loss for Dense Object Detection
- Ultralytics: YOLOv8
- PyTorch Geometric: Fast Graph Representation Learning

## 📧 Contact

For questions or issues, please refer to the technical report in `report/main.pdf`.

---

**Note:** Dataset files are excluded from the repository (see `.gitignore`). Please ensure you have access to the NiAD dataset before running the pipeline.
