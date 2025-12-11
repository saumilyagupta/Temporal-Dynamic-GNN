
# 📘 **3D Dynamic Graph Neural Network — Final Documentation**

This document describes the **dynamic graph dataset**, **graph generation pipeline**, and the **TemporalGNN model** used for anomaly detection in 30-frame traffic videos.

---

# 🚀 **1. Dataset Overview**

**Dataset Path:**

```
/workspace/saumilya/GNN-Research/NiAD_Large_Videos_processed_graphs
```

Each processed video clip contains:

* raw 30-frame video
* visualization video (bounding boxes + edges drawn)
* a sequence of **30 PyTorch Geometric graphs**, one per frame

---

# 📂 **2. Directory Structure**

```
NiAD_Large_Videos_processed_graphs/
├── train/
│   ├── Normal/
│   │   └── clip_folder/
│   │       ├── raw_video.mp4
│   │       ├── visualization_video.mp4
│   │       └── graph_sequence/
│   │            ├── graph_000.pt
│   │            ├── graph_001.pt
│   │            └── graph_029.pt
│   └── Anomalous/
├── val/
├── test/
├── train_manifest.csv
├── val_manifest.csv
└── test_manifest.csv
```

---

# 📝 **3. Manifest CSV Format**

Each row contains:

| Column              | Description               |
| ------------------- | ------------------------- |
| `video_folder_path` | Folder of that video clip |
| `label`             | Normal / Anomalous        |
| `source_video`      | Original file name        |
| `num_vehicles`      | Total tracked vehicle IDs |
| `num_frames`        | Always 30                 |

---

# 🔍 **4. Graph Generation Pipeline**

Each video is converted into **30 graphs**.
Each graph contains **only nodes present in that frame**.

This removes earlier issues with:

* fixed-size matrices
* –inf padding
* large negative constants
* heavy adjacency computation

### ✔️ Each frame is now a standalone dynamic graph.

---

# 📦 **5. Final Data Object (Per Frame)**

Each frame graph is a PyTorch Geometric `Data` object:

```python
Data(
    x: [num_nodes_t, 8],            
    edge_index: [2, num_edges_t],   
    edge_attr: [num_edges_t, d],    
    edge_weight: [num_edges_t],     
    node_ids: [num_nodes_t],        
    t: [1]                          
)
```

---

# 🔢 **6. Node Features (8-D)**

For each vehicle detected in the frame:

1. centroid_x
2. centroid_y
3. bbox_x
4. bbox_y
5. bbox_w
6. bbox_h
7. YOLO class ID (2,3,5,7)
8. normalized_timestamp = frame_index / 29

Only active vehicles appear in the graph.
No padding, no missing values, no –inf.

---

# 🔗 **7. Edges**

### `edge_index`

All valid edges between present nodes.

### `edge_attr`

Edge descriptors such as:

* Euclidean distance
* Bearing angle
* IoU (optional)
* Speed difference (optional)

### `edge_weight`

A scalar typically computed as:

```
1 / (distance + eps)
```

Used in GCNConv.

---

# 🆔 **8. node_ids — Temporal Identity**

`node_ids` = YOLO tracking IDs.

They ensure:

* the same vehicle keeps the same identity across frames
* node embeddings can be aligned temporally
* GRU sees per-vehicle progression consistently

### ⚠️ **Important:**

Node IDs are **used inside the model** for ordering and alignment.
They are **NOT treated as numeric features**.

---

# 📥 **9. Loading Graph Sequence**

```python
import torch

graphs = []
for t in range(30):
    g = torch.load(f"{video_path}/graph_sequence/graph_{t:03d}.pt")
    graphs.append(g)
```

The model receives a **list of 30 graphs**.

---

# 🧠 **10. TemporalGNN Model (Final Version with node_ids)**

```python
class TemporalGNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Spatial GNN
        self.conv1 = GCNConv(8, 64)
        self.conv2 = GCNConv(64, 64)

        # Temporal model
        self.gru = nn.GRU(64, 128)

        # Final classifier
        self.fc = nn.Linear(128, 1)

    def forward(self, graph_sequence):
        frame_embeddings = []

        for data in graph_sequence:
            # --- Identity alignment using node_ids ---
            sorted_idx = torch.argsort(data.node_ids)
            x = data.x[sorted_idx]
            node_ids = data.node_ids[sorted_idx]

            # Spatial GNN
            h = F.relu(self.conv1(x, data.edge_index, data.edge_weight))
            h = F.relu(self.conv2(h, data.edge_index, data.edge_weight))

            # Frame embedding (mean pooling)
            frame_emb = h.mean(dim=0)
            frame_embeddings.append(frame_emb)

        # Convert list → temporal tensor
        H = torch.stack(frame_embeddings).unsqueeze(1)  # [30, 1, 64]

        # Temporal reasoning
        _, h_final = self.gru(H)

        # Final prediction
        return self.fc(h_final.squeeze(0))
```

### ✔️ Notes

* Identity sorting ensures node `k` at t=0 aligns with node `k` at t=29.
* Prevents mixing up vehicle identities.
* The model uses x, edge_index, edge_weight, and node_ids to build consistent temporal patterns.

---

# ⚖️ **11. Loss Function**

Since:

* training set is **balanced**
* val/test sets are **highly imbalanced**

Use:

### **Binary Cross-Entropy**

or

### **Focal Loss**

Focal Loss parameters:

* γ = 1.0
* α = num_normal / (num_normal + num_anomalous)

---

# 🧾 **12. Summary**

| Component     | Description                                     |
| ------------- | ----------------------------------------------- |
| Graph format  | Dynamic PyG graphs (no padding)                 |
| Node identity | Handled using node_ids                          |
| Model         | GCN → GRU → FC                                  |
| Edges         | Fully dynamic, weighted                         |
| Input         | List of 30 graphs                               |
| Loss          | BCE or Focal Loss                               |
| Improvement   | Removed –inf values and huge negative constants |

### ✔️ This is the **final, clean, optimized DGNN pipeline**.


