
# **Graph Data Production Guide (Updated)**

This document explains how temporal graph data is generated from 30-frame video sequences for Dynamic Graph Neural Networks.

## **Overview**

The updated pipeline converts each 30-frame video into a sequence of **30 independent graphs**, where:

* **Nodes** represent vehicles present *in that frame only*
* **node_ids** provide temporal consistency (same vehicle keeps same ID)
* **Edges** represent spatial relationships between vehicles
* **edge_weight** encodes the Euclidean distance between vehicle centers
* Graph sizes vary naturally across frames

This format is computationally efficient and resolves the dynamic-node issue from earlier versions.

---

# **1. Video Input**

* Each video contains **30 frames**
* Class labels come from parent folders (Normal / Accident)
* Dataset is already split into Train / Validation / Test
* **Train set is fully balanced**, but Validation and Test are significantly imbalanced

---

# **2. Vehicle Detection & Tracking**

### **2.1 YOLO Detection**

* YOLOv8 detects vehicles in each frame
* Allowed classes: car(2), motorcycle(3), bus(5), truck(7)
* Confidence threshold: 0.3
* Output per frame:

  * bounding box
  * centroid
  * class ID
  * confidence score

### **2.2 Tracking**

* A centroid-based tracker assigns **consistent IDs** across frames
* Maintains identity even with brief occlusions
* Produces:

  * list of vehicles present per frame
  * persistent ID for each vehicle

---

# **3. Graph Construction (New Format)**

Each frame is converted into a **PyG-compatible graph**:

```
Data(
    x: [num_nodes_t, 8],            # node features
    edge_index: [2, num_edges_t],   # connectivity
    edge_weight: [num_edges_t],     # Euclidean distance between centers
    node_ids: [num_nodes_t],        # YOLO track IDs
    t: [1]                          # frame index
)
```

## **3.1 Node Features (8-dim)**

For each vehicle present in the frame:

1. centroid_x
2. centroid_y
3. bbox_x
4. bbox_y
5. bbox_w
6. bbox_h
7. class_id
8. normalized_timestamp = frame_index / 29

No padding, no missing entries, no –inf.

Nodes **not present** in the frame are **not included**.

---

## **3.2 Edge Construction**

For each pair of nodes in the frame:

### **edge_index**

Directed edges between nodes present in this frame.

### **edge_weight**

Euclidean distance between vehicle centers:

```
edge_weight = Euclidean(center_i, center_j)
```

Shape:

```
[num_edges_t]
```

---

# **4. Data Storage Format**

Each video produces 30 PyTorch Geometric `Data` objects stored as:

```
video_graphs/
  ├── graph_000.pt
  ├── graph_001.pt
  ├── ...
  └── graph_029.pt
```

Each `graph_xxx.pt` contains:

```
x, edge_index, edge_weight, node_ids, t
```

Additionally, a summary file stores:

* mapping of track IDs
* number of vehicles per frame
* total vehicles across sequence
* class label (Normal / Accident)

---

# **5. Key Changes From Old Pipeline**

| Old Pipeline                                     | New Pipeline                       |
| ------------------------------------------------ | ---------------------------------- |
| Fixed-size matrices (N_total × N_total)          | Variable-size graphs per frame     |
| Missing vehicles stored as –inf                  | Missing vehicles simply absent     |
| Adjacency matrix with –inf                       | `edge_index` with valid edges only |
| Very heavy memory + computation                  | Much faster and lightweight        |
| Hard dynamic-node handling                       | Clean node_ids tracking            |
| –inf replaced with huge negative values (-10e40) | No need for invalid values         |

---

# **6. Temporal Consistency**

* `node_ids` ensures same vehicle keeps same identity across frames
* Temporal edges (optional) can be added to connect node_id across t→t+1
* GRU/LSTM/Transformer consumes the 30 embeddings sequentially

---

# **7. Visualization Output**

Optional visualization includes:

* YOLO boxes
* Vehicle ID
* Class label
* Lines showing graph edges
* Distance values
* Trajectory traces across frames

---

# **8. Usage in Training**

Example of loading graph sequence:

```python
import torch

graphs = []
for t in range(30):
    data = torch.load(f"video_graphs/graph_{t:03d}.pt")
    graphs.append(data)
```

Model receives the graph sequence:

```
model(graphs)   # list of 30 Data objects
```

Each graph flows through GNN → temporal model.

---

# **Summary (Short)**

* Each frame = its **own graph**
* Nodes = vehicles present in that frame
* node_ids maintain identity across frames
* edge_index and edge_weight define structure
* No padding, no –inf, no fixed matrices
* Clean, efficient, fully dynamic graph sequences
* Train set balanced. but validation/test imbalanced

This is the final, corrected, optimized data format for your Dynamic Graph Neural Network.

.
