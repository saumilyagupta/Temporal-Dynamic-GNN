#!/usr/bin/env python3
"""
Vehicle detection, tracking, and PyG graph generation module.

This module processes videos to detect vehicles, track them across frames,
and generate PyTorch Geometric (PyG) graph data for each frame.

For datasets that already ship bbox tensors (e.g. CarCrash NPZ ``det``), use
``node_feature_based_Temporal_DGNN/niad_graph_from_det.py`` instead of YOLO —
same 8-D node layout as :func:`create_frame_graph`.
"""

import os
import cv2
import json
import torch
import numpy as np
from collections import defaultdict
from datetime import datetime
from scipy.spatial.distance import cdist
from torch_geometric.data import Data
from ultralytics import YOLO


class VehicleTracker:
    """
    Centroid-based vehicle tracker that maintains consistent IDs across frames.
    
    Attributes:
        max_disappeared: Number of frames a vehicle can disappear before being deregistered.
        max_distance: Maximum distance for centroid matching.
    """
    
    def __init__(self, max_disappeared: int = 10, max_distance: float = 100.0):
        self.next_id = 0
        self.objects = {}  # object_id -> centroid
        self.disappeared = {}  # object_id -> frames_disappeared
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.trajectories = defaultdict(list)  # object_id -> list of centroids
        
    def register(self, centroid: np.ndarray) -> int:
        """Register a new object with the next available ID."""
        object_id = self.next_id
        self.objects[object_id] = centroid
        self.disappeared[object_id] = 0
        self.trajectories[object_id].append(centroid.tolist())
        self.next_id += 1
        return object_id
        
    def deregister(self, object_id: int) -> None:
        """Remove an object from tracking."""
        del self.objects[object_id]
        del self.disappeared[object_id]
        
    def update(self, rects: list) -> tuple:
        """
        Update tracker with new detections.
        
        Args:
            rects: List of bounding boxes (x, y, w, h)
            
        Returns:
            Tuple of (objects dict, assignments dict mapping object_id -> detection_index)
        """
        assignments = {}

        if len(rects) == 0:
            # No detections - mark all as disappeared
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects, assignments

        # Calculate centroids for input rectangles
        input_centroids = np.zeros((len(rects), 2), dtype=np.float32)
        for i, (x, y, w, h) in enumerate(rects):
            cx = x + w / 2.0
            cy = y + h / 2.0
            input_centroids[i] = (cx, cy)

        if len(self.objects) == 0:
            # No existing objects - register all detections
            for i in range(len(input_centroids)):
                new_id = self.register(input_centroids[i])
                assignments[new_id] = i
        else:
            # Match existing objects to new detections
            object_ids = list(self.objects.keys())
            object_centroids = np.array(list(self.objects.values()))
            
            # Compute distance matrix
            D = cdist(object_centroids, input_centroids)
            
            # Sort by minimum distance
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_row_indices = set()
            used_col_indices = set()

            for row, col in zip(rows, cols):
                if row in used_row_indices or col in used_col_indices:
                    continue

                if D[row, col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                self.trajectories[object_id].append(input_centroids[col].tolist())

                assignments[object_id] = col
                used_row_indices.add(row)
                used_col_indices.add(col)

            # Handle unmatched existing objects
            unused_row_indices = set(range(D.shape[0])).difference(used_row_indices)
            unused_col_indices = set(range(D.shape[1])).difference(used_col_indices)

            if D.shape[0] >= D.shape[1]:
                # More existing objects than detections
                for row in unused_row_indices:
                    if row < len(object_ids):
                        object_id = object_ids[row]
                        self.disappeared[object_id] += 1
                        if self.disappeared[object_id] > self.max_disappeared:
                            self.deregister(object_id)
            else:
                # More detections than existing objects - register new ones
                for col in unused_col_indices:
                    new_id = self.register(input_centroids[col])
                    assignments[new_id] = col

        return self.objects, assignments


def create_frame_graph(
    frame_data: dict,
    frame_idx: int,
    total_frames: int = 30
) -> Data:
    """
    Create a PyG Data object for a single frame.
    
    Args:
        frame_data: Dict mapping vehicle_id -> {centroid, bbox, class_id}
        frame_idx: Current frame index
        total_frames: Total number of frames for normalization
        
    Returns:
        PyG Data object with x, edge_index, edge_weight, node_ids, t
    """
    vehicle_ids = sorted(frame_data.keys())
    num_nodes = len(vehicle_ids)
    
    if num_nodes == 0:
        # Empty graph for frames with no vehicles
        return Data(
            x=torch.zeros((0, 8), dtype=torch.float32),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_weight=torch.zeros((0,), dtype=torch.float32),
            node_ids=torch.zeros((0,), dtype=torch.long),
            t=torch.tensor([frame_idx], dtype=torch.long)
        )
    
    # Normalized timestamp
    normalized_t = frame_idx / (total_frames - 1) if total_frames > 1 else 0.0
    
    # Build node features [num_nodes, 8]
    # Features: centroid_x, centroid_y, bbox_x, bbox_y, bbox_w, bbox_h, class_id, normalized_timestamp
    node_features = []
    node_id_list = []
    centroids = []
    
    for vid in vehicle_ids:
        data = frame_data[vid]
        centroid = data['centroid']
        bbox = data['bbox']
        class_id = data['class_id']
        
        features = [
            float(centroid[0]),  # centroid_x
            float(centroid[1]),  # centroid_y
            float(bbox[0]),      # bbox_x
            float(bbox[1]),      # bbox_y
            float(bbox[2]),      # bbox_w
            float(bbox[3]),      # bbox_h
            float(class_id),     # class_id
            float(normalized_t)  # normalized_timestamp
        ]
        node_features.append(features)
        node_id_list.append(vid)
        centroids.append(centroid)
    
    x = torch.tensor(node_features, dtype=torch.float32)
    node_ids = torch.tensor(node_id_list, dtype=torch.long)
    
    # Build edges (fully connected graph)
    if num_nodes > 1:
        # Create all pairs of edges (directed, both directions)
        edge_list = []
        edge_weights = []
        
        centroids_array = np.array(centroids, dtype=np.float32)
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_list.append([i, j])
                    # Euclidean distance between centers
                    dist = float(np.linalg.norm(centroids_array[i] - centroids_array[j]))
                    edge_weights.append(dist)
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weights, dtype=torch.float32)
    else:
        # Single node - no edges
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_weight = torch.zeros((0,), dtype=torch.float32)
    
    return Data(
        x=x,
        edge_index=edge_index,
        edge_weight=edge_weight,
        node_ids=node_ids,
        t=torch.tensor([frame_idx], dtype=torch.long)
    )


def process_video_to_graphs(
    video_path: str,
    output_folder: str,
    clip_name: str,
    yolo_model: str = 'yolov8n.pt',
    detection_conf: float = 0.3,
    vehicle_classes: list = None,
    frames_per_video: int = 30,
    device: str = 'cuda:0',
    generate_visualization: bool = False
) -> dict:
    """
    Process a video to generate PyG graphs for each frame.
    
    Args:
        video_path: Path to input video file
        output_folder: Directory to save output graphs
        clip_name: Name of the video clip
        yolo_model: Path to YOLO model weights
        detection_conf: Detection confidence threshold
        vehicle_classes: List of YOLO class IDs for vehicles
        frames_per_video: Number of frames to process
        device: Device for YOLO inference
        generate_visualization: Whether to generate visualization video
        
    Returns:
        Dict with processing results including paths and statistics
    """
    if vehicle_classes is None:
        vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck

    # Initialize YOLO model
    model = YOLO(yolo_model)
    model.model.to(device)

    # Initialize tracker
    tracker = VehicleTracker()

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_width == 0 or frame_height == 0:
        raise ValueError(f"Video has invalid dimensions: {video_path}")

    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Setup visualization video writer if requested
    vis_video_path = None
    out = None
    if generate_visualization:
        vis_video_path = os.path.join(output_folder, "visualization_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(vis_video_path, fourcc, fps, (frame_width, frame_height))

    # Process frames
    frame_records = []
    vehicle_class_map = {}
    all_vehicle_ids = set()
    frame_idx = 0

    while frame_idx < frames_per_video:
        ret, frame = cap.read()
        if not ret:
            # Video ended early - pad with empty frames
            print(f"  Warning: Video {clip_name} ended at frame {frame_idx}, padding to {frames_per_video} frames")
            while frame_idx < frames_per_video:
                frame_records.append({})
                if out is not None:
                    black_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
                    out.write(black_frame)
                frame_idx += 1
            break

        # Run YOLO detection
        results = model(frame, classes=vehicle_classes, conf=detection_conf, verbose=False, device=device)

        # Parse detections
        detections = []
        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < detection_conf:
                    continue

                class_id = int(box.cls[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                cx = int(x + w / 2.0)
                cy = int(y + h / 2.0)

                detections.append({
                    'bbox': (x, y, w, h),
                    'centroid': (cx, cy),
                    'class_id': class_id,
                    'confidence': conf,
                })

        # Update tracker
        boxes = [det['bbox'] for det in detections]
        objects, assignments = tracker.update(boxes)

        # Build frame data
        frame_vehicle_data = {}
        for object_id, det_idx in assignments.items():
            det = detections[det_idx]
            centroid = det['centroid']
            bbox = det['bbox']
            class_id = det['class_id']

            if object_id not in vehicle_class_map:
                vehicle_class_map[object_id] = class_id

            vehicle_class = vehicle_class_map[object_id]
            frame_vehicle_data[object_id] = {
                'centroid': centroid,
                'bbox': bbox,
                'class_id': vehicle_class,
            }
            all_vehicle_ids.add(object_id)

        frame_records.append(frame_vehicle_data)

        # Generate visualization if requested
        if out is not None:
            vis_frame = frame.copy()
            
            # Draw detections
            for object_id, data in frame_vehicle_data.items():
                x, y, w, h = data['bbox']
                cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(vis_frame, f"ID:{object_id}", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(vis_frame, f"C:{data['class_id']}", (x, y + h + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # Draw edges between vehicles
            present_ids = list(frame_vehicle_data.keys())
            if len(present_ids) > 1:
                for i in range(len(present_ids)):
                    id1 = present_ids[i]
                    c1 = frame_vehicle_data[id1]['centroid']
                    for j in range(i + 1, len(present_ids)):
                        id2 = present_ids[j]
                        c2 = frame_vehicle_data[id2]['centroid']
                        dist = float(np.linalg.norm(np.array(c1) - np.array(c2)))
                        cv2.line(vis_frame, c1, c2, (0, 0, 255), 2)
                        mid = ((c1[0] + c2[0]) // 2, (c1[1] + c2[1]) // 2)
                        cv2.putText(vis_frame, f"{dist:.1f}", mid,
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            # Draw trajectories
            for obj_id, trajectory in tracker.trajectories.items():
                if len(trajectory) > 1:
                    for k in range(1, len(trajectory)):
                        pt1 = (int(trajectory[k-1][0]), int(trajectory[k-1][1]))
                        pt2 = (int(trajectory[k][0]), int(trajectory[k][1]))
                        cv2.line(vis_frame, pt1, pt2, (255, 0, 0), 2)

            out.write(vis_frame)

        frame_idx += 1

    cap.release()
    if out is not None:
        out.release()

    # Ensure we have exactly frames_per_video frames
    if len(frame_records) < frames_per_video:
        while len(frame_records) < frames_per_video:
            frame_records.append({})
    elif len(frame_records) > frames_per_video:
        frame_records = frame_records[:frames_per_video]

    # Create and save PyG graphs for each frame
    graph_paths = []
    nodes_per_frame = []
    
    for t, frame_data in enumerate(frame_records):
        graph = create_frame_graph(frame_data, t, frames_per_video)
        graph_path = os.path.join(output_folder, f"graph_{t:03d}.pt")
        torch.save(graph, graph_path)
        graph_paths.append(graph_path)
        nodes_per_frame.append(graph.x.shape[0])

    # Create metadata
    metadata = {
        'clip_name': clip_name,
        'video_path': video_path,
        'creation_time': datetime.now().isoformat(),
        'total_frames': len(frame_records),
        'total_unique_vehicles': len(all_vehicle_ids),
        'vehicle_ids': sorted(list(all_vehicle_ids)),
        'vehicle_class_map': {str(k): v for k, v in vehicle_class_map.items()},
        'nodes_per_frame': nodes_per_frame,
        'max_nodes_per_frame': max(nodes_per_frame) if nodes_per_frame else 0,
        'frames_with_vehicles': sum(1 for n in nodes_per_frame if n > 0),
        'fps': fps,
        'frame_width': frame_width,
        'frame_height': frame_height,
        'node_feature_columns': [
            'centroid_x', 'centroid_y',
            'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h',
            'class_id', 'normalized_timestamp'
        ]
    }
    
    metadata_path = os.path.join(output_folder, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return {
        'output_folder': output_folder,
        'graph_paths': graph_paths,
        'metadata_path': metadata_path,
        'visualization_video': vis_video_path,
        'num_vehicles': len(all_vehicle_ids),
        'num_frames': len(frame_records),
        'nodes_per_frame': nodes_per_frame
    }

