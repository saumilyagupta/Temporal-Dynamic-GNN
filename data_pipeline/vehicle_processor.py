#!/usr/bin/env python3
"""
Vehicle detection, tracking, and graph generation module
"""

import os
import cv2
import numpy as np
import json
import h5py
from collections import defaultdict
from scipy.spatial.distance import cdist
from datetime import datetime
from ultralytics import YOLO


class VehicleTracker:
    def __init__(self, max_disappeared=10, max_distance=100):
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.trajectories = defaultdict(list)
        
    def register(self, centroid):
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.trajectories[self.next_id].append(centroid)
        self.next_id += 1
        
    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
        
    def update(self, rects):
        assignments = {}

        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects, assignments

        input_centroids = np.zeros((len(rects), 2), dtype='int')

        for (i, (x, y, w, h)) in enumerate(rects):
            cx = int(x + w / 2.0)
            cy = int(y + h / 2.0)
            input_centroids[i] = (cx, cy)

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
                assignments[self.next_id - 1] = i
        else:
            object_ids = list(self.objects.keys())
            object_centroids = np.array(list(self.objects.values()))
            D = cdist(object_centroids, input_centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_row_indices = set()
            used_col_indices = set()

            for (row, col) in zip(rows, cols):
                if row in used_row_indices or col in used_col_indices:
                    continue

                if D[row, col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                self.trajectories[object_id].append(input_centroids[col])

                assignments[object_id] = col
                used_row_indices.add(row)
                used_col_indices.add(col)

            unused_row_indices = set(range(0, D.shape[0])).difference(used_row_indices)
            unused_col_indices = set(range(0, D.shape[1])).difference(used_col_indices)

            if D.shape[0] >= D.shape[1]:
                # Handle unused existing objects
                for row in unused_row_indices:
                    if row < len(object_ids):
                        object_id = object_ids[row]
                        self.disappeared[object_id] += 1

                        if self.disappeared[object_id] > self.max_disappeared:
                            self.deregister(object_id)
            else:
                # Handle unused new detections
                for col in unused_col_indices:
                    self.register(input_centroids[col])
                    assignments[self.next_id - 1] = col

        return self.objects, assignments


def process_video_for_vehicles(
    video_path,
    output_folder,
    clip_name,
    yolo_model='yolov8n.pt',
    detection_conf=0.3,
    vehicle_classes=None,
    frames_per_video=30,
    missing_value=-1,
    device='cuda:3',
):
    """
    Process video to detect, track vehicles and create dynamic graph data suitable for
    Dynamic Graph Networks.
    """
    if vehicle_classes is None:
        vehicle_classes = [2, 3, 5, 7]

    # Initialize YOLO model on specified device (CUDA 3)
    import torch
    
    model = YOLO(yolo_model)
    # Move model to the specified device
    model.model.to(device)

    tracker = VehicleTracker()

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

    raw_video_path = video_path
    vis_video_path = os.path.join(output_folder, "visualization_video.mp4")
    h5_path = os.path.join(output_folder, "temporal_graphs.h5")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(vis_video_path, fourcc, fps, (frame_width, frame_height))

    frame_records = []
    vehicle_class_map = {}
    all_vehicle_ids = set()
    frame_idx = 0

    # Process exactly frames_per_video frames (30 frames)
    while frame_idx < frames_per_video:
        ret, frame = cap.read()
        if not ret:
            # Video ended early - pad with empty frames
            print(f"  Warning: Video {clip_name} ended at frame {frame_idx}, padding to {frames_per_video} frames")
            while frame_idx < frames_per_video:
                # Add empty frame (no vehicles)
                frame_records.append({})
                # Write a black frame to visualization video
                black_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
                out.write(black_frame)
                frame_idx += 1
            break

        results = model(frame, classes=vehicle_classes, conf=detection_conf, verbose=False, device=device)

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

        boxes = [det['bbox'] for det in detections]
        objects, assignments = tracker.update(boxes)

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

            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID:{object_id}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"C:{vehicle_class}",
                (x, y + h + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )

        present_ids = list(frame_vehicle_data.keys())
        if len(present_ids) > 1:
            for i in range(len(present_ids)):
                id1 = present_ids[i]
                centroid1 = frame_vehicle_data[id1]['centroid']
                for j in range(i + 1, len(present_ids)):
                    id2 = present_ids[j]
                    centroid2 = frame_vehicle_data[id2]['centroid']
                    distance = float(np.linalg.norm(np.array(centroid1) - np.array(centroid2)))
                    cv2.line(frame, centroid1, centroid2, (0, 0, 255), 2)
                    mid_point = (
                        (centroid1[0] + centroid2[0]) // 2,
                        (centroid1[1] + centroid2[1]) // 2,
                    )
                    cv2.putText(
                        frame,
                        f"{distance:.1f}",
                        mid_point,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 0, 255),
                        1,
                    )

        for obj_id, trajectory in tracker.trajectories.items():
            if len(trajectory) > 1:
                for i in range(1, len(trajectory)):
                    cv2.line(frame, tuple(trajectory[i - 1]), tuple(trajectory[i]), (255, 0, 0), 2)

        out.write(frame)
        frame_records.append(frame_vehicle_data)
        frame_idx += 1

    # Ensure we have exactly frames_per_video frames
    if len(frame_records) < frames_per_video:
        # Pad with empty frames if needed
        print(f"  Warning: Video {clip_name} has only {len(frame_records)} frames, padding to {frames_per_video}")
        while len(frame_records) < frames_per_video:
            frame_records.append({})
            black_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            out.write(black_frame)
    elif len(frame_records) > frames_per_video:
        # Truncate if more than expected
        print(f"  Warning: Video {clip_name} has {len(frame_records)} frames, truncating to {frames_per_video}")
        frame_records = frame_records[:frames_per_video]

    cap.release()
    out.release()

    num_frames = len(frame_records)
    vehicle_ids = sorted(all_vehicle_ids)

    # Final check - should always be frames_per_video now
    if num_frames != frames_per_video:
        raise ValueError(
            f"Frame count mismatch: Expected {frames_per_video} frames but got {num_frames} for {clip_name}"
        )

    normalized_timestamps = [
        (idx / (num_frames - 1)) if num_frames > 1 else 0.0 for idx in range(num_frames)
    ]

    save_temporal_graphs_h5(
        frame_records=frame_records,
        vehicle_ids=vehicle_ids,
        vehicle_class_map=vehicle_class_map,
        normalized_timestamps=normalized_timestamps,
        video_path=video_path,
        h5_path=h5_path,
        clip_name=clip_name,
        fps=fps,
        missing_value=missing_value,
    )

    return {
        'raw_video': raw_video_path,
        'visualization_video': vis_video_path,
        'temporal_graphs_h5': h5_path,
        'num_vehicles': len(vehicle_ids),
        'num_frames': num_frames,
    }


def save_temporal_graphs_h5(
    frame_records,
    vehicle_ids,
    vehicle_class_map,
    normalized_timestamps,
    video_path,
    h5_path,
    clip_name,
    fps,
    missing_value,
):
    """Save temporal graph data in HDF5 format for Dynamic Graph Networks."""

    num_frames = len(frame_records)
    num_vehicles = len(vehicle_ids)
    vehicle_id_to_idx = {vehicle_id: idx for idx, vehicle_id in enumerate(vehicle_ids)}

    with h5py.File(h5_path, 'w') as f:
        f.attrs['clip_name'] = clip_name
        f.attrs['video_path'] = video_path
        f.attrs['creation_time'] = datetime.now().isoformat()
        f.attrs['total_vehicles'] = num_vehicles
        f.attrs['total_frames'] = num_frames
        f.attrs['missing_value'] = missing_value

        graphs_group = f.create_group('graphs')

        for frame_idx, frame_data in enumerate(frame_records):
            time_group = graphs_group.create_group(f"t_{frame_idx:04d}")

            node_features = np.full((num_vehicles, 8), missing_value, dtype=np.float32)
            adjacency_matrix = np.full(
                (num_vehicles, num_vehicles), missing_value, dtype=np.float32
            )
            node_mask = np.zeros((num_vehicles,), dtype=np.uint8)

            for vehicle_id, data in frame_data.items():
                node_idx = vehicle_id_to_idx[vehicle_id]
                centroid = data['centroid']
                bbox = data['bbox']
                class_id = data['class_id']

                node_features[node_idx] = np.array(
                    [
                        float(centroid[0]),
                        float(centroid[1]),
                        float(bbox[0]),
                        float(bbox[1]),
                        float(bbox[2]),
                        float(bbox[3]),
                        float(class_id),
                        float(normalized_timestamps[frame_idx]),
                    ],
                    dtype=np.float32,
                )
                node_mask[node_idx] = 1

            for i, vehicle_id_i in enumerate(vehicle_ids):
                if node_mask[i] == 0:
                    continue

                adjacency_matrix[i, i] = 0.0
                centroid_i = np.array(frame_data[vehicle_id_i]['centroid'], dtype=np.float32)

                for j in range(i + 1, num_vehicles):
                    if node_mask[j] == 0:
                        continue

                    vehicle_id_j = vehicle_ids[j]
                    centroid_j = np.array(frame_data[vehicle_id_j]['centroid'], dtype=np.float32)
                    distance = float(np.linalg.norm(centroid_i - centroid_j))
                    adjacency_matrix[i, j] = distance
                    adjacency_matrix[j, i] = distance

            edge_mask = (adjacency_matrix != missing_value).astype(np.uint8)

            time_group.create_dataset('node_features', data=node_features)
            time_group.create_dataset('adjacency_matrix', data=adjacency_matrix)
            time_group.create_dataset('node_mask', data=node_mask, dtype=np.uint8)
            time_group.create_dataset('edge_mask', data=edge_mask, dtype=np.uint8)

            time_group.attrs['frame_number'] = frame_idx
            time_group.attrs['num_nodes_present'] = int(node_mask.sum())
            time_group.attrs['timestamp'] = frame_idx / fps if fps > 0 else 0.0
            time_group.attrs['normalized_timestamp'] = normalized_timestamps[frame_idx]

        summary_group = f.create_group('summary')
        summary_group.attrs['total_frames'] = num_frames
        summary_group.attrs['total_unique_vehicles'] = num_vehicles
        summary_group.attrs['frames_with_vehicles'] = sum(1 for frame in frame_records if frame)
        summary_group.attrs['max_vehicles_per_frame'] = max(
            (len(frame) for frame in frame_records), default=0
        )
        summary_group.attrs['vehicle_detection_rate'] = (
            summary_group.attrs['frames_with_vehicles'] / num_frames if num_frames else 0.0
        )
        summary_group.attrs['node_feature_columns'] = json.dumps([
            'centroid_x',
            'centroid_y',
            'bbox_x',
            'bbox_y',
            'bbox_w',
            'bbox_h',
            'vehicle_class',
            'normalized_timestamp',
        ])

        vehicle_class_vector = np.array(
            [vehicle_class_map.get(vehicle_id, -1) for vehicle_id in vehicle_ids],
            dtype=np.int32,
        )
        summary_group.create_dataset('vehicle_class_ids', data=vehicle_class_vector)

        vehicle_mapping = {str(vehicle_id): idx for idx, vehicle_id in enumerate(vehicle_ids)}
        summary_group.create_dataset(
            'vehicle_id_mapping',
            data=json.dumps(vehicle_mapping),
            dtype=h5py.special_dtype(vlen=str),
        )

