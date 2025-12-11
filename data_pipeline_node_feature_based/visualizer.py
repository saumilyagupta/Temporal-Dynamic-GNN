#!/usr/bin/env python3
"""
Visualization utilities for the node-feature-based graph pipeline.

This module provides tools to visualize:
- Graph structure overlaid on video frames
- Node features and edge weights
- Trajectories across frames
"""

import os
import cv2
import json
import torch
import numpy as np
from typing import List, Optional, Tuple


def load_graphs_from_folder(folder_path: str, num_frames: int = 30) -> List:
    """Load all graph files from a video folder."""
    graphs = []
    for t in range(num_frames):
        graph_path = os.path.join(folder_path, f"graph_{t:03d}.pt")
        if os.path.exists(graph_path):
            graphs.append(torch.load(graph_path))
        else:
            graphs.append(None)
    return graphs


def draw_graph_on_frame(
    frame: np.ndarray,
    graph,
    show_node_ids: bool = True,
    show_edge_weights: bool = True,
    node_color: Tuple[int, int, int] = (0, 255, 0),
    edge_color: Tuple[int, int, int] = (0, 0, 255),
    trajectory_color: Tuple[int, int, int] = (255, 0, 0)
) -> np.ndarray:
    """
    Draw graph visualization on a frame.
    
    Args:
        frame: Input frame (BGR)
        graph: PyG Data object
        show_node_ids: Whether to show node IDs
        show_edge_weights: Whether to show edge weights
        node_color: BGR color for nodes
        edge_color: BGR color for edges
        trajectory_color: BGR color for trajectories
        
    Returns:
        Frame with visualization overlay
    """
    vis_frame = frame.copy()
    
    if graph is None or graph.x.shape[0] == 0:
        return vis_frame
    
    num_nodes = graph.x.shape[0]
    
    # Extract node positions (centroid_x, centroid_y from features)
    centroids = graph.x[:, :2].numpy().astype(int)
    bboxes = graph.x[:, 2:6].numpy().astype(int)
    node_ids = graph.node_ids.numpy()
    
    # Draw edges first (so they appear behind nodes)
    if graph.edge_index.shape[1] > 0 and show_edge_weights:
        edge_index = graph.edge_index.numpy()
        edge_weight = graph.edge_weight.numpy()
        
        # Only draw each edge once (avoid duplicates for undirected)
        drawn_edges = set()
        for idx in range(edge_index.shape[1]):
            i, j = edge_index[0, idx], edge_index[1, idx]
            edge_key = tuple(sorted([i, j]))
            
            if edge_key in drawn_edges:
                continue
            drawn_edges.add(edge_key)
            
            pt1 = tuple(centroids[i])
            pt2 = tuple(centroids[j])
            cv2.line(vis_frame, pt1, pt2, edge_color, 1)
            
            if show_edge_weights:
                mid = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
                cv2.putText(vis_frame, f"{edge_weight[idx]:.0f}", mid,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, edge_color, 1)
    
    # Draw nodes (bounding boxes and centroids)
    for i in range(num_nodes):
        x, y, w, h = bboxes[i]
        cx, cy = centroids[i]
        nid = node_ids[i]
        
        # Draw bounding box
        cv2.rectangle(vis_frame, (x, y), (x + w, y + h), node_color, 2)
        
        # Draw centroid
        cv2.circle(vis_frame, (cx, cy), 4, node_color, -1)
        
        # Draw node ID
        if show_node_ids:
            cv2.putText(vis_frame, f"ID:{nid}", (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, node_color, 2)
    
    return vis_frame


def create_visualization_video(
    video_path: str,
    graphs_folder: str,
    output_path: str,
    num_frames: int = 30,
    show_node_ids: bool = True,
    show_edge_weights: bool = True
) -> None:
    """
    Create a visualization video from original video and generated graphs.
    
    Args:
        video_path: Path to original video
        graphs_folder: Path to folder containing graph_XXX.pt files
        output_path: Path for output visualization video
        num_frames: Number of frames to process
        show_node_ids: Whether to show node IDs
        show_edge_weights: Whether to show edge weights
    """
    # Load graphs
    graphs = load_graphs_from_folder(graphs_folder, num_frames)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    frame_idx = 0
    while frame_idx < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        graph = graphs[frame_idx] if frame_idx < len(graphs) else None
        vis_frame = draw_graph_on_frame(
            frame, graph,
            show_node_ids=show_node_ids,
            show_edge_weights=show_edge_weights
        )
        
        # Add frame number
        cv2.putText(vis_frame, f"Frame: {frame_idx}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        if graph is not None:
            cv2.putText(vis_frame, f"Nodes: {graph.x.shape[0]}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(vis_frame)
        frame_idx += 1
    
    cap.release()
    out.release()
    print(f"Created visualization video: {output_path}")


def visualize_graph_stats(graphs_folder: str, num_frames: int = 30) -> dict:
    """
    Print statistics about a graph sequence.
    
    Args:
        graphs_folder: Path to folder containing graph files
        num_frames: Number of frames
        
    Returns:
        Dict with statistics
    """
    graphs = load_graphs_from_folder(graphs_folder, num_frames)
    
    stats = {
        'num_frames': num_frames,
        'nodes_per_frame': [],
        'edges_per_frame': [],
        'unique_node_ids': set(),
    }
    
    for t, graph in enumerate(graphs):
        if graph is None:
            stats['nodes_per_frame'].append(0)
            stats['edges_per_frame'].append(0)
        else:
            num_nodes = graph.x.shape[0]
            num_edges = graph.edge_index.shape[1] // 2  # Divide by 2 for undirected
            stats['nodes_per_frame'].append(num_nodes)
            stats['edges_per_frame'].append(num_edges)
            
            if num_nodes > 0:
                for nid in graph.node_ids.numpy():
                    stats['unique_node_ids'].add(int(nid))
    
    # Compute summary stats
    stats['total_unique_vehicles'] = len(stats['unique_node_ids'])
    stats['avg_nodes_per_frame'] = np.mean(stats['nodes_per_frame'])
    stats['max_nodes_per_frame'] = max(stats['nodes_per_frame'])
    stats['min_nodes_per_frame'] = min(stats['nodes_per_frame'])
    stats['frames_with_vehicles'] = sum(1 for n in stats['nodes_per_frame'] if n > 0)
    
    # Print stats
    print(f"\nGraph Sequence Statistics for: {graphs_folder}")
    print(f"  Total frames: {stats['num_frames']}")
    print(f"  Frames with vehicles: {stats['frames_with_vehicles']}")
    print(f"  Total unique vehicles: {stats['total_unique_vehicles']}")
    print(f"  Avg nodes per frame: {stats['avg_nodes_per_frame']:.2f}")
    print(f"  Max nodes per frame: {stats['max_nodes_per_frame']}")
    print(f"  Min nodes per frame: {stats['min_nodes_per_frame']}")
    print(f"  Nodes per frame: {stats['nodes_per_frame']}")
    
    return stats


def inspect_single_graph(graph_path: str) -> None:
    """
    Print detailed information about a single graph file.
    
    Args:
        graph_path: Path to a .pt graph file
    """
    graph = torch.load(graph_path)
    
    print(f"\nGraph: {graph_path}")
    print(f"  x shape: {graph.x.shape}")
    print(f"  edge_index shape: {graph.edge_index.shape}")
    print(f"  edge_weight shape: {graph.edge_weight.shape}")
    print(f"  node_ids: {graph.node_ids.tolist()}")
    print(f"  t (frame index): {graph.t.item()}")
    
    if graph.x.shape[0] > 0:
        print(f"\n  Node features (first 3 nodes):")
        feature_names = ['cx', 'cy', 'bx', 'by', 'bw', 'bh', 'class', 'norm_t']
        for i in range(min(3, graph.x.shape[0])):
            features = graph.x[i].tolist()
            print(f"    Node {i} (ID={graph.node_ids[i].item()}):")
            for name, val in zip(feature_names, features):
                print(f"      {name}: {val:.2f}")
    
    if graph.edge_index.shape[1] > 0:
        print(f"\n  Sample edges (first 5):")
        for i in range(min(5, graph.edge_index.shape[1])):
            src, dst = graph.edge_index[:, i].tolist()
            weight = graph.edge_weight[i].item()
            print(f"    {src} -> {dst}: {weight:.2f}")


# ============================================================================
# CLI for standalone visualization
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize graph data")
    parser.add_argument("--folder", type=str, help="Path to graphs folder")
    parser.add_argument("--video", type=str, help="Path to original video")
    parser.add_argument("--output", type=str, help="Output video path")
    parser.add_argument("--graph", type=str, help="Path to single graph file to inspect")
    parser.add_argument("--stats", action="store_true", help="Print statistics only")
    
    args = parser.parse_args()
    
    if args.graph:
        inspect_single_graph(args.graph)
    elif args.folder and args.stats:
        visualize_graph_stats(args.folder)
    elif args.folder and args.video and args.output:
        create_visualization_video(args.video, args.folder, args.output)
    else:
        print("Usage examples:")
        print("  python visualizer.py --graph path/to/graph_000.pt")
        print("  python visualizer.py --folder path/to/graphs --stats")
        print("  python visualizer.py --folder path/to/graphs --video path/to/video.mp4 --output vis.mp4")

