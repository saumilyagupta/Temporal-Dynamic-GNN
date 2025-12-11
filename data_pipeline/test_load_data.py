#!/usr/bin/env python3
"""
Test script to demonstrate loading data from the pipeline output
"""

import h5py
import torch
import pandas as pd
import os


def load_video_data(video_folder):
    """Load all data from a video folder"""
    h5_path = os.path.join(video_folder, 'temporal_graphs.h5')
    
    with h5py.File(h5_path, 'r') as f:
        # Load metadata
        metadata = {
            'clip_name': f.attrs['clip_name'],
            'total_frames': f.attrs['total_frames'],
            'total_vehicles': f.attrs['total_vehicles'],
        }
        
        # Load all temporal graphs
        temporal_graphs = []
        for t in range(metadata['total_frames']):
            time_key = f"t_{t:04d}"
            if time_key in f['graphs']:
                time_group = f[f'graphs/{time_key}']
                graph_data = {
                    'node_features': torch.tensor(time_group['node_features'][:]),
                    'adjacency_matrix': torch.tensor(time_group['adjacency_matrix'][:]),
                    'node_mask': torch.tensor(time_group['node_mask'][:]),
                    'edge_mask': torch.tensor(time_group['edge_mask'][:]),
                    'frame_number': time_group.attrs['frame_number'],
                    'num_nodes_present': time_group.attrs['num_nodes_present'],
                    'timestamp': time_group.attrs['timestamp'],
                    'normalized_timestamp': time_group.attrs['normalized_timestamp'],
                }
                temporal_graphs.append(graph_data)
        
        return metadata, temporal_graphs


def main():
    print("=" * 80)
    print("TESTING DATA LOADING FROM PIPELINE OUTPUT")
    print("=" * 80)
    
    # Load manifests
    output_base = '/workspace/saumilya/GNN-Research/output_pipeline'
    
    print("\n=== Loading Train Manifest ===")
    train_df = pd.read_csv(f'{output_base}/train_manifest.csv')
    print(f"Train videos: {len(train_df)}")
    print(train_df.head())
    
    print("\n=== Loading Test Manifest ===")
    test_df = pd.read_csv(f'{output_base}/test_manifest.csv')
    print(f"Test videos: {len(test_df)}")
    print(test_df.head())
    
    # Load a sample video
    print("\n=== Loading Sample Video ===")
    sample_video_folder = train_df.iloc[0]['video_folder_path']
    sample_label = train_df.iloc[0]['label']
    
    print(f"Loading: {sample_video_folder}")
    print(f"Label: {sample_label}")
    
    metadata, temporal_graphs = load_video_data(sample_video_folder)
    
    print(f"\nMetadata:")
    print(f"  Clip name: {metadata['clip_name']}")
    print(f"  Total frames: {metadata['total_frames']}")
    print(f"  Total vehicles: {metadata['total_vehicles']}")
    
    print(f"\nLoaded {len(temporal_graphs)} temporal graphs")
    
    # Show a sample frame with vehicles
    for i, graph in enumerate(temporal_graphs):
        if graph['num_nodes_present'] > 1:  # Frame with multiple vehicles
            print(f"\n=== Sample Frame {i} ===")
            print(f"Number of vehicles: {graph['num_nodes_present']}")
            print(f"Node features shape: {graph['node_features'].shape}")
            print(f"Adjacency matrix shape: {graph['adjacency_matrix'].shape}")
            print(f"Node mask shape: {graph['node_mask'].shape}")
            
            print(f"\nFirst vehicle features:")
            print(f"  Centroid: ({graph['node_features'][0, 0]:.1f}, {graph['node_features'][0, 1]:.1f})")
            print(f"  BBox (x,y,w,h): {graph['node_features'][0, 2:].tolist()}")
            print(f"  Vehicle class: {int(graph['node_features'][0, 6])}")
            print(f"  Normalized timestamp: {graph['node_features'][0, 7]:.3f}")
            
            if graph['num_nodes_present'] > 1:
                print(f"\nFirst edge:")
                print(f"  Distance (0->1): {graph['adjacency_matrix'][0, 1]:.1f} pixels")
            
            break
    
    print("\n" + "=" * 80)
    print("DATA LOADING TEST SUCCESSFUL!")
    print("=" * 80)
    
    print("\nNext steps:")
    print("1. Use pandas to load manifests and iterate through videos")
    print("2. Load temporal graphs from HDF5 files")
    print("3. Convert to PyTorch tensors for GNN training")
    print("4. Process complete graphs with all vehicle interactions")


if __name__ == "__main__":
    main()

