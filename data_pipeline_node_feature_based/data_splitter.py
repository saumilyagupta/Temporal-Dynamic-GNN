#!/usr/bin/env python3
"""
Data organization utilities for the node-feature-based graph pipeline.

This module handles organizing processed clips into the proper folder structure
and creating CSV manifests for training.
"""

import os
import csv
import shutil
import json
from typing import Dict, List


def organize_processed_clip(
    processed_result: dict,
    clip_info: dict,
    output_base: str,
    split_type: str,
    copy_raw_video: bool = False
) -> dict:
    """
    Organize a processed clip into the proper folder structure.
    
    Args:
        processed_result: Dict returned from vehicle_processor.process_video_to_graphs
        clip_info: Original clip info dict from discover_niaad_videos
        output_base: Base output directory
        split_type: 'train', 'val', or 'test'
        copy_raw_video: Whether to copy the raw video to the output folder
    
    Returns:
        Dict with final file paths and metadata
    """
    video_name = clip_info['video_name']
    class_name = clip_info['class_name']

    # Create final destination folder
    clip_folder = os.path.join(output_base, split_type, class_name, video_name)
    os.makedirs(clip_folder, exist_ok=True)
    
    # Move graph files and metadata from temp folder to final location
    temp_folder = processed_result['output_folder']
    
    # Move all graph_XXX.pt files
    graph_files = []
    for filename in sorted(os.listdir(temp_folder)):
        if filename.startswith('graph_') and filename.endswith('.pt'):
            src = os.path.join(temp_folder, filename)
            dst = os.path.join(clip_folder, filename)
            shutil.move(src, dst)
            graph_files.append(dst)
    
    # Move metadata.json
    metadata_src = os.path.join(temp_folder, 'metadata.json')
    metadata_dst = os.path.join(clip_folder, 'metadata.json')
    if os.path.exists(metadata_src):
        shutil.move(metadata_src, metadata_dst)
    
    # Move visualization video if it exists
    vis_video_dst = None
    if processed_result.get('visualization_video') and os.path.exists(processed_result['visualization_video']):
        vis_video_dst = os.path.join(clip_folder, 'visualization_video.mp4')
        shutil.move(processed_result['visualization_video'], vis_video_dst)
    
    # Optionally copy raw video
    raw_video_dst = None
    if copy_raw_video:
        raw_video_dst = os.path.join(clip_folder, 'raw_video.mp4')
        shutil.copy2(clip_info['video_path'], raw_video_dst)
    
    return {
        'clip_name': video_name,
        'class_name': class_name,
        'source_video': clip_info.get('video_name', video_name),
        'video_name': video_name,
        'original_class_name': clip_info.get('original_class_name', class_name),
        'original_split': clip_info.get('original_split', split_type),
        'split': split_type,
        'folder_path': clip_folder,
        'graph_files': graph_files,
        'metadata_path': metadata_dst,
        'visualization_video': vis_video_dst,
        'raw_video': raw_video_dst,
        'num_vehicles': processed_result['num_vehicles'],
        'num_frames': processed_result['num_frames'],
        'nodes_per_frame': processed_result.get('nodes_per_frame', [])
    }


def create_manifest_csv(
    organized_clips: List[dict],
    output_path: str,
    split_name: str
) -> None:
    """
    Create CSV manifest for a split.
    
    Args:
        organized_clips: List of organized clip info dicts
        output_path: Path to save CSV file
        split_name: Name of the split (train/val/test)
    
    CSV columns: video_folder_path, label, source_video, num_vehicles, num_frames, max_nodes_per_frame
    """
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = [
            'video_folder_path',
            'label',
            'source_video',
            'num_vehicles',
            'num_frames',
            'max_nodes_per_frame'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for clip in organized_clips:
            nodes_per_frame = clip.get('nodes_per_frame', [])
            max_nodes = max(nodes_per_frame) if nodes_per_frame else 0
            
            writer.writerow({
                'video_folder_path': clip['folder_path'],
                'label': clip['class_name'],
                'source_video': clip['source_video'],
                'num_vehicles': clip['num_vehicles'],
                'num_frames': clip['num_frames'],
                'max_nodes_per_frame': max_nodes
            })
    
    print(f"  Created {split_name} manifest: {output_path}")


def create_all_manifests(
    organized_clips_by_split: Dict[str, List[dict]],
    output_base: str
) -> None:
    """
    Create CSV manifests for all splits.
    
    Args:
        organized_clips_by_split: Dict with 'train', 'val', 'test' keys
        output_base: Base output directory
    """
    for split_name in ['train', 'val', 'test']:
        clips = organized_clips_by_split.get(split_name, [])
        manifest_path = os.path.join(output_base, f"{split_name}_manifest.csv")
        create_manifest_csv(clips, manifest_path, split_name)


def load_graph_sequence(folder_path: str, num_frames: int = 30):
    """
    Load a sequence of graphs from a video folder.
    
    Args:
        folder_path: Path to the folder containing graph_XXX.pt files
        num_frames: Number of frames to load
        
    Returns:
        List of PyG Data objects
    """
    import torch
    
    graphs = []
    for t in range(num_frames):
        graph_path = os.path.join(folder_path, f"graph_{t:03d}.pt")
        if os.path.exists(graph_path):
            data = torch.load(graph_path)
            graphs.append(data)
        else:
            print(f"Warning: Missing graph file {graph_path}")
            graphs.append(None)
    
    return graphs


def load_manifest(manifest_path: str) -> List[dict]:
    """
    Load a manifest CSV file.
    
    Args:
        manifest_path: Path to manifest CSV file
        
    Returns:
        List of dicts with video info
    """
    videos = []
    with open(manifest_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            videos.append({
                'folder_path': row['video_folder_path'],
                'label': row['label'],
                'source_video': row['source_video'],
                'num_vehicles': int(row['num_vehicles']),
                'num_frames': int(row['num_frames']),
                'max_nodes_per_frame': int(row.get('max_nodes_per_frame', 0))
            })
    return videos

