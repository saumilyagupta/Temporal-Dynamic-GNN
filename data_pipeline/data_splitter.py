#!/usr/bin/env python3
"""
Data organization utilities for NiAD-based Dynamic Graph Network pipeline.
"""

import os
import csv
import shutil


def organize_processed_clip(processed_result, clip_info, output_base, split_type):
    """
    Organize a processed clip into the proper folder structure
    
    Args:
        processed_result: Dict returned from vehicle_processor
        clip_info: Original clip info dict
        output_base: Base output directory
        split_type: 'train', 'val', or 'test'
    
    Returns:
        Dict with final file paths
    """
    video_name = clip_info['video_name']
    class_name = clip_info['class_name']

    clip_folder = os.path.join(output_base, split_type, class_name, video_name)
    os.makedirs(clip_folder, exist_ok=True)
    
    # Copy/move files to proper locations
    raw_dest = os.path.join(clip_folder, "raw_video.mp4")
    vis_dest = os.path.join(clip_folder, "visualization_video.mp4")
    h5_dest = os.path.join(clip_folder, "temporal_graphs.h5")
    
    # Copy files
    shutil.copy2(processed_result['raw_video'], raw_dest)
    shutil.move(processed_result['visualization_video'], vis_dest)
    shutil.move(processed_result['temporal_graphs_h5'], h5_dest)
    
    return {
        'clip_name': video_name,
        'class_name': class_name,
        'source_video': clip_info.get('video_name', video_name),
        'video_name': video_name,
        'original_class_name': clip_info.get('original_class_name', class_name),
        'original_split': clip_info.get('original_split', split_type),
        'split': split_type,
        'folder_path': clip_folder,
        'raw_video': raw_dest,
        'visualization_video': vis_dest,
        'temporal_graphs_h5': h5_dest,
        'num_vehicles': processed_result['num_vehicles'],
        'num_frames': processed_result['num_frames']
    }


def create_manifest_csv(organized_clips, output_path, split_name):
    """
    Create CSV manifest for a split
    
    Args:
        organized_clips: List of organized clip info dicts
        output_path: Path to save CSV file
        split_name: Name of the split (train/val/test)
    
    CSV columns: video_folder_path, label, source_video, num_vehicles, num_frames
    """
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['video_folder_path', 'label', 'source_video', 'num_vehicles', 'num_frames']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for clip in organized_clips:
            writer.writerow({
                'video_folder_path': clip['folder_path'],
                'label': clip['class_name'],
                'source_video': clip['source_video'],
                'num_vehicles': clip['num_vehicles'],
                'num_frames': clip['num_frames']
            })
    
    print(f"Created {split_name} manifest: {output_path}")


def create_all_manifests(organized_clips_by_split, output_base):
    """
    Create CSV manifests for all splits
    
    Args:
        organized_clips_by_split: Dict with 'train', 'val', 'test' keys
        output_base: Base output directory
    """
    for split_name in ['train', 'val', 'test']:
        manifest_path = os.path.join(output_base, f"{split_name}_manifest.csv")
        create_manifest_csv(organized_clips_by_split[split_name], manifest_path, split_name)

