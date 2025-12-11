#!/usr/bin/env python3
"""
Node-Feature-Based Graph Data Pipeline.

This pipeline converts videos from the NiAD dataset into sequences of
PyTorch Geometric graphs, where each frame becomes an independent graph
with variable-size node sets.

Usage:
    python main.py
    
Output Structure:
    output_folder/
      train/Normal/video_name/graph_000.pt...graph_029.pt, metadata.json
      train/Anomalous/video_name/...
      val/...
      test/...
      train_manifest.csv, val_manifest.csv, test_manifest.csv
"""

import os
import sys
import time
import shutil
from collections import defaultdict

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

from utils import discover_niaad_videos, ensure_dir, ProgressLogger, format_time
from vehicle_processor import process_video_to_graphs
from data_splitter import organize_processed_clip, create_all_manifests


# ============================================================================
# CONFIGURATION - Edit these paths and parameters
# ============================================================================

CONFIG = {
    # Input/Output paths
    'input_folder': '/workspace/saumilya/GNN-Research/accident_prediction/NiAD_Large_Videos',
    'output_folder': '/workspace/saumilya/GNN-Research/NiAD_graphs_node_feature_based',

    # Directory mappings
    'split_mapping': {'Training': 'train', 'Validation': 'val', 'Testing': 'test'},
    'class_mapping': {'Normal': 'Normal', 'Accident': 'Anomalous'},

    # Vehicle detection parameters
    'yolo_model': '/workspace/saumilya/GNN-Research/data_pipeline/yolov8n.pt',
    'detection_conf': 0.3,
    'vehicle_classes': [2, 3, 5, 7],  # car, motorcycle, bus, truck
    'device': 'cuda:0',  # GPU device to use (change as needed)

    # Graph generation parameters
    'frames_per_video': 30,
    
    # Options
    'generate_visualization': True,  # Generate visualization videos for each sample
    'copy_raw_video': False,  # Copy raw videos to output (uses more disk space)
}


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    """Main pipeline execution for node-feature-based graph generation."""

    print("=" * 80)
    print("NODE-FEATURE-BASED GRAPH DATA PIPELINE")
    print("=" * 80)
    print("\nOutput format: PyTorch Geometric Data objects (.pt files)")
    print("Each video → 30 graphs (one per frame)")
    print("Each graph → variable number of nodes (only present vehicles)")

    start_time = time.time()

    # ------------------------------------------------------------------------
    # Step 1: Discover videos from NiAD directory structure
    # ------------------------------------------------------------------------
    print("\n[Step 1/4] Discovering video files...")

    video_list = discover_niaad_videos(
        CONFIG['input_folder'],
        CONFIG['split_mapping'],
        CONFIG['class_mapping']
    )

    if not video_list:
        print("Error: No videos found in input folder!")
        return

    video_counts = defaultdict(lambda: defaultdict(int))
    for info in video_list:
        video_counts[info['split']][info['class_name']] += 1

    total_videos = sum(sum(class_counts.values()) for class_counts in video_counts.values())
    print(f"Discovered {total_videos} videos.")
    for split_name in ['train', 'val', 'test']:
        split_total = sum(video_counts[split_name].values())
        print(f"  {split_name.upper()}: {split_total} videos")
        for class_name, count in video_counts[split_name].items():
            print(f"    - {class_name}: {count}")

    # Organize videos by split
    videos_by_split = {'train': [], 'val': [], 'test': []}
    for info in video_list:
        videos_by_split.setdefault(info['split'], []).append(info)

    # Create output directories
    for split in ['train', 'val', 'test']:
        for class_name in CONFIG['class_mapping'].values():
            ensure_dir(os.path.join(CONFIG['output_folder'], split, class_name))

    # Create temporary processing directory
    temp_processing_dir = os.path.join(CONFIG['output_folder'], '_temp_processing')
    ensure_dir(temp_processing_dir)

    organized_clips_by_split = {'train': [], 'val': [], 'test': []}

    # ------------------------------------------------------------------------
    # Step 2: Process videos for vehicle detection and graph construction
    # ------------------------------------------------------------------------
    print("\n[Step 2/4] Processing videos (detection, tracking, graph generation)...")

    for split_name in ['train', 'val', 'test']:
        videos = videos_by_split.get(split_name, [])

        if not videos:
            print(f"\n  No videos in {split_name} split, skipping...")
            continue

        print(f"\n  Processing {split_name} split ({len(videos)} videos)...")
        progress = ProgressLogger(len(videos), f"  {split_name.capitalize()}")

        for video_info in videos:
            video_name = video_info['video_name']
            video_path = video_info['video_path']

            # Create temp folder for this video
            clip_temp_folder = os.path.join(temp_processing_dir, video_name)
            ensure_dir(clip_temp_folder)

            try:
                # Process video to generate graphs
                processed_result = process_video_to_graphs(
                    video_path=video_path,
                    output_folder=clip_temp_folder,
                    clip_name=video_name,
                    yolo_model=CONFIG['yolo_model'],
                    detection_conf=CONFIG['detection_conf'],
                    vehicle_classes=CONFIG['vehicle_classes'],
                    frames_per_video=CONFIG['frames_per_video'],
                    device=CONFIG['device'],
                    generate_visualization=CONFIG['generate_visualization']
                )

                # Organize into final folder structure
                organized_clip = organize_processed_clip(
                    processed_result=processed_result,
                    clip_info=video_info,
                    output_base=CONFIG['output_folder'],
                    split_type=split_name,
                    copy_raw_video=CONFIG['copy_raw_video']
                )

                organized_clips_by_split[split_name].append(organized_clip)

            except Exception as exc:
                print(f"\n    Error processing {video_name}: {exc}")
                import traceback
                traceback.print_exc()
                continue

            progress.update()

        progress.finish()

    # ------------------------------------------------------------------------
    # Step 3: Create CSV manifests for each split
    # ------------------------------------------------------------------------
    print("\n[Step 3/4] Creating CSV manifests...")

    create_all_manifests(organized_clips_by_split, CONFIG['output_folder'])

    # ------------------------------------------------------------------------
    # Step 4: Cleanup temporary processing directory
    # ------------------------------------------------------------------------
    print("\n[Step 4/4] Cleaning up temporary files...")

    if os.path.exists(temp_processing_dir):
        shutil.rmtree(temp_processing_dir)

    # ------------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------------
    elapsed_time = time.time() - start_time

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)

    print(f"\nProcessing time: {format_time(elapsed_time)}")
    print(f"Output directory: {CONFIG['output_folder']}")

    print("\nDataset statistics:")
    for split_name in ['train', 'val', 'test']:
        clips = organized_clips_by_split[split_name]
        if not clips:
            print(f"\n  {split_name.upper()}: No processed videos")
            continue

        class_counts = defaultdict(int)
        total_vehicles = 0
        total_frames = 0
        all_nodes_per_frame = []

        for clip in clips:
            class_counts[clip['class_name']] += 1
            total_vehicles += clip['num_vehicles']
            total_frames += clip['num_frames']
            all_nodes_per_frame.extend(clip.get('nodes_per_frame', []))

        avg_nodes = sum(all_nodes_per_frame) / len(all_nodes_per_frame) if all_nodes_per_frame else 0
        max_nodes = max(all_nodes_per_frame) if all_nodes_per_frame else 0

        print(f"\n  {split_name.upper()}:")
        print(f"    Total videos: {len(clips)}")
        for class_name, count in class_counts.items():
            print(f"    {class_name}: {count}")
        print(f"    Total unique vehicles (summed): {total_vehicles}")
        print(f"    Average nodes per frame: {avg_nodes:.2f}")
        print(f"    Max nodes per frame: {max_nodes}")

    print("\nGenerated files:")
    print(f"  - {CONFIG['output_folder']}/train_manifest.csv")
    print(f"  - {CONFIG['output_folder']}/val_manifest.csv")
    print(f"  - {CONFIG['output_folder']}/test_manifest.csv")

    print("\nFolder structure:")
    print(f"  {CONFIG['output_folder']}/")
    print(f"    ├── train/")
    print(f"    │   ├── Normal/")
    print(f"    │   │   └── video_name/")
    print(f"    │   │       ├── graph_000.pt ... graph_029.pt")
    print(f"    │   │       └── metadata.json")
    print(f"    │   └── Anomalous/")
    print(f"    ├── val/")
    print(f"    ├── test/")
    print(f"    ├── train_manifest.csv")
    print(f"    ├── val_manifest.csv")
    print(f"    └── test_manifest.csv")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

