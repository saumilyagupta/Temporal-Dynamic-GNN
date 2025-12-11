#!/usr/bin/env python3
"""
Dynamic Graph Network data pipeline.
Discovers pre-split NiAD videos, detects vehicles, and generates temporal graphs.
"""

import os
import sys
import time
import shutil
from collections import defaultdict

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

from utils import discover_niaad_videos, ensure_dir, ProgressLogger, format_time
from vehicle_processor import process_video_for_vehicles
from data_splitter import (
    organize_processed_clip,
    create_all_manifests
)


# ============================================================================
# CONFIGURATION - Edit these paths and parameters
# ============================================================================

CONFIG = {
    # Input/Output paths
    'input_folder': '/workspace/saumilya/GNN-Research/accident_prediction/NiAD_Large_Videos',
    'output_folder': '/workspace/saumilya/GNN-Research/NiAD_Large_Videos_processed_graphs',

    # Directory mappings
    'split_mapping': {'Training': 'train', 'Validation': 'val', 'Testing': 'test'},
    'class_mapping': {'Normal': 'Normal', 'Accident': 'Anomalous'},

    # Vehicle detection parameters
    'yolo_model': 'yolov8n.pt',
    'detection_conf': 0.3,
    'vehicle_classes': [2, 3, 5, 7],  # car, motorcycle, bus, truck
    'device': 'cuda:5',  # GPU device to use

    # Dynamic graph parameters
    'frames_per_video': 30,
    'missing_value': float('-inf'),
}


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    """Main pipeline execution for Dynamic Graph Networks"""

    print("=" * 80)
    print("DYNAMIC GRAPH NETWORK DATA PIPELINE")
    print("=" * 80)

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

    videos_by_split = {'train': [], 'val': [], 'test': []}
    for info in video_list:
        videos_by_split.setdefault(info['split'], []).append(info)

    for split in ['train', 'val', 'test']:
        for class_name in CONFIG['class_mapping'].values():
            ensure_dir(os.path.join(CONFIG['output_folder'], split, class_name))

    temp_processing_dir = os.path.join(CONFIG['output_folder'], '_temp_processing')
    ensure_dir(temp_processing_dir)

    organized_clips_by_split = {'train': [], 'val': [], 'test': []}

    # ------------------------------------------------------------------------
    # Step 2: Process videos for vehicle detection and graph construction
    # ------------------------------------------------------------------------
    print("\n[Step 2/4] Processing videos (vehicle detection & graph generation)...")

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

            clip_temp_folder = os.path.join(temp_processing_dir, video_name)
            ensure_dir(clip_temp_folder)

            try:
                processed_result = process_video_for_vehicles(
                    video_path=video_path,
                    output_folder=clip_temp_folder,
                    clip_name=video_name,
                    yolo_model=CONFIG['yolo_model'],
                    detection_conf=CONFIG['detection_conf'],
                    vehicle_classes=CONFIG['vehicle_classes'],
                    frames_per_video=CONFIG['frames_per_video'],
                    missing_value=CONFIG['missing_value'],
                    device=CONFIG['device']
                )

                organized_clip = organize_processed_clip(
                    processed_result=processed_result,
                    clip_info=video_info,
                    output_base=CONFIG['output_folder'],
                    split_type=split_name
                )

                organized_clips_by_split[split_name].append(organized_clip)

            except Exception as exc:
                print(f"\n    Error processing {video_name}: {exc}")
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
    print(f"\nOutput directory: {CONFIG['output_folder']}")

    print("\nDataset statistics:")
    for split_name in ['train', 'val', 'test']:
        clips = organized_clips_by_split[split_name]
        if not clips:
            print(f"\n  {split_name.upper()}: No processed videos")
            continue

        class_counts = defaultdict(int)
        total_vehicles = 0
        total_frames = 0

        for clip in clips:
            class_counts[clip['class_name']] += 1
            total_vehicles += clip['num_vehicles']
            total_frames += clip['num_frames']

        print(f"\n  {split_name.upper()}:")
        print(f"    Total videos: {len(clips)}")
        for class_name, count in class_counts.items():
            print(f"    {class_name}: {count}")
        print(f"    Total unique vehicles (summed across videos): {total_vehicles}")
        print(f"    Total frames processed: {total_frames}")

    print("\nGenerated files:")
    print(f"  - {CONFIG['output_folder']}/train_manifest.csv")
    print(f"  - {CONFIG['output_folder']}/val_manifest.csv")
    print(f"  - {CONFIG['output_folder']}/test_manifest.csv")

    print("\nFolder structure:")
    print(f"  {CONFIG['output_folder']}/")
    print(f"    ├── train/")
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

