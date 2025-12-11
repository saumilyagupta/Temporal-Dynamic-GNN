#!/usr/bin/env python3
"""
Utility functions for the video processing pipeline
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional


def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    os.makedirs(directory, exist_ok=True)
    return directory


def load_json(json_path):
    """Load JSON file"""
    with open(json_path, 'r') as f:
        return json.load(f)


def save_json(data, output_path):
    """Save data to JSON file"""
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def discover_niaad_videos(
    base_dir: str,
    split_mapping: Optional[Dict[str, str]] = None,
    class_mapping: Optional[Dict[str, str]] = None,
) -> List[Dict[str, str]]:
    """
    Discover videos within the NiAD_Large_Videos directory structure.

    Args:
        base_dir: Root directory containing Training/Validation/Testing folders.
        split_mapping: Optional mapping from directory names to split names.
        class_mapping: Optional mapping from class directory names to label names.

    Returns:
        List of dictionaries with keys:
            - split: Mapped split name (train/val/test)
            - original_split: Original split directory name
            - class_name: Mapped class label
            - original_class_name: Original class directory name
            - video_name: Stem of the video file
            - video_path: Absolute path to the video file
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Input folder does not exist: {base_dir}")

    if split_mapping is None:
        split_mapping = {'Training': 'train', 'Validation': 'val', 'Testing': 'test'}

    if class_mapping is None:
        class_mapping = {'Normal': 'Normal', 'Accident': 'Anomalous'}

    discovered = []

    for split_dir in base_path.iterdir():
        if not split_dir.is_dir():
            continue

        mapped_split = split_mapping.get(split_dir.name)
        if not mapped_split:
            continue

        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue

            mapped_class = class_mapping.get(class_dir.name, class_dir.name)

            for video_file in class_dir.glob("*.mp4"):
                discovered.append({
                    'split': mapped_split,
                    'original_split': split_dir.name,
                    'class_name': mapped_class,
                    'original_class_name': class_dir.name,
                    'video_name': video_file.stem,
                    'video_path': str(video_file.resolve()),
                })

    discovered.sort(key=lambda item: (item['split'], item['class_name'], item['video_name']))
    return discovered


def get_video_info(video_path):
    """Get basic video information"""
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()

    return {
        'fps': fps,
        'frame_count': frame_count,
        'width': width,
        'height': height,
        'duration': duration
    }


class ProgressLogger:
    """Simple progress logger"""
    def __init__(self, total, desc="Processing"):
        self.total = total
        self.current = 0
        self.desc = desc
    
    def update(self, n=1):
        self.current += n
        percent = (self.current / self.total) * 100 if self.total > 0 else 0
        print(f"{self.desc}: {self.current}/{self.total} ({percent:.1f}%)")
    
    def set_description(self, desc):
        self.desc = desc
    
    def finish(self):
        print(f"{self.desc}: Complete ({self.total}/{self.total})")


def format_time(seconds):
    """Format seconds to readable time string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

