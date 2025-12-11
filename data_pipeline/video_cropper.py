#!/usr/bin/env python3
"""
Video cropping module - extracts clips from tagged videos
"""

import os
import cv2
import json
from pathlib import Path


def crop_video_by_timestamps(video_path, json_path, output_dir, source_video_name):
    """
    Crop video based on timestamps from JSON file
    
    Args:
        video_path: Path to source video file
        json_path: Path to JSON tags file
        output_dir: Temporary directory to save cropped clips
        source_video_name: Name of source video (without extension)
    
    Returns:
        List of dicts containing clip information:
        [{
            'clip_name': 'A22_Normal_1',
            'class_name': 'Normal',
            'source_video': 'A22',
            'output_path': '/path/to/clip.mp4',
            'duration': 3.18,
            'start_time': 0.0,
            'end_time': 3.18
        }, ...]
    """
    # Read JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Process each tag
    clip_counters = {'Normal': 1, 'Anomalous': 1}
    clips_info = []
    
    for tag in data['tags']:
        start_time = tag['startTime']
        end_time = tag['endTime']
        class_name = tag['className']
        
        # Map class names to standard names
        if class_name in ['Class A', 'Normal']:
            class_name = 'Normal'
        elif class_name == 'Anomalous':
            class_name = 'Anomalous'
        else:
            print(f"Unknown class: {class_name}, skipping...")
            continue
        
        # Convert time to frame numbers
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        # Set video position to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Create output filename
        clip_number = clip_counters[class_name]
        clip_name = f"{source_video_name}_{class_name}_{clip_number}"
        output_filename = f"{clip_name}.mp4"
        output_path = os.path.join(output_dir, output_filename)
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        # Write frames
        for frame_num in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        
        out.release()
        
        # Store clip information
        clip_info = {
            'clip_name': clip_name,
            'class_name': class_name,
            'source_video': source_video_name,
            'output_path': output_path,
            'duration': end_time - start_time,
            'start_time': start_time,
            'end_time': end_time
        }
        clips_info.append(clip_info)
        
        print(f"  Cropped: {clip_name} ({class_name}) - Duration: {end_time - start_time:.2f}s")
        
        # Increment counter for this class
        clip_counters[class_name] += 1
    
    cap.release()
    return clips_info


def crop_videos_from_list(video_list, temp_dir):
    """
    Crop multiple videos based on list of video-json pairs
    
    Args:
        video_list: List of tuples (source_video_name, video_path, json_path)
        temp_dir: Temporary directory for cropped clips
    
    Returns:
        List of all clip information dicts
    """
    os.makedirs(temp_dir, exist_ok=True)
    all_clips = []
    
    for source_video_name, video_path, json_path in video_list:
        print(f"\nCropping video: {source_video_name}")
        clips = crop_video_by_timestamps(video_path, json_path, temp_dir, source_video_name)
        all_clips.extend(clips)
    
    return all_clips

