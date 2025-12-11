"""
Analyze timestamp ranges and frame counts across all samples in the dataset.
"""

import os
import h5py
import pandas as pd
import numpy as np
from pathlib import Path

def analyze_dataset_timestamps(manifest_path):
    """Analyze timestamps and frames across all samples."""
    
    df = pd.read_csv(manifest_path)
    
    all_timestamps = []
    all_frame_counts = []
    sample_info = []
    
    print(f"Analyzing {len(df)} samples from {manifest_path}\n")
    
    for idx, row in df.iterrows():
        video_folder = row['video_folder_path']
        h5_path = os.path.join(video_folder, 'temporal_graphs.h5')
        
        if not os.path.exists(h5_path):
            continue
        
        with h5py.File(h5_path, 'r') as f:
            total_frames = f.attrs.get('total_frames', 0)
            
            # Get all timestamps from all frames
            timestamps_in_sample = []
            frames_with_data = 0
            
            for t in range(total_frames):
                time_key = f"t_{t:04d}"
                if time_key in f['graphs']:
                    frame_group = f[f'graphs/{time_key}']
                    timestamp = frame_group.attrs.get('timestamp', float(t))
                    timestamps_in_sample.append(timestamp)
                    
                    # Also check node features for timestamp column
                    node_features = f[f'graphs/{time_key}/node_features'][:]
                    if len(node_features) > 0:
                        frames_with_data += 1
                        # Column 4 is timestamp in node_features [x, y, w, h, timestamp, vehicle_id]
                        node_timestamps = node_features[:, 4] if node_features.shape[1] >= 5 else []
                        timestamps_in_sample.extend(node_timestamps.tolist())
                    
                    # Edge features also have timestamp in column 2 [distance, frame, timestamp]
                    edge_features = f[f'graphs/{time_key}/edge_features'][:]
                    if len(edge_features) > 0:
                        edge_timestamps = edge_features[:, 2]
                        timestamps_in_sample.extend(edge_timestamps.tolist())
            
            if timestamps_in_sample:
                min_ts = min(timestamps_in_sample)
                max_ts = max(timestamps_in_sample)
                all_timestamps.extend(timestamps_in_sample)
                
                sample_info.append({
                    'video_folder': video_folder,
                    'total_frames': total_frames,
                    'frames_with_data': frames_with_data,
                    'min_timestamp': min_ts,
                    'max_timestamp': max_ts,
                    'num_unique_timestamps': len(set(timestamps_in_sample))
                })
                all_frame_counts.append(total_frames)
    
    # Summary statistics
    print("=" * 80)
    print("TIMESTAMP ANALYSIS")
    print("=" * 80)
    
    if all_timestamps:
        print(f"\nGlobal Timestamp Range:")
        print(f"  Min: {min(all_timestamps):.6f}")
        print(f"  Max: {max(all_timestamps):.6f}")
        print(f"  Range: {max(all_timestamps) - min(all_timestamps):.6f}")
        
        print(f"\nTimestamp Statistics:")
        print(f"  Mean: {np.mean(all_timestamps):.6f}")
        print(f"  Median: {np.median(all_timestamps):.6f}")
        print(f"  Std: {np.std(all_timestamps):.6f}")
    
    print("\n" + "=" * 80)
    print("FRAME COUNT ANALYSIS")
    print("=" * 80)
    
    if all_frame_counts:
        print(f"\nFrame Count per Sample:")
        print(f"  Min: {min(all_frame_counts)}")
        print(f"  Max: {max(all_frame_counts)}")
        print(f"  Mean: {np.mean(all_frame_counts):.2f}")
        print(f"  Median: {np.median(all_frame_counts):.2f}")
        print(f"  Std: {np.std(all_frame_counts):.2f}")
        print(f"  Total samples: {len(all_frame_counts)}")
    
    print("\n" + "=" * 80)
    print("SAMPLE-LEVEL STATISTICS (first 10)")
    print("=" * 80)
    print(f"{'Sample':<50} {'Frames':<10} {'Min TS':<12} {'Max TS':<12} {'Range':<12}")
    print("-" * 80)
    
    for info in sample_info[:10]:
        ts_range = info['max_timestamp'] - info['min_timestamp']
        sample_name = os.path.basename(info['video_folder'])
        print(f"{sample_name:<50} {info['total_frames']:<10} {info['min_timestamp']:<12.6f} {info['max_timestamp']:<12.6f} {ts_range:<12.6f}")
    
    if len(sample_info) > 10:
        print(f"\n... and {len(sample_info) - 10} more samples")
    
    return {
        'global_min_timestamp': min(all_timestamps) if all_timestamps else None,
        'global_max_timestamp': max(all_timestamps) if all_timestamps else None,
        'frame_count_min': min(all_frame_counts) if all_frame_counts else None,
        'frame_count_max': max(all_frame_counts) if all_frame_counts else None,
        'frame_count_mean': np.mean(all_frame_counts) if all_frame_counts else None,
        'num_samples': len(sample_info)
    }

if __name__ == "__main__":
    # Check for manifest files
    base_dir = "/workspace/saumilya/GNN-Research"
    
    manifests = [
        f"{base_dir}/final_dataset/train_manifest.csv",
        f"{base_dir}/final_dataset/val_manifest.csv",
        f"{base_dir}/final_dataset/test_manifest.csv",
    ]
    
    all_stats = {}
    
    for manifest in manifests:
        if os.path.exists(manifest):
            print(f"\n{'='*80}")
            print(f"Analyzing: {manifest}")
            print(f"{'='*80}\n")
            stats = analyze_dataset_timestamps(manifest)
            all_stats[os.path.basename(manifest)] = stats
    
    # Combined summary
    if all_stats:
        print("\n" + "=" * 80)
        print("COMBINED SUMMARY ACROSS ALL MANIFESTS")
        print("=" * 80)
        
        all_global_mins = [s['global_min_timestamp'] for s in all_stats.values() if s['global_min_timestamp'] is not None]
        all_global_maxs = [s['global_max_timestamp'] for s in all_stats.values() if s['global_max_timestamp'] is not None]
        all_frame_mins = [s['frame_count_min'] for s in all_stats.values() if s['frame_count_min'] is not None]
        all_frame_maxs = [s['frame_count_max'] for s in all_stats.values() if s['frame_count_max'] is not None]
        
        if all_global_mins:
            print(f"\nGlobal Timestamp Range (across all sets):")
            print(f"  Min: {min(all_global_mins):.6f}")
            print(f"  Max: {max(all_global_maxs):.6f}")
        
        if all_frame_mins:
            print(f"\nFrame Count Range (across all sets):")
            print(f"  Min: {min(all_frame_mins)}")
            print(f"  Max: {max(all_frame_maxs)}")
            print(f"  Total samples analyzed: {sum(s['num_samples'] for s in all_stats.values())}")

