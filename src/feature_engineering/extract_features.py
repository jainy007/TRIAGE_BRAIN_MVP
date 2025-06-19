import json
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from pathlib import Path
from typing import Dict, List, Any
from pose_extractor import load_pose_data


def smooth_motion_data(df: pd.DataFrame, window_length: int = 5, polyorder: int = 2) -> pd.DataFrame:
    """
    Apply Savitzky-Golay smoothing to velocity before computing acceleration and jerk.
    This reduces noise amplification in higher-order derivatives.
    """
    df_smooth = df.copy()
    
    # Only smooth if we have enough data points
    if len(df_smooth) >= window_length:
        # First, handle any extremely small dt values that cause numerical issues
        df_smooth['dt'] = df_smooth['dt'].clip(lower=1e-6)  # Minimum 1 microsecond
        
        # Smooth velocity using Savitzky-Golay filter
        df_smooth['velocity_smooth'] = savgol_filter(
            df_smooth['velocity'], 
            window_length=window_length, 
            polyorder=polyorder
        )
        
        # Recompute acceleration and jerk from smoothed velocity
        df_smooth['acceleration_smooth'] = df_smooth['velocity_smooth'].diff() / df_smooth['dt']
        df_smooth['jerk_smooth'] = df_smooth['acceleration_smooth'].diff() / df_smooth['dt']
        
        # Fill NaNs and handle any remaining infinite values
        df_smooth['acceleration_smooth'] = df_smooth['acceleration_smooth'].fillna(0)
        df_smooth['jerk_smooth'] = df_smooth['jerk_smooth'].fillna(0)
        
        # Clip any remaining extreme values immediately after computation
        df_smooth['acceleration_smooth'] = df_smooth['acceleration_smooth'].clip(-100, 100)  # ±100 m/s²
        df_smooth['jerk_smooth'] = df_smooth['jerk_smooth'].clip(-1000, 1000)  # ±1000 m/s³
        
    else:
        # Not enough data for smoothing, use original values but still clip
        df_smooth['velocity_smooth'] = df_smooth['velocity'].clip(-50, 50)  # ±50 m/s
        df_smooth['acceleration_smooth'] = df_smooth['acceleration'].clip(-100, 100)
        df_smooth['jerk_smooth'] = df_smooth['jerk'].clip(-1000, 1000)
    
    return df_smooth


def clip_outliers(series: pd.Series, max_value: float) -> pd.Series:
    """
    Clip extreme values that are physically unrealistic.
    """
    return series.clip(-max_value, max_value)


def align_annotation_to_pose(segment: Dict, fps: float, pose_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert frame-based annotation segment to timestamp-aligned pose data.
    
    Since pose data covers ~51.5% of video duration, we map video time proportionally
    to the available pose timerange.
    
    Args:
        segment: Dict with 'start', 'end' frame indices
        fps: Frames per second of the video
        pose_df: DataFrame with pose data including 'timestamp_s'
    
    Returns:
        Subset of pose_df corresponding to the annotation segment
    """
    # Convert frame indices to seconds within the video
    start_time_in_video = segment['start'] / fps
    end_time_in_video = segment['end'] / fps
    
    # Get pose data time range
    pose_start_time = pose_df['timestamp_s'].iloc[0]
    pose_end_time = pose_df['timestamp_s'].iloc[-1]
    pose_duration = pose_end_time - pose_start_time
    
    # Since pose data covers first ~51.5% of video, check if segment is within coverage
    # For direct mapping: video_time maps to pose_time within the pose duration
    if start_time_in_video > pose_duration:
        print(f"Warning: Segment starts after pose coverage ({start_time_in_video:.1f}s > {pose_duration:.1f}s)")
        return pd.DataFrame()
    
    if end_time_in_video > pose_duration:
        print(f"Warning: Segment extends beyond pose coverage, clipping from {end_time_in_video:.1f}s to {pose_duration:.1f}s")
        end_time_in_video = pose_duration
    
    # Map video times to absolute pose timestamps
    start_timestamp = pose_start_time + start_time_in_video
    end_timestamp = pose_start_time + end_time_in_video
    
    print(f"Debug: Segment frames {segment['start']}-{segment['end']} -> {start_time_in_video:.2f}-{end_time_in_video:.2f}s in video")
    print(f"Debug: Pose coverage: 0 to {pose_duration:.2f}s")
    print(f"Debug: Mapping to pose timestamps: {start_timestamp:.1f} to {end_timestamp:.1f}s")
    
    # Filter pose data to this time window
    mask = (pose_df['timestamp_s'] >= start_timestamp) & (pose_df['timestamp_s'] <= end_timestamp)
    segment_data = pose_df[mask].copy()
    
    if len(segment_data) == 0:
        print(f"Warning: No pose data found for segment {segment['start']}-{segment['end']}")
        return pd.DataFrame()
    
    actual_duration = segment_data['timestamp_s'].iloc[-1] - segment_data['timestamp_s'].iloc[0]
    expected_duration = end_time_in_video - start_time_in_video
    print(f"Debug: Found {len(segment_data)} samples, actual duration: {actual_duration:.2f}s (expected: {expected_duration:.2f}s)")
    
    return segment_data


def extract_motion_features(segment_data: pd.DataFrame, segment_info: Dict) -> Dict[str, Any]:
    """
    Extract comprehensive motion features from a pose data segment.
    
    Args:
        segment_data: DataFrame with pose data for this segment
        segment_info: Dict with segment metadata (start, end, comment)
    
    Returns:
        Dictionary of extracted features
    """
    if len(segment_data) == 0:
        return {}
    
    # Apply smoothing and outlier clipping
    segment_data = smooth_motion_data(segment_data)
    
    # Clip extreme values (physically realistic thresholds)
    segment_data['velocity_smooth'] = clip_outliers(segment_data['velocity_smooth'], 50.0)  # 50 m/s max
    segment_data['acceleration_smooth'] = clip_outliers(segment_data['acceleration_smooth'], 20.0)  # 20 m/s² max
    segment_data['jerk_smooth'] = clip_outliers(segment_data['jerk_smooth'], 50.0)  # 50 m/s³ max
    
    # Duration and temporal features
    duration_s = segment_data['timestamp_s'].iloc[-1] - segment_data['timestamp_s'].iloc[0]
    num_samples = len(segment_data)
    
    # Basic motion statistics
    features = {
        # Metadata
        'start_frame': segment_info['start'],
        'end_frame': segment_info['end'],
        'comment': segment_info['comment'],
        'duration_s': duration_s,
        'num_samples': num_samples,
        'sample_rate_hz': num_samples / duration_s if duration_s > 0 else 0,
        
        # Velocity features
        'velocity_mean': segment_data['velocity_smooth'].mean(),
        'velocity_std': segment_data['velocity_smooth'].std(),
        'velocity_min': segment_data['velocity_smooth'].min(),
        'velocity_max': segment_data['velocity_smooth'].max(),
        'velocity_range': segment_data['velocity_smooth'].max() - segment_data['velocity_smooth'].min(),
        
        # Acceleration features
        'acceleration_mean': segment_data['acceleration_smooth'].mean(),
        'acceleration_std': segment_data['acceleration_smooth'].std(),
        'acceleration_min': segment_data['acceleration_smooth'].min(),
        'acceleration_max': segment_data['acceleration_smooth'].max(),
        'acceleration_range': segment_data['acceleration_smooth'].max() - segment_data['acceleration_smooth'].min(),
        
        # Jerk features
        'jerk_mean': segment_data['jerk_smooth'].mean(),
        'jerk_std': segment_data['jerk_smooth'].std(),
        'jerk_min': segment_data['jerk_smooth'].min(),
        'jerk_max': segment_data['jerk_smooth'].max(),
        'jerk_rms': np.sqrt(np.mean(segment_data['jerk_smooth'] ** 2)),
    }
    
    # Behavioral signature features
    
    # 1. Overshoot detection (deceleration patterns)
    negative_accel = segment_data['acceleration_smooth'][segment_data['acceleration_smooth'] < 0]
    if len(negative_accel) > 0:
        features['max_deceleration'] = negative_accel.min()  # Most negative = strongest braking
        features['deceleration_events'] = len(negative_accel)
    else:
        features['max_deceleration'] = 0.0
        features['deceleration_events'] = 0
    
    # 2. Hesitation detection (stop-go patterns)
    # Count velocity zero crossings (speed ups and slow downs)
    velocity_signs = np.sign(segment_data['velocity_smooth'])
    velocity_zero_crossings = np.sum(np.abs(np.diff(velocity_signs)) > 0)
    features['velocity_zero_crossings'] = velocity_zero_crossings
    
    # Count acceleration reversals (changing from speeding up to slowing down)
    accel_signs = np.sign(segment_data['acceleration_smooth'])
    accel_reversals = np.sum(np.abs(np.diff(accel_signs)) > 0)
    features['acceleration_reversals'] = accel_reversals
    
    # 3. Smoothness metrics
    # High jerk RMS indicates jerky, non-smooth motion
    features['motion_smoothness'] = 1.0 / (1.0 + features['jerk_rms'])  # Higher = smoother
    
    # 4. Duration-normalized features (to handle varying segment lengths)
    if duration_s > 0:
        features['jerk_per_second'] = features['jerk_rms'] / duration_s
        features['accel_changes_per_second'] = accel_reversals / duration_s
        features['distance_traveled'] = segment_data['velocity_smooth'].mean() * duration_s
    else:
        features['jerk_per_second'] = 0.0
        features['accel_changes_per_second'] = 0.0
        features['distance_traveled'] = 0.0
    
    return features


def process_clip(annotation_entry: Dict, pose_data_dir: str) -> List[Dict[str, Any]]:
    """
    Process all segments in a single annotated clip.
    
    Args:
        annotation_entry: Single entry from annotated_clips.jsonl
        pose_data_dir: Directory containing pose feather files
    
    Returns:
        List of feature dictionaries, one per segment
    """
    clip_name = annotation_entry['clip']
    fps = annotation_entry['fps']
    
    # Extract full clip name (including season) - no splitting needed
    # The directory name matches the full clip name minus .mp4 extension
    scene_dir = clip_name.replace('.mp4', '')
    pose_file = Path(pose_data_dir) / scene_dir / "city_SE3_egovehicle.feather"
    
    if not pose_file.exists():
        print(f"Warning: Pose file not found: {pose_file}")
        return []
    
    # Load pose data
    try:
        pose_df = load_pose_data(str(pose_file))
    except Exception as e:
        print(f"Error loading pose data for {clip_name}: {e}")
        return []
    
    # Process each segment
    segment_features = []
    for segment in annotation_entry['segments']:
        # Align annotation to pose data
        segment_data = align_annotation_to_pose(segment, fps, pose_df)
        
        if len(segment_data) == 0:
            continue
        
        # Extract features
        features = extract_motion_features(segment_data, segment)
        
        if features:  # Only add if extraction was successful
            # Add clip-level metadata
            features['clip_name'] = clip_name
            features['scene_id'] = scene_dir
            features['total_frames'] = annotation_entry['total_frames']
            features['fps'] = fps
            
            segment_features.append(features)
    
    return segment_features


def extract_features_from_annotations(
    annotations_file: str, 
    pose_data_dir: str, 
    output_file: str = "feature_vectors.jsonl"
) -> None:
    """
    Main function to extract features from all annotated clips.
    
    Args:
        annotations_file: Path to annotated_clips.jsonl
        pose_data_dir: Directory containing pose data (e.g., /mnt/db/av_dataset/sensor/train/)
        output_file: Output file for extracted features
    """
    all_features = []
    
    # Read annotations
    with open(annotations_file, 'r') as f:
        annotations = [json.loads(line) for line in f]
    
    print(f"Processing {len(annotations)} annotated clips...")
    
    # Process each clip
    for i, annotation_entry in enumerate(annotations):
        print(f"Processing clip {i+1}/{len(annotations)}: {annotation_entry['clip']}")
        
        clip_features = process_clip(annotation_entry, pose_data_dir)
        all_features.extend(clip_features)
        
        print(f"  Extracted {len(clip_features)} segment features")
    
    # Save results
    with open(output_file, 'w') as f:
        for features in all_features:
            # Convert numpy types to native Python types for JSON serialization
            json_features = {}
            for key, value in features.items():
                if hasattr(value, 'item'):  # numpy scalar
                    json_features[key] = value.item()
                elif isinstance(value, np.integer):
                    json_features[key] = int(value)
                elif isinstance(value, np.floating):
                    json_features[key] = float(value)
                else:
                    json_features[key] = value
            
            f.write(json.dumps(json_features) + '\n')
    
    print(f"\n✅ Extraction complete!")
    print(f"📊 Total segments processed: {len(all_features)}")
    print(f"💾 Features saved to: {output_file}")


if __name__ == "__main__":
    # Example usage
    annotations_file = "assets/data/annotated_clips.jsonl"
    pose_data_dir = "/mnt/db/av_dataset/"
    output_file = "feature_vectors.jsonl"
    
    extract_features_from_annotations(annotations_file, pose_data_dir, output_file)