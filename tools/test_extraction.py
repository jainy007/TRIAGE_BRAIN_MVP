#!/usr/bin/env python3
"""
Test script for feature extraction pipeline.
Run this to verify the feature extraction works on a single clip.
"""

import json
import pandas as pd
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from feature_engineering.extract_features import process_clip, extract_motion_features, align_annotation_to_pose
from feature_engineering.pose_extractor import load_pose_data


def test_single_clip():
    """Test feature extraction on the first clip from annotations."""
    
    # Load first annotation for testing
    with open("annotated_clips.jsonl", 'r') as f:
        first_annotation = json.loads(f.readline())
    
    print("🔍 Testing with first annotation:")
    print(f"Clip: {first_annotation['clip']}")
    print(f"Segments: {len(first_annotation['segments'])}")
    for i, seg in enumerate(first_annotation['segments']):
        print(f"  Segment {i+1}: frames {seg['start']}-{seg['end']} ({seg['comment']})")
    
    # Test the full clip processing pipeline
    print("\n⚙️ Running full clip processing...")
    pose_data_dir = "/mnt/db/av_dataset/"
    
    try:
        features = process_clip(first_annotation, pose_data_dir)
        
        print(f"✅ Successfully extracted features for {len(features)} segments")
        
        # Display first segment's features
        if features:
            print(f"\n📊 Sample features for first segment:")
            first_features = features[0]
            
            # Core metrics
            print(f"  Duration: {first_features['duration_s']:.2f}s")
            print(f"  Comment: {first_features['comment']}")
            print(f"  Samples: {first_features['num_samples']}")
            
            # Motion signatures  
            print(f"  Velocity: mean={first_features['velocity_mean']:.3f}, std={first_features['velocity_std']:.3f}")
            print(f"  Acceleration: mean={first_features['acceleration_mean']:.3f}, max={first_features['acceleration_max']:.3f}")
            print(f"  Jerk RMS: {first_features['jerk_rms']:.3f}")
            
            # Behavioral features
            print(f"  Max deceleration: {first_features['max_deceleration']:.3f}")
            print(f"  Acceleration reversals: {first_features['acceleration_reversals']}")
            print(f"  Motion smoothness: {first_features['motion_smoothness']:.3f}")
            
        return features
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return []


def test_pose_loading():
    """Test pose data loading independently."""
    
    print("🔍 Testing pose data loading...")
    
    # Use the full clip name from the first annotation
    scene_dir = "01bb304d7bd835f8bbef7086b688e35e__Summer_2019"
    pose_file = f"/mnt/db/av_dataset/{scene_dir}/city_SE3_egovehicle.feather"
    
    try:
        df = load_pose_data(pose_file)
        print(f"✅ Loaded pose data: {len(df)} samples")
        print(f"  Time range: {df['timestamp_s'].min():.6f} to {df['timestamp_s'].max():.6f} seconds")
        print(f"  Velocity range: {df['velocity'].min():.3f} to {df['velocity'].max():.3f} m/s")
        print(f"  Acceleration range: {df['acceleration'].min():.3f} to {df['acceleration'].max():.3f} m/s²")
        
        # Check for extreme jerk values
        extreme_jerk = df['jerk'].abs() > 1000
        print(f"  Extreme jerk samples (>1000): {extreme_jerk.sum()}/{len(df)} ({100*extreme_jerk.sum()/len(df):.1f}%)")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading pose data: {e}")
        return None


def test_temporal_alignment():
    """Test the frame-to-timestamp alignment."""
    
    print("🔍 Testing temporal alignment...")
    
    # Load pose data
    df = test_pose_loading()
    if df is None:
        return
    
    # Test alignment with first segment
    with open("annotated_clips.jsonl", 'r') as f:
        first_annotation = json.loads(f.readline())
    
    first_segment = first_annotation['segments'][0]
    fps = first_annotation['fps']
    
    print(f"\nAligning segment: frames {first_segment['start']}-{first_segment['end']} at {fps} fps")
    print(f"Expected duration: {(first_segment['end'] - first_segment['start'])/fps:.2f} seconds")
    
    try:
        aligned_data = align_annotation_to_pose(first_segment, fps, df)
        
        if len(aligned_data) > 0:
            actual_duration = aligned_data['timestamp_s'].iloc[-1] - aligned_data['timestamp_s'].iloc[0]
            print(f"✅ Aligned {len(aligned_data)} pose samples")
            print(f"  Actual duration: {actual_duration:.2f} seconds")
            print(f"  Sample rate in segment: {len(aligned_data)/actual_duration:.1f} Hz")
        else:
            print("❌ No aligned data found")
            
    except Exception as e:
        print(f"❌ Alignment error: {e}")


if __name__ == "__main__":
    print("🚀 Testing Feature Extraction Pipeline\n")
    
    # Run tests
    test_pose_loading()
    print("\n" + "="*50 + "\n")
    
    test_temporal_alignment()
    print("\n" + "="*50 + "\n")
    
    features = test_single_clip()
    
    if features:
        print(f"\n🎯 Test completed successfully!")
        print(f"Ready to run full extraction on all {len(features)} segments.")
    else:
        print(f"\n❌ Test failed. Check the errors above.")