#!/usr/bin/env python3
"""
Lightweight Dataset Processor
- Cross-verify with annotated_clips.jsonl
- Generate MP4 clips for annotated scenes only
- Use multiprocessing for speed
"""

import os
import glob
import json
import cv2
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Dict, List

# HARDCODED PATHS
DATASET_ROOT = "/mnt/db/av_dataset"
ANNOTATIONS_FILE = "/home/jainy007/PEM/mvp3_0/annotated_clips.jsonl"
OUTPUT_DIR = "outputs"
REPORTS_DIR = f"{OUTPUT_DIR}/reports"
CLIPS_DIR = f"{OUTPUT_DIR}/clips"

def load_annotated_scenes() -> set:
    """Load annotated scene IDs from JSONL file"""
    annotated_scenes = set()
    
    print(f"📝 Loading annotations from: {ANNOTATIONS_FILE}")
    
    if not os.path.exists(ANNOTATIONS_FILE):
        print(f"❌ Annotations file not found: {ANNOTATIONS_FILE}")
        return annotated_scenes
    
    with open(ANNOTATIONS_FILE, 'r') as f:
        for line_num, line in enumerate(f, 1):
            annotation = json.loads(line)
            clip_name = annotation['clip']
            
            # Extract scene ID from clip name (remove .mp4 and other suffixes)
            scene_id = clip_name.replace('.mp4', '').split('__')[0]
            
            # DON'T convert anything - use the scene ID as-is from the clip name
            annotated_scenes.add(scene_id)
            
            # Debug first few entries
            if line_num <= 3:
                print(f"  Clip {line_num}: {clip_name} -> Scene ID: {scene_id}")
    
    print(f"✅ Found {len(annotated_scenes)} annotated scenes")
    
    # Show sample of what we're looking for
    sample_scenes = list(annotated_scenes)[:5]
    print(f"🔍 Sample scene IDs to match: {sample_scenes}")
    
    return annotated_scenes

def find_valid_scenes(annotated_scenes: set) -> List[Dict]:
    """Find scenes with ring_front_center images AND pose data AND in annotations"""
    print(f"🔍 Scanning dataset: {DATASET_ROOT}")
    
    # Find all potential scene directories - look for directories that START with annotated scene IDs
    scene_dirs = []
    
    for root, dirs, files in os.walk(DATASET_ROOT):
        for dir_name in dirs:
            # Check if this directory name starts with any of our annotated scene IDs
            for annotated_id in annotated_scenes:
                if dir_name.startswith(annotated_id):
                    scene_dirs.append((os.path.join(root, dir_name), annotated_id))
                    print(f"✅ Found: {dir_name} -> {annotated_id}")
                    break
    
    print(f"Found {len(scene_dirs)} matching scene directories")
    
    valid_scenes = []
    
    for scene_path, scene_id in scene_dirs:
        print(f"🔍 Checking scene: {os.path.basename(scene_path)}")
        
        # Check for ring_front_center images
        camera_path = os.path.join(scene_path, "sensors", "cameras", "ring_front_center")
        if not os.path.exists(camera_path):
            print(f"❌ {scene_id}: No ring_front_center camera")
            continue
        
        # Count JPEG files
        jpeg_files = glob.glob(os.path.join(camera_path, "*.jpg"))
        if not jpeg_files:
            print(f"❌ {scene_id}: No JPEG files")
            continue
        
        # Check for ANY pose data file
        pose_patterns = [
            "annotations.feather",
            "city_SE3_egovehicle.feather", 
            "poses.feather",
            "ego_pose.feather"
        ]
        
        pose_file = None
        for pattern in pose_patterns:
            potential_pose = os.path.join(scene_path, pattern)
            if os.path.exists(potential_pose) and os.path.getsize(potential_pose) > 0:
                pose_file = potential_pose
                break
        
        if not pose_file:
            print(f"❌ {scene_id}: No valid pose data")
            continue
        
        print(f"✅ {scene_id}: {len(jpeg_files)} frames + pose data")
        
        valid_scenes.append({
            'scene_id': scene_id,  # Use the original scene ID
            'scene_path': scene_path,
            'camera_path': camera_path,
            'pose_path': pose_file,
            'num_frames': len(jpeg_files),
            'jpeg_files': sorted(jpeg_files)
        })
    
    return valid_scenes

def create_mp4_clip(scene_data: Dict) -> bool:
    """Create MP4 clip from JPEG sequence"""
    scene_id = scene_data['scene_id']
    jpeg_files = scene_data['jpeg_files']
    
    output_path = os.path.join(CLIPS_DIR, f"{scene_id}.mp4")
    
    # Skip if already exists
    if os.path.exists(output_path):
        print(f"⏭️ {scene_id}: MP4 already exists")
        return True
    
    try:
        # Read first frame to get dimensions
        first_frame = cv2.imread(jpeg_files[0])
        if first_frame is None:
            print(f"❌ {scene_id}: Cannot read first frame")
            return False
        
        height, width, _ = first_frame.shape
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 10.0  # Standard for Argoverse
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frames_written = 0
        for jpeg_file in jpeg_files:
            frame = cv2.imread(jpeg_file)
            if frame is not None:
                writer.write(frame)
                frames_written += 1
        
        writer.release()
        
        if frames_written > 0:
            print(f"✅ {scene_id}: Created MP4 with {frames_written} frames")
            return True
        else:
            print(f"❌ {scene_id}: No frames written")
            return False
            
    except Exception as e:
        print(f"❌ {scene_id}: MP4 creation failed - {e}")
        return False

def generate_csv(valid_scenes: List[Dict]):
    """Generate CSV with valid scenes"""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    csv_data = []
    for scene in valid_scenes:
        csv_data.append({
            'scene_id': scene['scene_id'],
            'path': scene['scene_path'],
            'camera_path': scene['camera_path'],
            'pose_path': scene['pose_path'],
            'num_frames_total': scene['num_frames'],
            'mp4_path': os.path.join(CLIPS_DIR, f"{scene['scene_id']}.mp4")
        })
    
    df = pd.DataFrame(csv_data)
    csv_path = os.path.join(REPORTS_DIR, 'annotated_scenes.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"📄 Generated CSV: {csv_path}")
    return csv_path

def main():
    """Main processing pipeline"""
    print("🚀 Lightweight Dataset Processor")
    print("="*50)
    
    # Create output directories
    os.makedirs(CLIPS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    # Step 1: Load annotated scenes
    annotated_scenes = load_annotated_scenes()
    if not annotated_scenes:
        print("❌ No annotated scenes found. Exiting.")
        return
    
    # Step 2: Find valid scenes
    valid_scenes = find_valid_scenes(annotated_scenes)
    
    if not valid_scenes:
        print("❌ No valid scenes found. Exiting.")
        return
    
    print(f"\n📊 FOUND {len(valid_scenes)} VALID ANNOTATED SCENES")
    
    # Step 3: Generate CSV
    csv_path = generate_csv(valid_scenes)
    
    # Step 4: Create MP4 clips using multiprocessing
    print(f"\n🎬 Creating MP4 clips using {cpu_count()} processes...")
    
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(create_mp4_clip, valid_scenes)
    
    # Summary
    successful_clips = sum(results)
    print(f"\n✅ SUMMARY:")
    print(f"  - Valid annotated scenes: {len(valid_scenes)}")
    print(f"  - MP4 clips created: {successful_clips}")
    print(f"  - Failed: {len(valid_scenes) - successful_clips}")
    print(f"  - CSV report: {csv_path}")
    print(f"  - MP4 clips: {CLIPS_DIR}")

if __name__ == "__main__":
    main()