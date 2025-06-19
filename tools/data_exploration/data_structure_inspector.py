#!/usr/bin/env python3
"""
Quick inspector to understand your data structure for GUI development
"""

import pandas as pd
import json
import os
from pathlib import Path

def inspect_data_structure():
    """Inspect the actual data structure"""
    
    print("🔍 INSPECTING DATA STRUCTURE FOR GUI")
    print("=" * 50)
    
    # Load scene summary
    try:
        df = pd.read_csv("scene_summary.csv")
        print(f"✅ Loaded scene_summary.csv: {len(df)} scenes")
        
        # Show first few rows
        print("\n📊 Sample scenes:")
        print(df.head(3).to_string())
        
        # Check what camera views are available
        if 'camera_views' in df.columns:
            unique_views = df['camera_views'].value_counts()
            print(f"\n📹 Camera views available:")
            print(unique_views.head())
        
        # Check a sample scene directory
        sample_scene = df.iloc[0]
        scene_path = sample_scene['path']
        print(f"\n📁 Sample scene path: {scene_path}")
        
        if os.path.exists(scene_path):
            files = list(Path(scene_path).rglob("*"))[:10]
            print("Sample files in scene:")
            for f in files:
                print(f"  • {f}")
        
        # Check pose data
        if 'pose_path' in df.columns and pd.notna(sample_scene['pose_path']):
            pose_path = sample_scene['pose_path']
            print(f"\n🚶 Sample pose path: {pose_path}")
            
            if os.path.exists(pose_path):
                print(f"Pose file exists: ✅")
                # Try to peek at pose data
                if pose_path.endswith('.json'):
                    try:
                        with open(pose_path, 'r') as f:
                            pose_data = json.load(f)
                        print("Pose data structure:")
                        if isinstance(pose_data, dict):
                            print(f"  Keys: {list(pose_data.keys())[:5]}")
                        elif isinstance(pose_data, list):
                            print(f"  List with {len(pose_data)} items")
                            if pose_data:
                                print(f"  First item keys: {list(pose_data[0].keys())[:5] if isinstance(pose_data[0], dict) else 'Not a dict'}")
                    except Exception as e:
                        print(f"  Could not read pose JSON: {e}")
            else:
                print(f"Pose file missing: ❌")
    
    except Exception as e:
        print(f"❌ Error loading scene_summary.csv: {e}")
    
    # Check other data files
    print(f"\n📋 Other data files:")
    data_files = ["scene_core_map.jsonl", "scene_full_map.jsonl", "scene_sensor_map.jsonl"]
    
    for file in data_files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
            # Quick peek at JSONL structure
            try:
                with open(file, 'r') as f:
                    first_line = f.readline()
                    sample = json.loads(first_line)
                    print(f"     Sample keys: {list(sample.keys())[:5]}")
            except:
                print(f"     Could not parse JSONL")
        else:
            print(f"  ❌ {file}")
    
    print(f"\n🎯 GUI REQUIREMENTS ASSESSMENT:")
    print("For the Tkinter GUI, we need:")
    print("  1. Video files in scene directories")
    print("  2. Pose/motion data for graphs") 
    print("  3. Feature extraction pipeline")
    print("  4. Ensemble model integration")
    
    return df if 'df' in locals() else None

if __name__ == "__main__":
    # Run from the data_exploration directory
    inspect_data_structure()