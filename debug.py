#!/usr/bin/env python3
"""
Debug script to check why analysis is finding wrong frames
"""

import json
import numpy as np
import pandas as pd

def debug_analysis_results():
    """Debug the cached analysis to see what's happening"""
    
    # Check the cached analysis
    cache_file = "cache/analysis_0OF3lawgf9vsILBoZVT27MF8zDNuq7zh.pkl"
    
    try:
        import pickle
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        
        print("🔍 DEBUGGING ANALYSIS RESULTS")
        print("="*50)
        
        # Show what the analysis found
        frame_predictions = cached_data.get('frame_predictions', [])
        all_clusters = cached_data.get('all_clusters', [])
        top_clusters = cached_data.get('top_clusters', [])
        
        print(f"Total frame predictions: {len(frame_predictions)}")
        print(f"All clusters found: {len(all_clusters)}")
        print(f"Top clusters selected: {len(top_clusters)}")
        
        # Show frame predictions around the annotated area (frames 378-482)
        print(f"\n📊 FRAME PREDICTIONS AROUND ANNOTATION (378-482):")
        annotated_frames = [p for p in frame_predictions if 370 <= p['frame_idx'] <= 490]
        
        for pred in annotated_frames[:10]:  # Show first 10
            print(f"Frame {pred['frame_idx']}: {pred['classification']} (conf: {pred['confidence']:.3f}, risk: {pred['risk_level']})")
        
        print(f"\n📊 ALL CLUSTERS FOUND:")
        for i, cluster in enumerate(all_clusters):
            start_frame = cluster['start_frame']
            end_frame = cluster['end_frame']
            classification = cluster['classification']
            score = cluster['final_score']
            duration = cluster['duration_seconds']
            
            # Check if this overlaps with annotation
            overlaps_annotation = not (end_frame < 378 or start_frame > 482)
            overlap_mark = "✅ OVERLAPS ANNOTATION" if overlaps_annotation else "❌ No overlap"
            
            print(f"Cluster {i+1}: {classification} frames {start_frame}-{end_frame} ({duration:.1f}s, score: {score:.3f}) {overlap_mark}")
        
        print(f"\n📊 TOP CLUSTERS (what gets displayed):")
        for i, cluster in enumerate(top_clusters):
            start_frame = cluster['start_frame']
            end_frame = cluster['end_frame']
            classification = cluster['classification']
            score = cluster['final_score']
            
            overlaps_annotation = not (end_frame < 378 or start_frame > 482)
            overlap_mark = "✅ OVERLAPS ANNOTATION" if overlaps_annotation else "❌ No overlap"
            
            print(f"Top cluster {i+1}: {classification} frames {start_frame}-{end_frame} (score: {score:.3f}) {overlap_mark}")
        
        # Check if there are any clusters that DO overlap with annotation but weren't selected as top
        print(f"\n🔍 CHECKING FOR MISSED OVERLAPPING CLUSTERS:")
        overlapping_clusters = [c for c in all_clusters if not (c['end_frame'] < 378 or c['start_frame'] > 482)]
        
        if overlapping_clusters:
            print(f"Found {len(overlapping_clusters)} clusters overlapping with annotation:")
            for cluster in overlapping_clusters:
                print(f"  - {cluster['classification']} frames {cluster['start_frame']}-{cluster['end_frame']} (score: {cluster['final_score']:.3f})")
        else:
            print("❌ NO clusters found overlapping with annotation area!")
            print("This suggests the analysis algorithm is not detecting risk in the annotated region.")
        
        # Show risk distribution
        print(f"\n📊 RISK DISTRIBUTION:")
        risk_counts = {}
        for pred in frame_predictions:
            risk = pred['risk_level']
            risk_counts[risk] = risk_counts.get(risk, 0) + 1
        
        for risk, count in risk_counts.items():
            print(f"  {risk}: {count} frames ({count/len(frame_predictions)*100:.1f}%)")
        
    except Exception as e:
        print(f"❌ Failed to debug: {e}")

if __name__ == "__main__":
    debug_analysis_results()