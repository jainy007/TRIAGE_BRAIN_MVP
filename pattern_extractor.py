#!/usr/bin/env python3
"""
Direct Pattern Learning - Learn directly from annotated segments
"""

import torch
import numpy as np
import json
from collections import Counter, defaultdict
import sys
import os

class DirectPatternLearner:
    """Learn patterns directly from annotated segments"""
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.patterns = {}  # behavior -> list of feature patterns
        
    def extract_annotated_patterns(self):
        """Extract feature patterns directly from annotated segments"""
        
        print("🎯 Extracting patterns from annotated segments...")
        
        # Load annotations
        annotations = {}
        with open("assets/data/annotated_clips.jsonl", 'r') as f:
            for line in f:
                clip_data = json.loads(line)
                annotations[clip_data['clip']] = clip_data['segments']
        
        print(f"   Found {len(annotations)} annotated clips")
        
        # Load feature vectors
        feature_vectors = []
        with open("assets/data/feature_vectors.jsonl", 'r') as f:
            for line in f:
                feature_vectors.append(json.loads(line))
        
        print(f"   Found {len(feature_vectors)} feature vectors")
        
        # Extract patterns for each behavior
        behavior_patterns = defaultdict(list)
        matches_found = 0
        total_segments = 0
        
        for clip_name, segments in annotations.items():
            for segment in segments:
                total_segments += 1
                start_frame = segment['start']
                end_frame = segment['end']
                comment = segment['comment']
                behavior = self._map_comment_to_standard(comment)
                
                # Find feature vectors that overlap with this segment
                matching_features = []
                for fv in feature_vectors:
                    if fv['clip_name'] != clip_name:
                        continue
                    
                    fv_start = fv['start_frame']
                    fv_end = fv['end_frame']
                    
                    # Check if feature vector overlaps with segment
                    overlap_start = max(start_frame, fv_start)
                    overlap_end = min(end_frame, fv_end)
                    
                    if overlap_start <= overlap_end:
                        # Calculate overlap percentage
                        overlap_length = overlap_end - overlap_start + 1
                        fv_length = fv_end - fv_start + 1
                        overlap_pct = overlap_length / fv_length
                        
                        if overlap_pct >= 0.5:  # At least 50% overlap
                            matching_features.append(fv)
                
                if matching_features:
                    matches_found += 1
                    
                    # Extract feature patterns from all matching feature vectors
                    for fv in matching_features:
                        pattern = self._extract_feature_pattern(fv)
                        behavior_patterns[behavior].append({
                            'pattern': pattern,
                            'comment': comment,
                            'clip': clip_name,
                            'frames': (fv['start_frame'], fv['end_frame']),
                            'overlap_frames': (start_frame, end_frame)
                        })
                    
                    print(f"   ✅ {clip_name}: {comment} → {behavior} ({len(matching_features)} feature vectors)")
                else:
                    print(f"   ❌ {clip_name}: {comment} → NO MATCHING FEATURES")
        
        print(f"\n📊 Pattern extraction results:")
        print(f"   Total segments: {total_segments}")
        print(f"   Segments with features: {matches_found}")
        print(f"   Match rate: {matches_found/total_segments*100:.1f}%")
        
        print(f"\n📋 Patterns extracted by behavior:")
        for behavior, patterns in behavior_patterns.items():
            print(f"   {behavior}: {len(patterns)} patterns")
        
        return dict(behavior_patterns)
    
    def _extract_feature_pattern(self, feature_vector):
        """Extract the feature pattern from a feature vector"""
        
        # Define the features to use (same as your DL model)
        dl_features = [
            'velocity_mean', 'velocity_std', 'velocity_min', 'velocity_max', 'velocity_range',
            'acceleration_mean', 'acceleration_std', 'acceleration_min', 'acceleration_max', 'acceleration_range',
            'jerk_mean', 'jerk_std', 'jerk_min', 'jerk_max', 'jerk_rms',
            'max_deceleration', 'motion_smoothness', 'acceleration_reversals',
            'velocity_zero_crossings', 'deceleration_events', 'jerk_per_second',
            'accel_changes_per_second', 'distance_traveled', 'duration_s'
        ]
        
        pattern = []
        for feature_name in dl_features:
            value = feature_vector.get(feature_name, 0.0)
            if np.isnan(value) or np.isinf(value):
                value = 0.0
            pattern.append(float(value))
        
        return pattern
    
    def _map_comment_to_standard(self, comment: str) -> str:
        """Map comment to standard behavior class"""
        
        comment_lower = comment.lower()
        
        # Direct mapping based on your actual comments
        if 'overshoot' in comment_lower:
            return 'overshoot'
        elif 'near miss' in comment_lower or 'nearmiss' in comment_lower:
            return 'nearmiss'
        elif 'man crossing' in comment_lower or 'pedestrian' in comment_lower or 'women' in comment_lower:
            return 'pedestrian'
        elif 'bicycle' in comment_lower:
            return 'bicycle'
        elif 'overste' in comment_lower:  # oversteering
            return 'oversteering'
        elif 'hesitation' in comment_lower:
            return 'hesitation'
        elif 'halt' in comment_lower or 'stop' in comment_lower:
            return 'stop_behavior'
        elif 'traffic' in comment_lower or 'cones' in comment_lower:
            return 'obstacles'
        elif 'turn' in comment_lower and ('left' in comment_lower or 'right' in comment_lower):
            return 'turning'
        elif 'lane' in comment_lower:
            return 'lane_change'
        elif 'park' in comment_lower:
            return 'parking'
        elif 'leading' in comment_lower or 'vehicle' in comment_lower:
            return 'vehicle_interaction'
        elif 'road color' in comment_lower:
            return 'road_conditions'
        else:
            return 'other'
    
    def create_training_dataset(self, behavior_patterns):
        """Create training dataset from extracted patterns"""
        
        print("🔄 Creating training dataset...")
        
        # Create examples
        training_examples = []
        
        for behavior, patterns in behavior_patterns.items():
            for pattern_data in patterns:
                example = {
                    'features': pattern_data['pattern'],
                    'label': behavior,
                    'comment': pattern_data['comment'],
                    'clip': pattern_data['clip'],
                    'frames': pattern_data['frames']
                }
                training_examples.append(example)
        
        if not training_examples:
            print("❌ No training examples created!")
            return None, None, None, None
        
        # Create class mapping
        unique_behaviors = sorted(set(ex['label'] for ex in training_examples))
        class_to_idx = {behavior: idx for idx, behavior in enumerate(unique_behaviors)}
        
        print(f"📝 Classes found:")
        behavior_counts = Counter([ex['label'] for ex in training_examples])
        for behavior, count in behavior_counts.most_common():
            idx = class_to_idx[behavior]
            print(f"   {idx}: {behavior} ({count} examples)")
        
        # Convert to tensors
        X = torch.FloatTensor([ex['features'] for ex in training_examples])
        y = torch.LongTensor([class_to_idx[ex['label']] for ex in training_examples])
        
        metadata = [{
            'comment': ex['comment'],
            'clip': ex['clip'], 
            'frames': ex['frames'],
            'label': ex['label']
        } for ex in training_examples]
        
        print(f"✅ Dataset created: {X.shape} features, {y.shape} labels")
        
        return X, y, metadata, class_to_idx
    
    def test_pattern_matching(self, model, test_features, class_to_idx):
        """Test the model on new feature vectors"""
        
        print("🧪 Testing pattern matching...")
        
        model.eval()
        idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
        
        predictions = []
        
        with torch.no_grad():
            for i, fv in enumerate(test_features):
                # Extract pattern
                pattern = self._extract_feature_pattern(fv)
                
                # Create sequence (repeat pattern for temporal model)
                sequence = [pattern] * 8  # Your model expects sequences
                sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
                
                # Predict
                behavior_logits, risk_score = model(sequence_tensor)
                probs = torch.softmax(behavior_logits, dim=-1)
                predicted_idx = torch.argmax(probs).item()
                confidence = torch.max(probs).item()
                predicted_behavior = idx_to_class[predicted_idx]
                
                prediction = {
                    'clip': fv['clip_name'],
                    'frames': (fv['start_frame'], fv['end_frame']),
                    'predicted_behavior': predicted_behavior,
                    'confidence': confidence,
                    'risk_score': risk_score.item()
                }
                
                predictions.append(prediction)
                
                if i < 10:  # Show first 10
                    print(f"   {fv['clip_name']} frames {fv['start_frame']}-{fv['end_frame']}: {predicted_behavior} (conf: {confidence:.3f})")
        
        return predictions
    
    def find_behavioral_segments(self, predictions, confidence_threshold=0.7):
        """Find high-confidence behavioral segments"""
        
        print(f"🔍 Finding behavioral segments (confidence > {confidence_threshold})...")
        
        # Filter high-confidence non-other predictions
        behavioral_predictions = [
            p for p in predictions 
            if p['predicted_behavior'] != 'other' and p['confidence'] > confidence_threshold
        ]
        
        # Group by clip
        clip_segments = defaultdict(list)
        for pred in behavioral_predictions:
            clip_segments[pred['clip']].append(pred)
        
        # Sort by frame for each clip
        for clip in clip_segments:
            clip_segments[clip].sort(key=lambda x: x['frames'][0])
        
        print(f"📊 Found {len(behavioral_predictions)} high-confidence behavioral segments in {len(clip_segments)} clips")
        
        for clip, segments in clip_segments.items():
            print(f"\n📽️  {clip}:")
            for segment in segments:
                frames = segment['frames']
                behavior = segment['predicted_behavior']
                confidence = segment['confidence']
                duration = (frames[1] - frames[0] + 1) / 10.0  # assuming 10fps
                print(f"   {frames[0]:4d}-{frames[1]:4d}: {behavior} (conf: {confidence:.3f}, {duration:.1f}s)")
        
        return dict(clip_segments)

def main():
    """Run direct pattern learning"""
    
    learner = DirectPatternLearner()
    
    # Extract patterns from annotations
    behavior_patterns = learner.extract_annotated_patterns()
    
    if not behavior_patterns:
        print("❌ No patterns extracted")
        return
    
    # Create training dataset
    X, y, metadata, class_to_idx = learner.create_training_dataset(behavior_patterns)
    
    if X is None:
        print("❌ Failed to create dataset")
        return
    
    # Save dataset
    dataset = {
        'features': X,
        'labels': y,
        'metadata': metadata,
        'class_mapping': class_to_idx,
        'approach': 'direct_pattern_learning'
    }
    
    torch.save(dataset, 'direct_pattern_dataset.pt')
    print(f"💾 Dataset saved to: direct_pattern_dataset.pt")
    
    # Test on a subset of feature vectors (simulate finding patterns in new data)
    print(f"\n🎯 Testing pattern recognition...")
    
    # Load some feature vectors for testing
    test_features = []
    with open("assets/data/feature_vectors.jsonl", 'r') as f:
        for i, line in enumerate(f):
            if i >= 100:  # Test on first 100 feature vectors
                break
            test_features.append(json.loads(line))
    
    if test_features:
        print(f"   Testing on {len(test_features)} feature vectors...")
        # Note: You'll need to load your trained model here
        # predictions = learner.test_pattern_matching(model, test_features, class_to_idx)
        # segments = learner.find_behavioral_segments(predictions)
        print("   (Load your trained model to test pattern matching)")
    
    print(f"\n✅ DIRECT PATTERN LEARNING COMPLETE!")
    print(f"   • Extracted patterns from {sum(len(patterns) for patterns in behavior_patterns.values())} annotated segments")
    print(f"   • Created dataset with {len(class_to_idx)} behavior classes")
    print(f"   • Ready to train model on these exact patterns")
    print(f"   • Model will learn to recognize these specific behaviors in new clips")

if __name__ == "__main__":
    main()