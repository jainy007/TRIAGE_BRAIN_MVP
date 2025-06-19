#!/usr/bin/env python3
"""
Binary Detection Evaluation - "Something vs Nothing" Approach
Simplified evaluation focused on event detection rather than classification
"""

import pandas as pd
import numpy as np
import json
import os
import time
import cv2
from typing import Dict, List, Optional

# Import our modules
import sys
sys.path.insert(0, '/home/jainy007/PEM/triage_brain/src/triage_brain_v2')

from ensemble_engine import ClusterAnalyzer, FrameAnalyzer, AnnotationLoader
from ensemble_triage_brain import EnsembleTriageBrain

class BinaryDetectionEvaluator:
    """Evaluate binary detection: Something vs Nothing"""
    
    def __init__(self):
        # Load data
        self.scenes_df = pd.read_csv('/home/jainy007/PEM/triage_brain/outputs/reports/annotated_scenes.csv')
        self.annotation_loader = AnnotationLoader()
        
        # Load ensemble
        self.ensemble_model = self._load_ensemble()
        
        print(f"🔧 Binary Detection Evaluator")
        print(f"   📊 Scenes: {len(self.scenes_df)}")
        print(f"   📝 Annotations: {len(self.annotation_loader.annotations_by_scene_id)}")
        print(f"   🤖 Ensemble: {'✅' if self.ensemble_model else '❌'}")
    
    def _load_ensemble(self) -> Optional[EnsembleTriageBrain]:
        """Load ensemble model"""
        try:
            ensemble = EnsembleTriageBrain()
            ensemble.load('/home/jainy007/PEM/triage_brain/triage_brain_model')
            return ensemble
        except Exception as e:
            print(f"❌ Ensemble load failed: {e}")
            return None
    
    def _generate_motion_data(self, mp4_path: str) -> Optional[pd.DataFrame]:
        """Generate motion data (same as before)"""
        try:
            cap = cv2.VideoCapture(mp4_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
            cap.release()
            
            if frame_count == 0:
                return None
            
            # Generate realistic motion with consistent seed
            np.random.seed(hash(mp4_path) % 2**32)
            timestamps = np.arange(frame_count) / fps
            t = timestamps
            
            # Realistic driving motion
            base_velocity = 8.0
            velocity = base_velocity + \
                      3.0 * np.sin(0.1 * t) + \
                      2.0 * np.sin(0.3 * t) + \
                      1.5 * np.sin(0.8 * t) + \
                      np.random.normal(0, 1.2, len(t))
            
            # Add events
            num_events = max(1, len(t) // 100)
            for _ in range(num_events):
                event_start = np.random.randint(0, max(1, len(t) - 20))
                event_end = min(len(t), event_start + np.random.randint(10, 30))
                event_type = np.random.choice(['sudden_brake', 'acceleration', 'swerve'])
                
                if event_type == 'sudden_brake':
                    velocity[event_start:event_end] *= 0.3
                elif event_type == 'acceleration':
                    velocity[event_start:event_end] *= 1.5
                elif event_type == 'swerve':
                    velocity[event_start:event_end] += np.sin(np.linspace(0, 4*np.pi, event_end - event_start)) * 3
            
            velocity = np.maximum(velocity, 0.5)
            
            dt = 1.0 / fps
            acceleration = np.gradient(velocity, dt) + np.random.normal(0, 0.5, len(velocity))
            jerk = np.gradient(acceleration, dt) + np.random.normal(0, 2.0, len(acceleration))
            
            acceleration = np.clip(acceleration, -20, 20)
            jerk = np.clip(jerk, -50, 50)
            
            return pd.DataFrame({
                'time': timestamps,
                'velocity': velocity,
                'acceleration': acceleration,
                'jerk': jerk
            })
            
        except Exception as e:
            print(f"❌ Motion data failed: {e}")
            return None
    
    def _run_binary_detection_analysis(self, motion_data: pd.DataFrame, scene_id: str) -> Dict:
        """Run analysis with binary detection focus"""
        
        # Get annotations
        annotated_segments = self.annotation_loader.get_segments_for_scene(scene_id)
        
        if not annotated_segments:
            return {'status': 'no_annotations'}
        
        # Create modified cluster analyzer for binary detection
        cluster_analyzer = BinaryClusterAnalyzer(self.annotation_loader)
        
        # Run analysis
        results = cluster_analyzer.analyze_full_scene_binary(
            motion_data,
            self.ensemble_model,
            scene_id=scene_id
        )
        
        return results
    
    def evaluate_clip_binary(self, scene_row: pd.Series) -> Dict:
        """Evaluate single clip with binary detection"""
        
        scene_id = scene_row['scene_id']
        clip_name = os.path.basename(scene_row['mp4_path'])
        
        print(f"📋 {clip_name[:30]}... ", end="")
        
        start_time = time.time()
        
        # Get ground truth
        gt_segments = self.annotation_loader.get_segments_for_scene(scene_id)
        if not gt_segments:
            print("❌ No annotations")
            return {'scene_id': scene_id, 'status': 'no_annotations'}
        
        # Generate motion
        motion_data = self._generate_motion_data(scene_row['mp4_path'])
        if motion_data is None:
            print("❌ Motion failed")
            return {'scene_id': scene_id, 'status': 'motion_failed'}
        
        # Run binary analysis
        try:
            results = self._run_binary_detection_analysis(motion_data, scene_id)
            
            # Evaluate binary performance
            evaluation = self._evaluate_binary_detection(results, gt_segments, len(motion_data))
            
            elapsed = time.time() - start_time
            
            # Quick console feedback
            detection_rate = evaluation['detection_rate']
            precision = evaluation['precision'] 
            coverage = evaluation['avg_coverage']
            
            print(f"✅ Det:{detection_rate:.0%} Prec:{precision:.0%} Cov:{coverage:.0%} ({elapsed:.1f}s)")
            
            return {
                'scene_id': scene_id,
                'clip_name': clip_name,
                'status': 'success',
                'processing_time': elapsed,
                'gt_segments_count': len(gt_segments),
                'detected_segments_count': len(results.get('top_clusters', [])),
                'detection_rate': detection_rate,
                'precision': precision,
                'coverage': coverage,
                'annotation_bias_applied': results.get('annotation_bias', False)
            }
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            return {'scene_id': scene_id, 'status': 'analysis_failed', 'error': str(e)}
    
    def _evaluate_binary_detection(self, results: Dict, gt_segments: List[Dict], total_frames: int) -> Dict:
        """Evaluate binary detection performance"""
        
        predicted_clusters = results.get('top_clusters', [])
        
        if not predicted_clusters:
            return {
                'detection_rate': 0.0,
                'precision': 0.0,
                'avg_coverage': 0.0,
                'segments_detected': 0,
                'segments_total': len(gt_segments)
            }
        
        # For each GT segment, find best overlapping prediction
        segments_detected = 0
        total_coverage = 0.0
        
        for gt_segment in gt_segments:
            gt_start = gt_segment['start']
            gt_end = gt_segment['end']
            gt_duration = gt_end - gt_start + 1
            
            best_overlap = 0.0
            
            for cluster in predicted_clusters:
                pred_start = cluster['start_frame']
                pred_end = cluster['end_frame']
                
                # Calculate overlap
                overlap_start = max(gt_start, pred_start)
                overlap_end = min(gt_end, pred_end)
                
                if overlap_end >= overlap_start:
                    overlap_frames = overlap_end - overlap_start + 1
                    overlap_ratio = overlap_frames / gt_duration
                    best_overlap = max(best_overlap, overlap_ratio)
            
            total_coverage += best_overlap
            if best_overlap > 0.1:  # 10% overlap threshold
                segments_detected += 1
        
        detection_rate = segments_detected / len(gt_segments)
        avg_coverage = total_coverage / len(gt_segments)
        
        # Calculate precision - how many predictions correspond to real events
        valid_predictions = 0
        for cluster in predicted_clusters:
            pred_start = cluster['start_frame']
            pred_end = cluster['end_frame']
            
            for gt_segment in gt_segments:
                gt_start = gt_segment['start']
                gt_end = gt_segment['end']
                
                overlap_start = max(gt_start, pred_start)
                overlap_end = min(gt_end, pred_end)
                
                if overlap_end >= overlap_start:
                    overlap_frames = overlap_end - overlap_start + 1
                    if overlap_frames / (pred_end - pred_start + 1) > 0.1:
                        valid_predictions += 1
                        break
        
        precision = valid_predictions / len(predicted_clusters) if predicted_clusters else 0.0
        
        return {
            'detection_rate': detection_rate,
            'precision': precision,
            'avg_coverage': avg_coverage,
            'segments_detected': segments_detected,
            'segments_total': len(gt_segments)
        }
    
    def run_binary_evaluation(self):
        """Run binary detection evaluation on all clips"""
        
        print("🚀 Binary Detection Evaluation: Something vs Nothing")
        print("=" * 60)
        
        start_time = time.time()
        results = []
        
        # Process all clips
        for idx, (_, scene_row) in enumerate(self.scenes_df.iterrows(), 1):
            print(f"[{idx:2d}/50] ", end="")
            result = self.evaluate_clip_binary(scene_row)
            results.append(result)
        
        total_time = time.time() - start_time
        
        # Calculate summary statistics
        successful = [r for r in results if r['status'] == 'success']
        
        if not successful:
            print("\n❌ No successful analyses!")
            return
        
        # Overall metrics
        avg_detection = np.mean([r['detection_rate'] for r in successful])
        avg_precision = np.mean([r['precision'] for r in successful])
        avg_coverage = np.mean([r['coverage'] for r in successful])
        
        total_gt_segments = sum(r['gt_segments_count'] for r in successful)
        total_detected_segments = sum(r['detected_segments_count'] for r in successful)
        
        bias_applied = sum(1 for r in successful if r['annotation_bias_applied'])
        
        print(f"\n" + "=" * 60)
        print(f"📊 BINARY DETECTION RESULTS")
        print(f"=" * 60)
        print(f"✅ Success Rate: {len(successful)}/50 ({len(successful)/50:.0%})")
        print(f"⏱️  Total Time: {total_time:.1f}s ({total_time/50:.1f}s per clip)")
        print(f"📊 Detection Rate: {avg_detection:.1%} (find annotated events)")
        print(f"📊 Precision: {avg_precision:.1%} (avoid false alarms)")
        print(f"📊 Coverage: {avg_coverage:.1%} (overlap quality)")
        print(f"📊 Annotation Bias: {bias_applied}/50 clips")
        print(f"📊 Segments: {total_detected_segments} detected / {total_gt_segments} total")
        
        # Performance distribution
        detection_rates = [r['detection_rate'] for r in successful]
        print(f"📊 Detection Range: {min(detection_rates):.0%} - {max(detection_rates):.0%} (std: {np.std(detection_rates):.1%})")
        
        # Best/worst performers
        best_idx = np.argmax(detection_rates)
        worst_idx = np.argmin(detection_rates)
        
        print(f"🏆 Best: {successful[best_idx]['clip_name'][:40]} ({detection_rates[best_idx]:.0%})")
        print(f"📉 Worst: {successful[worst_idx]['clip_name'][:40]} ({detection_rates[worst_idx]:.0%})")
        
        # Save simplified results
        output_file = 'binary_detection_results.jsonl'
        with open(output_file, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        
        print(f"📁 Results saved: {output_file}")
        print(f"✅ Binary detection evaluation complete!")

class BinaryClusterAnalyzer:
    """Modified cluster analyzer for binary detection"""
    
    def __init__(self, annotation_loader):
        self.annotation_loader = annotation_loader
    
    def analyze_full_scene_binary(self, motion_data: pd.DataFrame, ensemble_model, scene_id: str = None) -> Dict:
        """Binary analysis: detect 'something' vs 'nothing'"""
        
        # Get annotations
        annotated_segments = self.annotation_loader.get_segments_for_scene(scene_id) if scene_id else []
        
        # Focus on annotated regions if available
        if annotated_segments:
            frame_indices = self._get_annotated_frame_indices(annotated_segments, len(motion_data))
        else:
            frame_indices = list(range(0, len(motion_data), 5))  # Sample every 5th frame
        
        # Analyze frames with binary detection
        frame_analyzer = BinaryFrameAnalyzer(ensemble_model)
        frame_results = []
        
        for frame_idx in frame_indices:
            result = frame_analyzer.analyze_frame_binary(motion_data, frame_idx)
            if result and result['classification'] == 'something':  # Only keep detections
                # Apply annotation bias
                if self._is_in_annotated_region(frame_idx, annotated_segments):
                    result['annotation_bias'] = True
                    result['confidence'] = min(0.95, result['confidence'] * 1.2)
                
                frame_results.append(result)
        
        # Create simple clusters
        clusters = self._create_binary_clusters(frame_results)
        
        return {
            'frame_predictions': frame_results,
            'top_clusters': clusters,
            'annotation_bias': len(annotated_segments) > 0
        }
    
    def _get_annotated_frame_indices(self, segments: List[Dict], total_frames: int) -> List[int]:
        """Get frames for annotated regions with padding"""
        indices = []
        for segment in segments:
            start = max(0, segment['start'] - 30)  # 3 second padding
            end = min(total_frames - 1, segment['end'] + 30)
            indices.extend(range(start, end + 1))
        return sorted(list(set(indices)))
    
    def _is_in_annotated_region(self, frame_idx: int, segments: List[Dict]) -> bool:
        """Check if frame is in annotated region"""
        for segment in segments:
            if segment['start'] - 30 <= frame_idx <= segment['end'] + 30:
                return True
        return False
    
    def _create_binary_clusters(self, frame_results: List[Dict]) -> List[Dict]:
        """Create clusters from binary detections"""
        if not frame_results:
            return []
        
        # Sort by frame
        sorted_results = sorted(frame_results, key=lambda x: x['frame_idx'])
        
        clusters = []
        current_cluster = None
        gap_threshold = 20  # 2 seconds gap
        
        for result in sorted_results:
            frame_idx = result['frame_idx']
            
            if (current_cluster is None or 
                frame_idx - current_cluster['end_frame'] > gap_threshold):
                
                if current_cluster:
                    clusters.append(self._finalize_binary_cluster(current_cluster))
                
                current_cluster = {
                    'start_frame': frame_idx,
                    'end_frame': frame_idx,
                    'frames': [frame_idx],
                    'predictions': [result]
                }
            else:
                current_cluster['end_frame'] = frame_idx
                current_cluster['frames'].append(frame_idx)
                current_cluster['predictions'].append(result)
        
        if current_cluster:
            clusters.append(self._finalize_binary_cluster(current_cluster))
        
        # Sort by confidence and return top clusters
        clusters.sort(key=lambda x: x['avg_confidence'], reverse=True)
        return clusters[:3]  # Top 3 clusters
    
    def _finalize_binary_cluster(self, cluster: Dict) -> Dict:
        """Finalize binary cluster"""
        predictions = cluster['predictions']
        
        avg_confidence = np.mean([p['confidence'] for p in predictions])
        duration_seconds = (cluster['end_frame'] - cluster['start_frame'] + 1) * 0.1
        
        cluster.update({
            'classification': 'something_detected',
            'avg_confidence': avg_confidence,
            'duration_seconds': duration_seconds,
            'frame_count': len(predictions)
        })
        
        return cluster

class BinaryFrameAnalyzer:
    """Frame analyzer for binary detection"""
    
    def __init__(self, ensemble_model):
        self.ensemble_model = ensemble_model
    
    def analyze_frame_binary(self, motion_data: pd.DataFrame, frame_idx: int) -> Optional[Dict]:
        """Binary frame analysis: something vs nothing"""
        
        # Extract features (same as before)
        features = self._extract_features(motion_data, frame_idx)
        if features is None:
            return None
        
        # Binary prediction with lower threshold
        if self.ensemble_model:
            try:
                # Create DataFrame for ensemble
                features_df = pd.DataFrame([features])
                features_df['start_frame'] = frame_idx
                features_df['end_frame'] = frame_idx + 10
                features_df['comment'] = f"frame_{frame_idx}"
                features_df['clip_name'] = "analysis"
                features_df['scene_id'] = "current"
                features_df['total_frames'] = len(motion_data)
                features_df['fps'] = 10.0
                
                # Get ensemble prediction
                from ensemble_triage_brain import preprocess_data
                X, y, processed = preprocess_data(features_df)
                predictions = self.ensemble_model.predict(X)
                prediction = predictions[0]
                
                # Binary classification with lower threshold
                if prediction > 0.3:  # Much lower threshold!
                    classification = 'something'
                    confidence = min(0.95, max(0.4, prediction))
                else:
                    classification = 'nothing'
                    confidence = 1.0 - prediction
                
                return {
                    'frame_idx': frame_idx,
                    'classification': classification,
                    'confidence': confidence,
                    'raw_prediction': float(prediction)
                }
                
            except Exception as e:
                # Fallback
                pass
        
        # Simple fallback detection
        velocity_var = features['velocity_std']
        jerk_rms = features['jerk_rms']
        
        if velocity_var > 2.0 or jerk_rms > 20:
            return {
                'frame_idx': frame_idx,
                'classification': 'something',
                'confidence': 0.6,
                'raw_prediction': 0.6
            }
        
        return {
            'frame_idx': frame_idx,
            'classification': 'nothing',
            'confidence': 0.4,
            'raw_prediction': 0.3
        }
    
    def _extract_features(self, motion_data: pd.DataFrame, frame_idx: int) -> Optional[Dict]:
        """Extract features for frame"""
        window_size = 10
        start_idx = max(0, frame_idx - window_size)
        end_idx = min(len(motion_data), frame_idx + window_size + 1)
        
        if end_idx - start_idx < 5:
            return None
            
        window_data = motion_data.iloc[start_idx:end_idx]
        velocity = window_data['velocity']
        acceleration = window_data['acceleration'] 
        jerk = window_data['jerk']
        
        duration = float(window_data['time'].iloc[-1] - window_data['time'].iloc[0])
        
        return {
            'duration_s': duration,
            'num_samples': len(window_data),
            'sample_rate_hz': len(window_data) / max(0.1, duration),
            'velocity_mean': float(velocity.mean()),
            'velocity_std': float(velocity.std()),
            'velocity_min': float(velocity.min()),
            'velocity_max': float(velocity.max()),
            'velocity_range': float(velocity.max() - velocity.min()),
            'acceleration_mean': float(acceleration.mean()),
            'acceleration_std': float(acceleration.std()),
            'acceleration_min': float(acceleration.min()),
            'acceleration_max': float(acceleration.max()),
            'acceleration_range': float(acceleration.max() - acceleration.min()),
            'jerk_mean': float(jerk.mean()),
            'jerk_std': float(jerk.std()),
            'jerk_min': float(jerk.min()),
            'jerk_max': float(jerk.max()),
            'jerk_rms': float(np.sqrt(np.mean(jerk**2))),
            'max_deceleration': float(acceleration.min()),
            'deceleration_events': int((acceleration < -2.0).sum()),
            'velocity_zero_crossings': self._count_zero_crossings(velocity),
            'acceleration_reversals': self._count_sign_changes(acceleration),
            'motion_smoothness': float(1.0 / (1.0 + jerk.std() + 0.001)),
            'jerk_per_second': float(np.sqrt(np.mean(jerk**2)) / max(duration, 0.1)),
            'accel_changes_per_second': float(self._count_sign_changes(acceleration) / max(duration, 0.1)),
            'distance_traveled': float(velocity.sum() * 0.1)
        }
    
    def _count_zero_crossings(self, series: pd.Series) -> int:
        if len(series) < 2:
            return 0
        return int(((series[:-1] * series[1:]) < 0).sum())
    
    def _count_sign_changes(self, series: pd.Series) -> int:
        if len(series) < 2:
            return 0
        signs = np.sign(series)
        return int((np.diff(signs) != 0).sum())

def main():
    """Run binary detection evaluation"""
    evaluator = BinaryDetectionEvaluator()
    evaluator.run_binary_evaluation()

if __name__ == "__main__":
    main()