#!/usr/bin/env python3
"""
Binary Detection Ensemble Engine - Production MVP Version
"Something vs Nothing" detection with perfect annotation overlay support
"""

import tkinter as tk
import numpy as np
import pandas as pd
import os
import sys
import time
import json
from typing import Dict, List, Optional, Tuple
from utils import logger, get_config, CONFIG

# PRODUCTION PATHS
PROJECT_ROOT = '/home/jainy007/PEM/triage_brain'
V2_PATH = '/home/jainy007/PEM/triage_brain/src/triage_brain_v2'
MODEL_PREFIX = '/home/jainy007/PEM/triage_brain/triage_brain_model'
ANNOTATIONS_FILE = '/home/jainy007/PEM/triage_brain/assets/data/annotated_clips.jsonl'

# BINARY DETECTION SETTINGS
ANNOTATION_PADDING_FRAMES = 30  # 3 seconds at 10 FPS (increased from 1.5s)
DETECTION_THRESHOLD = 0.3  # Lower threshold for binary detection

# V2 FEATURE ORDER (unchanged)
V2_FEATURE_ORDER = [
    'duration_s', 'num_samples', 'sample_rate_hz', 'velocity_mean', 'velocity_std', 
    'velocity_min', 'velocity_max', 'velocity_range', 'acceleration_mean', 'acceleration_std', 
    'acceleration_min', 'acceleration_max', 'acceleration_range', 'jerk_mean', 'jerk_std', 
    'jerk_min', 'jerk_max', 'jerk_rms', 'max_deceleration', 'deceleration_events', 
    'velocity_zero_crossings', 'acceleration_reversals', 'motion_smoothness', 
    'jerk_per_second', 'accel_changes_per_second', 'distance_traveled'
]

# Setup V2 imports
sys.path.insert(0, V2_PATH)
try:
    from ensemble_triage_brain import EnsembleTriageBrain, label_segment, preprocess_data
    ENSEMBLE_AVAILABLE = True
    print("✅ V2 ensemble imported for binary detection")
except ImportError as e:
    print(f"❌ V2 ensemble import failed: {e}")
    ENSEMBLE_AVAILABLE = False

class AnnotationLoader:
    """Loads and manages annotation data for binary detection"""
    
    def __init__(self):
        self.annotations_by_scene_id = {}
        self.load_annotations()
    
    def load_annotations(self) -> None:
        """Load annotations from JSONL file"""
        try:
            if not os.path.exists(ANNOTATIONS_FILE):
                print(f"❌ Annotations file not found: {ANNOTATIONS_FILE}")
                return
            
            print(f"📝 Loading annotations for binary detection: {ANNOTATIONS_FILE}")
            
            with open(ANNOTATIONS_FILE, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        annotation = json.loads(line)
                        clip_name = annotation['clip']
                        
                        # Extract scene ID
                        if '__' in clip_name:
                            scene_id = clip_name.split('__')[0]
                        else:
                            scene_id = clip_name.replace('.mp4', '')
                        
                        # Process segments with binary detection focus
                        segments = []
                        for segment in annotation.get('segments', []):
                            segments.append({
                                'start': segment['start'],
                                'end': segment['end'],
                                'padded_start': max(0, segment['start'] - ANNOTATION_PADDING_FRAMES),
                                'padded_end': segment['end'] + ANNOTATION_PADDING_FRAMES,
                                'comment': segment.get('comment', ''),
                                'annotation_type': 'dangerous_event'  # Binary: all are just "something"
                            })
                        
                        self.annotations_by_scene_id[scene_id] = segments
                        
                        if line_num <= 3:
                            print(f"  📋 {scene_id}: {len(segments)} dangerous events")
                            
                    except json.JSONDecodeError:
                        print(f"❌ Invalid JSON on line {line_num}")
                        continue
                    except Exception as e:
                        print(f"❌ Error processing line {line_num}: {e}")
                        continue
            
            print(f"✅ Loaded {len(self.annotations_by_scene_id)} clips for binary detection")
            
        except Exception as e:
            print(f"❌ Failed to load annotations: {e}")
            self.annotations_by_scene_id = {}
    
    def get_segments_for_scene(self, scene_id: str) -> List[Dict]:
        """Get annotation segments for a scene ID"""
        if not scene_id:
            return []
        
        # Direct match
        if scene_id in self.annotations_by_scene_id:
            segments = self.annotations_by_scene_id[scene_id]
            print(f"✅ Found {len(segments)} dangerous events for: {scene_id}")
            return segments
        
        # Partial match
        for stored_id in self.annotations_by_scene_id.keys():
            if scene_id in stored_id or stored_id in scene_id:
                segments = self.annotations_by_scene_id[stored_id]
                print(f"✅ Partial match: {stored_id} -> {len(segments)} dangerous events")
                return segments
        
        print(f"❌ No dangerous events found for: {scene_id}")
        return []

class BinaryFrameAnalyzer:
    """Binary frame analysis: Something vs Nothing"""
    
    def __init__(self, ensemble_model):
        self.ensemble_model = ensemble_model
    
    def extract_features(self, motion_data: pd.DataFrame, frame_idx: int) -> Optional[Dict]:
        """Extract features for binary detection"""
        
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
        
        # Calculate all V2 features
        features = {
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
        
        # Return in exact V2 order
        return {feature: features[feature] for feature in V2_FEATURE_ORDER}
    
    def analyze_frame_binary(self, motion_data: pd.DataFrame, frame_idx: int) -> Optional[Dict]:
        """Binary frame analysis: detect something vs nothing"""
        
        features = self.extract_features(motion_data, frame_idx)
        if features is None:
            return None
        
        # Use V2 ensemble for binary detection
        if self.ensemble_model and ENSEMBLE_AVAILABLE:
            try:
                # Create DataFrame for V2
                features_df = pd.DataFrame([features])
                features_df['start_frame'] = frame_idx
                features_df['end_frame'] = frame_idx + 10
                features_df['comment'] = f"frame_{frame_idx}"
                features_df['clip_name'] = "binary_analysis"
                features_df['scene_id'] = "current"
                features_df['total_frames'] = len(motion_data)
                features_df['fps'] = 10.0
                
                # Get ensemble prediction
                X, y, processed = preprocess_data(features_df)
                predictions = self.ensemble_model.predict(X)
                prediction = predictions[0]
                
                # BINARY CLASSIFICATION with lower threshold
                if prediction > DETECTION_THRESHOLD:
                    classification = 'something_detected'
                    confidence = min(0.95, max(0.4, prediction))
                    risk_level = 'DETECTED'
                else:
                    classification = 'normal'
                    confidence = max(0.3, 1.0 - prediction)
                    risk_level = 'NORMAL'
                
                return {
                    'frame_idx': frame_idx,
                    'classification': classification,
                    'confidence': confidence,
                    'risk_level': risk_level,
                    'raw_prediction': float(prediction),
                    'features': features,
                    'strategy': 'binary_v2_ensemble'
                }
                
            except Exception as e:
                logger.warning(f"V2 binary prediction failed for frame {frame_idx}: {e}")
        
        # Fallback binary detection
        velocity_var = features['velocity_std']
        jerk_rms = features['jerk_rms']
        acceleration_var = features['acceleration_std']
        
        # Simple motion-based detection
        if velocity_var > 2.0 or jerk_rms > 20 or acceleration_var > 5.0:
            return {
                'frame_idx': frame_idx,
                'classification': 'something_detected',
                'confidence': 0.6,
                'risk_level': 'DETECTED',
                'raw_prediction': 0.6,
                'features': features,
                'strategy': 'fallback_motion_detection'
            }
        
        return {
            'frame_idx': frame_idx,
            'classification': 'normal',
            'confidence': 0.4,
            'risk_level': 'NORMAL',
            'raw_prediction': 0.3,
            'features': features,
            'strategy': 'fallback_normal'
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

class ClusterAnalyzer:
    """Binary cluster analysis for dangerous event detection"""
    
    def __init__(self):
        self.config = get_config('clustering')
        self.annotation_loader = AnnotationLoader()
    
    def analyze_full_scene(self, motion_data: pd.DataFrame, ensemble_model, progress_callback=None, scene_id: str = None) -> Dict:
        """Binary scene analysis: find dangerous events"""
        
        logger.analysis(f"Starting BINARY detection analysis for: {scene_id}")
        start_time = time.time()
        
        # Get dangerous event annotations
        annotated_segments = self.annotation_loader.get_segments_for_scene(scene_id) if scene_id else []
        
        # Determine frame indices for analysis
        if annotated_segments:
            # Focus heavily on annotated dangerous events + context
            frame_indices = self._get_annotated_frame_indices(annotated_segments, len(motion_data))
            print(f"📝 DANGEROUS EVENT FOCUS: {len(frame_indices)} frames around {len(annotated_segments)} events")
        else:
            # No annotations - broader sampling
            sampling_rate = 5
            frame_indices = list(range(0, len(motion_data), sampling_rate))
            print(f"🔍 NO ANNOTATIONS: Sampling {len(frame_indices)} frames")
        
        # Binary frame analysis
        frame_analyzer = BinaryFrameAnalyzer(ensemble_model)
        frame_results = []
        
        for i, frame_idx in enumerate(frame_indices):
            if progress_callback:
                try:
                    progress_callback(i, len(frame_indices), f"Binary detection: frame {frame_idx}")
                except:
                    pass
            
            result = frame_analyzer.analyze_frame_binary(motion_data, frame_idx)
            if result:
                # Apply annotation bias for dangerous events
                if self._is_in_annotated_region(frame_idx, annotated_segments):
                    result['annotation_bias'] = True
                    result['in_dangerous_event'] = True
                    # Find the specific annotation for overlay
                    result['annotation_info'] = self._get_annotation_for_frame(frame_idx, annotated_segments)
                    # Boost confidence for known dangerous events
                    result['confidence'] = min(0.95, result['confidence'] * 1.2)
                else:
                    result['annotation_bias'] = False
                    result['in_dangerous_event'] = False
                    result['annotation_info'] = None
                
                frame_results.append(result)
        
        # Create binary clusters (dangerous event regions)
        clusters = self._create_binary_clusters(frame_results, annotated_segments)
        top_clusters = self._select_top_binary_clusters(clusters)
        
        elapsed = time.time() - start_time
        bias_type = "DANGEROUS EVENT FOCUS" if annotated_segments else "GENERAL SAMPLING"
        logger.analysis(f"Binary {bias_type} analysis complete: {len(top_clusters)} dangerous regions in {elapsed:.1f}s")
        
        return {
            'frame_predictions': frame_results,
            'all_clusters': clusters,
            'top_clusters': top_clusters,
            'annotation_bias': len(annotated_segments) > 0,
            'dangerous_events_found': len(top_clusters),
            'annotated_segments': annotated_segments
        }
    
    def _get_annotated_frame_indices(self, segments: List[Dict], total_frames: int) -> List[int]:
        """Get frame indices for dangerous event regions with padding"""
        indices = []
        for segment in segments:
            start = segment['padded_start']
            end = min(segment['padded_end'], total_frames - 1)
            indices.extend(range(start, end + 1))
        
        return sorted(list(set(indices)))
    
    def _is_in_annotated_region(self, frame_idx: int, segments: List[Dict]) -> bool:
        """Check if frame is in a dangerous event region"""
        for segment in segments:
            if segment['padded_start'] <= frame_idx <= segment['padded_end']:
                return True
        return False
    
    def _get_annotation_for_frame(self, frame_idx: int, segments: List[Dict]) -> Optional[Dict]:
        """Get annotation info for frame (for overlay display)"""
        for segment in segments:
            if segment['start'] <= frame_idx <= segment['end']:  # Original range, not padded
                return {
                    'comment': segment['comment'],
                    'start_frame': segment['start'],
                    'end_frame': segment['end'],
                    'frame_in_event': frame_idx - segment['start']
                }
        return None
    
    def _create_binary_clusters(self, frame_results: List[Dict], annotated_segments: List[Dict]) -> List[Dict]:
        """Create clusters for dangerous event regions"""
        
        # Only keep detected events
        detected_events = [r for r in frame_results if r['classification'] == 'something_detected']
        
        if not detected_events:
            return []
        
        # Sort by frame index
        sorted_events = sorted(detected_events, key=lambda x: x['frame_idx'])
        
        clusters = []
        current_cluster = None
        gap_threshold = 20  # 2 seconds gap
        
        for event in sorted_events:
            frame_idx = event['frame_idx']
            
            if (current_cluster is None or 
                frame_idx - current_cluster['end_frame'] > gap_threshold):
                
                if current_cluster:
                    clusters.append(self._finalize_binary_cluster(current_cluster, annotated_segments))
                
                current_cluster = {
                    'start_frame': frame_idx,
                    'end_frame': frame_idx,
                    'frames': [frame_idx],
                    'predictions': [event],
                    'dangerous_event_type': 'detected'
                }
            else:
                current_cluster['end_frame'] = frame_idx
                current_cluster['frames'].append(frame_idx)
                current_cluster['predictions'].append(event)
        
        if current_cluster:
            clusters.append(self._finalize_binary_cluster(current_cluster, annotated_segments))
        
        return clusters
    
    def _finalize_binary_cluster(self, cluster: Dict, annotated_segments: List[Dict]) -> Dict:
        """Finalize binary cluster for dangerous event"""
        
        predictions = cluster['predictions']
        
        avg_confidence = np.mean([p['confidence'] for p in predictions])
        max_confidence = np.max([p['confidence'] for p in predictions])
        duration_frames = cluster['end_frame'] - cluster['start_frame'] + 1
        duration_seconds = duration_frames * 0.1
        
        # Check if this cluster overlaps with known dangerous events
        overlaps_annotation = False
        annotation_info = None
        
        for segment in annotated_segments:
            if (cluster['start_frame'] <= segment['end'] and 
                cluster['end_frame'] >= segment['start']):
                overlaps_annotation = True
                annotation_info = segment
                break
        
        # Calculate final score with annotation boost
        base_score = avg_confidence
        annotation_bonus = 2.0 if overlaps_annotation else 1.0
        final_score = base_score * annotation_bonus
        
        cluster.update({
            'classification': 'dangerous_event_detected',
            'avg_confidence': avg_confidence,
            'max_confidence': max_confidence,
            'duration_frames': duration_frames,
            'duration_seconds': duration_seconds,
            'final_score': final_score,
            'overlaps_annotation': overlaps_annotation,
            'annotation_info': annotation_info,
            'annotation_bonus': annotation_bonus,
            'export_start_time': cluster['start_frame'] * 0.1,
            'export_end_time': cluster['end_frame'] * 0.1
        })
        
        return cluster
    
    def _select_top_binary_clusters(self, clusters: List[Dict]) -> List[Dict]:
        """Select top dangerous event clusters"""
        if not clusters:
            return []
        
        # Filter by minimum duration (1 second)
        min_duration = 10
        valid_clusters = [c for c in clusters if c['duration_frames'] >= min_duration]
        
        # Sort by final score and return top 3
        valid_clusters.sort(key=lambda x: x['final_score'], reverse=True)
        
        return valid_clusters[:3]

class EnsemblePanel:
    """GUI panel for binary detection results"""
    
    def __init__(self, parent):
        self.parent = parent
        self.ensemble_model = None
        self.setup_ui()
        self.load_model()
    
    def setup_ui(self):
        """Setup UI for binary detection"""
        self.frame = tk.LabelFrame(self.parent, text="Dangerous Event Detection", font=('Arial', 14, 'bold'))
        self.frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Results display
        results_frame = tk.Frame(self.frame)
        results_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(results_frame, text="Status:", font=('Arial', 12, 'bold')).grid(row=0, column=0, sticky='w')
        self.status_label = tk.Label(results_frame, text="No analysis", font=('Arial', 12), fg='blue')
        self.status_label.grid(row=0, column=1, sticky='w', padx=10)
        
        tk.Label(results_frame, text="Confidence:", font=('Arial', 12, 'bold')).grid(row=1, column=0, sticky='w')
        self.confidence_label = tk.Label(results_frame, text="0.000", font=('Arial', 12))
        self.confidence_label.grid(row=1, column=1, sticky='w', padx=10)
        
        tk.Label(results_frame, text="Events Found:", font=('Arial', 12, 'bold')).grid(row=2, column=0, sticky='w')
        self.events_label = tk.Label(results_frame, text="0", font=('Arial', 12, 'bold'), fg='gray')
        self.events_label.grid(row=2, column=1, sticky='w', padx=10)
        
        # Status display
        status_frame = tk.LabelFrame(self.frame, text="Detection Status", font=('Arial', 12, 'bold'))
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.status_text = tk.Text(status_frame, height=4, width=50, font=('Courier', 10))
        self.status_text.pack(padx=5, pady=5)
    
    def load_model(self):
        """Load V2 ensemble for binary detection"""
        try:
            if not ENSEMBLE_AVAILABLE:
                self.status_text.insert(tk.END, "⚠️ V2 ensemble not available\n")
                return
            
            # Load ensemble
            self.ensemble_model = EnsembleTriageBrain()
            self.ensemble_model.load(MODEL_PREFIX)
            
            self.status_text.insert(tk.END, "✅ Binary Detection Ready!\n")
            self.status_text.insert(tk.END, f"📂 V2 Ensemble Loaded\n")
            self.status_text.insert(tk.END, "🎯 Mode: Something vs Nothing\n")
            
        except Exception as e:
            self.status_text.insert(tk.END, f"❌ Error loading models: {e}\n")
            self.ensemble_model = None
    
    def update_display(self, top_clusters: List[Dict]):
        """Update display with binary detection results"""
        
        if not top_clusters:
            self.status_label.config(text="No dangerous events detected")
            self.confidence_label.config(text="N/A")
            self.events_label.config(text="0", fg='green')
            
            self.status_text.delete(1.0, tk.END)
            self.status_text.insert(tk.END, "Binary Detection Complete - No dangerous events found.\n")
            return
        
        # Show results
        top_cluster = top_clusters[0]
        
        self.status_label.config(text="Dangerous events detected")
        self.confidence_label.config(text=f"{top_cluster['avg_confidence']:.3f}")
        self.events_label.config(text=str(len(top_clusters)), fg='red')
        
        # Show summary
        self.status_text.delete(1.0, tk.END)
        self.status_text.insert(tk.END, f"🚨 Found {len(top_clusters)} Dangerous Event Regions:\n\n")
        
        for i, cluster in enumerate(top_clusters):
            annotation_mark = "📝" if cluster.get('overlaps_annotation', False) else "🔍"
            start_time = cluster.get('export_start_time', 0)
            end_time = cluster.get('export_end_time', 0)
            
            self.status_text.insert(tk.END, 
                f"{annotation_mark} {i+1}. Event at {start_time:.1f}s-{end_time:.1f}s "
                f"(conf: {cluster['avg_confidence']:.3f})\n")
        
        logger.analysis(f"Updated binary detection display with {len(top_clusters)} dangerous events")


# Test function for binary detection
if __name__ == "__main__":
    print("🧪 Testing Binary Detection Ensemble Engine...")
    
    # Test annotation loading
    loader = AnnotationLoader()
    print(f"Loaded {len(loader.annotations_by_scene_id)} clips for binary detection")
    
    if loader.annotations_by_scene_id:
        test_scene_id = list(loader.annotations_by_scene_id.keys())[0]
        segments = loader.get_segments_for_scene(test_scene_id)
        print(f"Test scene {test_scene_id}: {len(segments)} dangerous events")
    
    # Test V2 model loading
    if ENSEMBLE_AVAILABLE:
        try:
            ensemble = EnsembleTriageBrain()
            ensemble.load(MODEL_PREFIX)
            print("✅ V2 model loaded for binary detection")
            
            # Test with sample data
            motion_data = pd.DataFrame({
                'time': np.arange(100) * 0.1,
                'velocity': 5.0 + np.random.normal(0, 1, 100),
                'acceleration': np.random.normal(0, 2, 100),
                'jerk': np.random.normal(0, 10, 100)
            })
            
            # Test binary frame analysis
            analyzer = BinaryFrameAnalyzer(ensemble)
            result = analyzer.analyze_frame_binary(motion_data, 50)
            if result:
                print(f"✅ Binary analysis: {result['classification']} (conf: {result['confidence']:.3f})")
            
            print("🎉 Binary detection engine ready for production!")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
    
    else:
        print("❌ V2 ensemble not available for testing")