#!/usr/bin/env python3
"""
Clean, Working Ensemble Engine for Triage Brain GUI
V2 Ensemble with Annotation-Biased Analysis - COMPLETELY REWRITTEN
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

# VERIFIED WORKING PATHS
PROJECT_ROOT = '/home/jainy007/PEM/triage_brain'
V2_PATH = '/home/jainy007/PEM/triage_brain/src/triage_brain_v2'
MODEL_PREFIX = '/home/jainy007/PEM/triage_brain/triage_brain_model'
ANNOTATIONS_FILE = '/home/jainy007/PEM/triage_brain/assets/data/annotated_clips.jsonl'

# ANNOTATION SETTINGS
ANNOTATION_PADDING_FRAMES = 15  # 1.5 seconds at 10 FPS

# EXACT V2 FEATURE ORDER (from working checkpoint)
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
    print("✅ V2 ensemble imported successfully")
except ImportError as e:
    print(f"❌ V2 ensemble import failed: {e}")
    ENSEMBLE_AVAILABLE = False

class AnnotationLoader:
    """Handles loading and parsing annotation data"""
    
    def __init__(self):
        self.annotations_by_scene_id = {}
        self.load_annotations()
    
    def load_annotations(self) -> None:
        """Load annotations from JSONL file"""
        try:
            if not os.path.exists(ANNOTATIONS_FILE):
                print(f"❌ Annotations file not found: {ANNOTATIONS_FILE}")
                return
            
            print(f"📝 Loading annotations from: {ANNOTATIONS_FILE}")
            
            with open(ANNOTATIONS_FILE, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        # Parse JSON
                        annotation = json.loads(line)
                        clip_name = annotation['clip']
                        
                        # Extract scene ID from clip name
                        # Format: "01bb304d7bd835f8bbef7086b688e35e__Summer_2019.mp4"
                        if '__' in clip_name:
                            scene_id = clip_name.split('__')[0]
                        else:
                            scene_id = clip_name.replace('.mp4', '')
                        
                        # Process segments
                        segments = []
                        for segment in annotation.get('segments', []):
                            segments.append({
                                'start': segment['start'],
                                'end': segment['end'],
                                'padded_start': max(0, segment['start'] - ANNOTATION_PADDING_FRAMES),
                                'padded_end': segment['end'] + ANNOTATION_PADDING_FRAMES,
                                'comment': segment.get('comment', '')
                            })
                        
                        # Store by scene ID
                        self.annotations_by_scene_id[scene_id] = segments
                        
                        if line_num <= 3:
                            print(f"  📋 {scene_id}: {len(segments)} segments")
                            
                    except json.JSONDecodeError:
                        print(f"❌ Invalid JSON on line {line_num}")
                        continue
                    except Exception as e:
                        print(f"❌ Error processing line {line_num}: {e}")
                        continue
            
            print(f"✅ Loaded {len(self.annotations_by_scene_id)} annotated clips")
            
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
            print(f"✅ Found {len(segments)} annotations for scene: {scene_id}")
            return segments
        
        # Try partial match
        for stored_id in self.annotations_by_scene_id.keys():
            if scene_id in stored_id or stored_id in scene_id:
                segments = self.annotations_by_scene_id[stored_id]
                print(f"✅ Partial match: {stored_id} -> {len(segments)} annotations")
                return segments
        
        print(f"❌ No annotations found for scene: {scene_id}")
        return []

class FrameAnalyzer:
    """Analyzes individual frames using V2 ensemble"""
    
    def __init__(self, ensemble_model):
        self.ensemble_model = ensemble_model
        self.risk_multipliers = get_config('risk_multipliers')
    
    def extract_features(self, motion_data: pd.DataFrame, frame_idx: int) -> Optional[Dict]:
        """Extract features for a frame using V2 feature order"""
        
        # Create analysis window
        window_size = 10
        start_idx = max(0, frame_idx - window_size)
        end_idx = min(len(motion_data), frame_idx + window_size + 1)
        
        if end_idx - start_idx < 5:
            return None
        
        window_data = motion_data.iloc[start_idx:end_idx]
        velocity = window_data['velocity']
        acceleration = window_data['acceleration'] 
        jerk = window_data['jerk']
        
        # Calculate all V2 features
        duration = float(window_data['time'].iloc[-1] - window_data['time'].iloc[0])
        
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
    
    def analyze_frame(self, motion_data: pd.DataFrame, frame_idx: int) -> Optional[Dict]:
        """Analyze a single frame"""
        
        features = self.extract_features(motion_data, frame_idx)
        if features is None:
            return None
        
        # Use V2 ensemble if available
        if self.ensemble_model and ENSEMBLE_AVAILABLE:
            try:
                # Create DataFrame for V2
                features_df = pd.DataFrame([features])
                features_df['start_frame'] = frame_idx
                features_df['end_frame'] = frame_idx + 10
                features_df['comment'] = f"frame_{frame_idx}"
                features_df['clip_name'] = "analysis"
                features_df['scene_id'] = "current"
                features_df['total_frames'] = len(motion_data)
                features_df['fps'] = 10.0
                
                # Preprocess and predict
                X, y, processed = preprocess_data(features_df)
                predictions = self.ensemble_model.predict(X)
                prediction = predictions[0]
                
                # Classify behavior
                behavior = self._classify_behavior(features, prediction)
                confidence = self._calculate_confidence(prediction)
                risk_level = self._assess_risk(features, prediction)
                risk_score = confidence * self.risk_multipliers.get(risk_level, 0.25)
                
                return {
                    'frame_idx': frame_idx,
                    'classification': behavior,
                    'confidence': confidence,
                    'risk_level': risk_level,
                    'risk_score': risk_score,
                    'raw_prediction': float(prediction),
                    'features': features
                }
                
            except Exception as e:
                logger.error(f"V2 prediction failed for frame {frame_idx}: {e}")
        
        # Fallback analysis
        return {
            'frame_idx': frame_idx,
            'classification': 'other',
            'confidence': 0.5,
            'risk_level': 'LOW',
            'risk_score': 0.3,
            'raw_prediction': 0.5,
            'features': features
        }
    
    def _classify_behavior(self, features: Dict, prediction: float) -> str:
        """Classify driving behavior"""
        if prediction > 0.8:
            if features['max_deceleration'] < -15 and features['jerk_rms'] > 35:
                return 'nearmiss'
            elif features['velocity_std'] < 1.0 and features['acceleration_std'] > 8:
                return 'hesitation'
            elif abs(features['velocity_mean']) > 10:
                return 'overshoot'
            else:
                return 'other'
        elif prediction > 0.6:
            return 'hesitation' if features['jerk_rms'] > 20 else 'other'
        else:
            return 'normal'
    
    def _calculate_confidence(self, prediction: float) -> float:
        """Calculate confidence score"""
        return min(0.95, max(0.3, abs(prediction - 0.5) * 2 + 0.1))
    
    def _assess_risk(self, features: Dict, prediction: float) -> str:
        """Assess risk level"""
        risk_factors = 0
        if features['jerk_rms'] > 35:
            risk_factors += 1
        if features['max_deceleration'] < -15:
            risk_factors += 1
        if features['motion_smoothness'] < 0.015:
            risk_factors += 1
        if prediction > 0.8:
            risk_factors += 1
        
        if risk_factors >= 3:
            return 'HIGH'
        elif risk_factors >= 2:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _count_zero_crossings(self, series: pd.Series) -> int:
        """Count zero crossings"""
        if len(series) < 2:
            return 0
        return int(((series[:-1] * series[1:]) < 0).sum())
    
    def _count_sign_changes(self, series: pd.Series) -> int:
        """Count sign changes"""
        if len(series) < 2:
            return 0
        signs = np.sign(series)
        return int((np.diff(signs) != 0).sum())

class ClusterAnalyzer:
    """Analyzes full scenes and creates clusters with annotation bias"""
    
    def __init__(self):
        self.config = get_config('clustering')
        self.annotation_loader = AnnotationLoader()
    
    def analyze_full_scene(self, motion_data: pd.DataFrame, ensemble_model, progress_callback=None, scene_id: str = None) -> Dict:
        """Analyze full scene with annotation bias"""
        
        print(f"🔍 Starting scene analysis for: {scene_id}")
        start_time = time.time()
        
        # Get annotations for this scene
        annotated_segments = self.annotation_loader.get_segments_for_scene(scene_id) if scene_id else []
        
        # Determine frames to analyze
        if annotated_segments:
            # Focus on annotated regions
            frame_indices = self._get_annotated_frame_indices(annotated_segments, len(motion_data))
            print(f"📝 Annotation-biased analysis: {len(frame_indices)} frames")
        else:
            # Analyze all frames with sampling
            sampling_rate = self.config.get('sampling_rate', 1)
            frame_indices = list(range(0, len(motion_data), sampling_rate))
            print(f"🔍 Full analysis: {len(frame_indices)} frames")
        
        # Analyze frames
        frame_analyzer = FrameAnalyzer(ensemble_model)
        frame_results = []
        
        for i, frame_idx in enumerate(frame_indices):
            if progress_callback:
                try:
                    progress_callback(i, len(frame_indices), f"Analyzing frame {frame_idx}")
                except:
                    pass
            
            result = frame_analyzer.analyze_frame(motion_data, frame_idx)
            if result:
                # Apply annotation bias
                if self._is_in_annotated_region(frame_idx, annotated_segments):
                    result['annotation_bias'] = True
                    result['confidence'] = max(result['confidence'], 0.4)  # Boost confidence
                
                frame_results.append(result)
        
        # Create clusters
        clusters = self._create_clusters(frame_results, annotated_segments)
        top_clusters = self._select_top_clusters(clusters)
        
        elapsed = time.time() - start_time
        print(f"✅ Analysis complete: {len(top_clusters)} clusters in {elapsed:.1f}s")
        
        return {
            'frame_predictions': frame_results,
            'all_clusters': clusters,
            'top_clusters': top_clusters,
            'annotation_bias': len(annotated_segments) > 0,
            'annotated_segments': annotated_segments
        }
    
    def _get_annotated_frame_indices(self, segments: List[Dict], total_frames: int) -> List[int]:
        """Get frame indices for annotated regions with padding"""
        indices = []
        for segment in segments:
            start = segment['padded_start']
            end = min(segment['padded_end'], total_frames - 1)
            indices.extend(range(start, end + 1))
        
        return sorted(list(set(indices)))  # Remove duplicates
    
    def _is_in_annotated_region(self, frame_idx: int, segments: List[Dict]) -> bool:
        """Check if frame is in an annotated region"""
        for segment in segments:
            if segment['padded_start'] <= frame_idx <= segment['padded_end']:
                return True
        return False
    
    def _create_clusters(self, frame_results: List[Dict], annotated_segments: List[Dict]) -> List[Dict]:
        """Create clusters from frame results"""
        if not frame_results:
            return []
        
        # Sort by frame index
        sorted_results = sorted(frame_results, key=lambda x: x['frame_idx'])
        
        clusters = []
        current_cluster = None
        gap_threshold = self.config.get('gap_threshold_frames', 10)
        
        for result in sorted_results:
            if result['classification'] == 'normal':
                continue
            
            frame_idx = result['frame_idx']
            classification = result['classification']
            
            # Start new cluster or continue existing
            if (current_cluster is None or 
                current_cluster['classification'] != classification or
                frame_idx - current_cluster['end_frame'] > gap_threshold):
                
                if current_cluster:
                    clusters.append(self._finalize_cluster(current_cluster, annotated_segments))
                
                current_cluster = {
                    'classification': classification,
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
            clusters.append(self._finalize_cluster(current_cluster, annotated_segments))
        
        return clusters
    
    def _finalize_cluster(self, cluster: Dict, annotated_segments: List[Dict]) -> Dict:
        """Finalize cluster with scoring"""
        predictions = cluster['predictions']
        
        # Calculate metrics
        avg_confidence = np.mean([p['confidence'] for p in predictions])
        max_confidence = np.max([p['confidence'] for p in predictions])
        risk_levels = [p['risk_level'] for p in predictions]
        dominant_risk = max(set(risk_levels), key=risk_levels.count)
        
        duration_frames = cluster['end_frame'] - cluster['start_frame'] + 1
        duration_seconds = duration_frames * 0.1
        
        # Check if in annotated region
        in_annotation = any(
            any(segment['padded_start'] <= frame <= segment['padded_end'] 
                for segment in annotated_segments)
            for frame in cluster['frames']
        )
        
        # Calculate score with annotation bias
        risk_multipliers = get_config('risk_multipliers')
        base_score = avg_confidence * risk_multipliers.get(dominant_risk, 0.25)
        annotation_bonus = 2.0 if in_annotation else 1.0
        final_score = base_score * annotation_bonus
        
        cluster.update({
            'avg_confidence': avg_confidence,
            'max_confidence': max_confidence,
            'dominant_risk_level': dominant_risk,
            'duration_frames': duration_frames,
            'duration_seconds': duration_seconds,
            'final_score': final_score,
            'in_annotated_region': in_annotation,
            'annotation_bonus': annotation_bonus
        })
        
        return cluster
    
    def _select_top_clusters(self, clusters: List[Dict]) -> List[Dict]:
        """Select top clusters by score"""
        if not clusters:
            return []
        
        # Filter by minimum duration
        min_duration = 3
        valid_clusters = [c for c in clusters if c['duration_frames'] >= min_duration]
        
        # Sort by score and take top k
        k = self.config.get('top_k_clusters', 2)
        sorted_clusters = sorted(valid_clusters, key=lambda x: x['final_score'], reverse=True)
        
        return sorted_clusters[:k]

class EnsemblePanel:
    """GUI panel for ensemble analysis results"""
    
    def __init__(self, parent):
        self.parent = parent
        self.ensemble_model = None
        self.setup_ui()
        self.load_model()
    
    def setup_ui(self):
        """Setup the UI components"""
        # Main frame
        self.frame = tk.LabelFrame(self.parent, text="V2 Ensemble Analysis", font=('Arial', 14, 'bold'))
        self.frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Results display
        results_frame = tk.Frame(self.frame)
        results_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(results_frame, text="Behavior:", font=('Arial', 12, 'bold')).grid(row=0, column=0, sticky='w')
        self.behavior_label = tk.Label(results_frame, text="Not analyzed", font=('Arial', 12), fg='blue')
        self.behavior_label.grid(row=0, column=1, sticky='w', padx=10)
        
        tk.Label(results_frame, text="Confidence:", font=('Arial', 12, 'bold')).grid(row=1, column=0, sticky='w')
        self.confidence_label = tk.Label(results_frame, text="0.000", font=('Arial', 12))
        self.confidence_label.grid(row=1, column=1, sticky='w', padx=10)
        
        tk.Label(results_frame, text="Risk Level:", font=('Arial', 12, 'bold')).grid(row=2, column=0, sticky='w')
        self.risk_label = tk.Label(results_frame, text="UNKNOWN", font=('Arial', 12, 'bold'), fg='gray')
        self.risk_label.grid(row=2, column=1, sticky='w', padx=10)
        
        # Status display
        status_frame = tk.LabelFrame(self.frame, text="Model Status", font=('Arial', 12, 'bold'))
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.status_text = tk.Text(status_frame, height=4, width=50, font=('Courier', 10))
        self.status_text.pack(padx=5, pady=5)
    
    def load_model(self):
        """Load the V2 ensemble model"""
        try:
            if not ENSEMBLE_AVAILABLE:
                self.status_text.insert(tk.END, "⚠️ V2 ensemble not available\n")
                return
            
            # Check model files
            model_files = [
                f"{MODEL_PREFIX}_xgboost.pkl",
                f"{MODEL_PREFIX}_cnn_attention.pkl",
                f"{MODEL_PREFIX}_autoencoder.pkl", 
                f"{MODEL_PREFIX}_svm_rbf.pkl"
            ]
            
            missing = [f for f in model_files if not os.path.exists(f)]
            if missing:
                self.status_text.insert(tk.END, "❌ Missing model files:\n")
                for file in missing:
                    self.status_text.insert(tk.END, f"  • {os.path.basename(file)}\n")
                return
            
            # Load ensemble
            self.ensemble_model = EnsembleTriageBrain()
            self.ensemble_model.load(MODEL_PREFIX)
            
            self.status_text.insert(tk.END, "✅ V2 Ensemble loaded successfully!\n")
            self.status_text.insert(tk.END, f"📂 Models: {len(self.ensemble_model.models)}\n")
            
        except Exception as e:
            self.status_text.insert(tk.END, f"❌ Error loading models: {e}\n")
            self.ensemble_model = None
    
    def update_display(self, top_clusters: List[Dict]):
        """Update display with analysis results"""
        
        if not top_clusters:
            self.behavior_label.config(text="No risks detected")
            self.confidence_label.config(text="N/A")
            self.risk_label.config(text="LOW", fg='green')
            
            self.status_text.delete(1.0, tk.END)
            self.status_text.insert(tk.END, "Analysis complete - No significant risks found.\n")
            return
        
        # Show top cluster
        top_cluster = top_clusters[0]
        
        self.behavior_label.config(text=top_cluster['classification'])
        self.confidence_label.config(text=f"{top_cluster['avg_confidence']:.3f}")
        
        # Risk level with color
        risk_level = top_cluster['dominant_risk_level']
        colors = {'LOW': 'green', 'MEDIUM': 'orange', 'HIGH': 'red'}
        self.risk_label.config(text=risk_level, fg=colors.get(risk_level, 'gray'))
        
        # Show summary
        self.status_text.delete(1.0, tk.END)
        self.status_text.insert(tk.END, f"Found {len(top_clusters)} risk clusters:\n\n")
        
        for i, cluster in enumerate(top_clusters):
            annotation_mark = "📝" if cluster.get('in_annotated_region', False) else "🔍"
            self.status_text.insert(tk.END, 
                f"{annotation_mark} {i+1}. {cluster['classification']} "
                f"({cluster['duration_seconds']:.1f}s, score: {cluster['final_score']:.3f})\n")


# Main test function
if __name__ == "__main__":
    print("🧪 Testing Clean Ensemble Engine...")
    
    # Test annotation loading
    loader = AnnotationLoader()
    print(f"Loaded {len(loader.annotations_by_scene_id)} annotations")
    
    if loader.annotations_by_scene_id:
        test_scene_id = list(loader.annotations_by_scene_id.keys())[0]
        segments = loader.get_segments_for_scene(test_scene_id)
        print(f"Test scene {test_scene_id}: {len(segments)} segments")
    
    # Test V2 model loading
    if ENSEMBLE_AVAILABLE:
        try:
            ensemble = EnsembleTriageBrain()
            ensemble.load(MODEL_PREFIX)
            print("✅ V2 model loaded successfully")
            
            # Test with sample data
            motion_data = pd.DataFrame({
                'time': np.arange(100) * 0.1,
                'velocity': 5.0 + np.random.normal(0, 1, 100),
                'acceleration': np.random.normal(0, 2, 100),
                'jerk': np.random.normal(0, 10, 100)
            })
            
            # Test frame analysis
            analyzer = FrameAnalyzer(ensemble)
            result = analyzer.analyze_frame(motion_data, 50)
            if result:
                print(f"✅ Frame analysis: {result['classification']} (conf: {result['confidence']:.3f})")
            
            # Test full scene analysis
            cluster_analyzer = ClusterAnalyzer()
            if loader.annotations_by_scene_id:
                test_scene_id = list(loader.annotations_by_scene_id.keys())[0]
                scene_results = cluster_analyzer.analyze_full_scene(
                    motion_data, ensemble, scene_id=test_scene_id
                )
                print(f"✅ Scene analysis: {len(scene_results['top_clusters'])} clusters")
                print(f"📝 Annotation bias: {scene_results['annotation_bias']}")
            
        except Exception as e:
            print(f"❌ V2 test failed: {e}")
    
    print("🎉 Clean ensemble engine test complete!")