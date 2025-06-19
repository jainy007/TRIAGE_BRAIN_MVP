#!/usr/bin/env python3
"""
Utilities for Triage Brain GUI
Logging, caching, data loading, and configuration
"""

import logging
import os
import time
import pickle
import pandas as pd
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
from multiprocessing import Pool, cpu_count

# Configuration
CONFIG = {
    'risk_multipliers': {
        'LOW': 0.25,
        'MEDIUM': 0.5, 
        'HIGH': 0.75,
        'CRITICAL': 1.0
    },
    'ensemble_weights': {
        'advanced_ml': 0.7,
        'simple_ml': 0.2,
        'rule_based': 0.1
    },
    'confidence_thresholds': {
        'high_confidence': 0.7,
        'medium_confidence': 0.4,
        'low_confidence': 0.2
    },
    'clustering': {
        'min_duration_frames': 20,  # Updated based on gold standard (2 seconds)
        'gap_threshold_frames': 10,  # 1 second gap tolerance
        'top_k_clusters': 2,
        'sampling_rate': 1  # Every frame instead of every other frame
    },
    'filtering': {
        'velocity_window': 5,
        'acceleration_window': 5, 
        'jerk_window': 5,
        'acceleration_clip': [-20, 20],
        'jerk_clip': [-50, 50]
    },
    'gold_standard_path': '/home/jainy007/PEM/mvp3_0/annotated_clips.jsonl'  # Fixed filename
}

class GoldStandardAnalyzer:
    """Analyzes gold-standard annotations to improve clustering"""
    
    def __init__(self):
        self.patterns = {}
        self.behavior_thresholds = {}
        
    def analyze_patterns(self, jsonl_path: str) -> Dict:
        """Analyze gold-standard annotation patterns"""
        
        logger.info("Analyzing gold-standard annotation patterns...")
        
        try:
            # Load annotations
            annotations = []
            with open(jsonl_path, 'r') as f:
                for line in f:
                    annotations.append(json.loads(line))
            
            # Extract segment statistics
            segment_stats = []
            behavior_durations = {}
            
            for clip in annotations:
                for segment in clip.get('segments', []):
                    start_frame = segment['start']
                    end_frame = segment['end']
                    duration_frames = end_frame - start_frame + 1
                    duration_seconds = duration_frames / clip.get('fps', 10)
                    comment = segment.get('comment', '').lower()
                    
                    # Classify behavior from comment
                    behavior = self._classify_behavior_from_comment(comment)
                    
                    segment_stat = {
                        'duration_frames': duration_frames,
                        'duration_seconds': duration_seconds,
                        'behavior': behavior,
                        'comment': comment
                    }
                    segment_stats.append(segment_stat)
                    
                    # Group by behavior
                    if behavior not in behavior_durations:
                        behavior_durations[behavior] = []
                    behavior_durations[behavior].append(duration_frames)
            
            # Calculate statistics
            all_durations = [s['duration_frames'] for s in segment_stats]
            
            patterns = {
                'total_segments': len(segment_stats),
                'total_clips': len(annotations),
                'avg_segments_per_clip': len(segment_stats) / len(annotations),
                'duration_stats': {
                    'mean_frames': np.mean(all_durations),
                    'median_frames': np.median(all_durations),
                    'min_frames': np.min(all_durations),
                    'max_frames': np.max(all_durations),
                    'std_frames': np.std(all_durations),
                    'percentile_25': np.percentile(all_durations, 25),
                    'percentile_75': np.percentile(all_durations, 75)
                },
                'behavior_stats': {},
                'recommended_thresholds': {}
            }
            
            # Behavior-specific analysis
            for behavior, durations in behavior_durations.items():
                if len(durations) > 0:
                    patterns['behavior_stats'][behavior] = {
                        'count': len(durations),
                        'mean_frames': np.mean(durations),
                        'median_frames': np.median(durations),
                        'min_frames': np.min(durations),
                        'max_frames': np.max(durations),
                        'std_frames': np.std(durations)
                    }
                    
                    # Recommend minimum threshold (25th percentile)
                    min_threshold = max(10, int(np.percentile(durations, 25)))
                    patterns['recommended_thresholds'][behavior] = min_threshold
            
            # Global recommended threshold
            global_min = max(15, int(np.percentile(all_durations, 20)))  # 20th percentile
            patterns['recommended_thresholds']['global'] = global_min
            
            logger.info(f"Gold-standard analysis complete:")
            logger.info(f"  Total segments: {patterns['total_segments']}")
            logger.info(f"  Avg duration: {patterns['duration_stats']['mean_frames']:.1f} frames")
            logger.info(f"  Recommended min threshold: {global_min} frames")
            
            self.patterns = patterns
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to analyze gold-standard patterns: {e}")
            return {}
    
    def _classify_behavior_from_comment(self, comment: str) -> str:
        """Classify behavior type from annotation comment"""
        comment = comment.lower()
        
        # Define behavior keywords
        behavior_keywords = {
            'overshoot': ['overshoot', 'oversh'],
            'nearmiss': ['near miss', 'nearmiss', 'narrow pass'],
            'hesitation': ['hesitation', 'hesitant', 'nervous', 'halt'],
            'pedestrian': ['pedestrian', 'man', 'woman', 'people', 'crossing'],
            'bicycle': ['bicycle', 'bike'],
            'oversteering': ['oversteer', 'oversteering'],
            'vehicle_interaction': ['vehicle', 'car', 'bus', 'truck'],
            'traffic_control': ['stop sign', 'traffic', 'light', 'signal'],
            'road_conditions': ['road', 'lane', 'cones', 'toll']
        }
        
        # Find best match
        for behavior, keywords in behavior_keywords.items():
            if any(keyword in comment for keyword in keywords):
                return behavior
        
        return 'other'
    
    def get_clustering_params(self) -> Dict:
        """Get optimized clustering parameters based on gold standard"""
        if not self.patterns:
            # Fallback defaults
            return {
                'min_duration_frames': 20,
                'gap_threshold_frames': 10,
                'sampling_rate': 1
            }
        
        # Use global recommended threshold
        min_duration = self.patterns['recommended_thresholds'].get('global', 20)
        
        return {
            'min_duration_frames': min_duration,
            'gap_threshold_frames': max(5, min_duration // 2),  # Half of min duration
            'sampling_rate': 1  # Every frame for longer segments
        }

class TriageLogger:
    """Comprehensive logging system for Triage Brain"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.setup_loggers()
    
    def setup_loggers(self):
        """Setup different loggers for different components"""
        
        # Main application logger
        self.app_logger = self._create_logger(
            'triage_app', 
            'triage_brain.log',
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Analysis logger
        self.analysis_logger = self._create_logger(
            'triage_analysis',
            'analysis.log', 
            '%(asctime)s - ANALYSIS - %(message)s'
        )
        
        # Performance logger
        self.perf_logger = self._create_logger(
            'triage_performance',
            'performance.log',
            '%(asctime)s - PERF - %(message)s'
        )
    
    def _create_logger(self, name: str, filename: str, format_str: str):
        """Create a logger with file handler"""
        
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(os.path.join(self.log_dir, filename), mode='a')
        file_handler.setLevel(logging.INFO)
        
        # Console handler for errors only
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.ERROR)
        
        # Formatter
        formatter = logging.Formatter(format_str)
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def info(self, message: str):
        self.app_logger.info(message)
    
    def error(self, message: str):
        self.app_logger.error(message)
    
    def warning(self, message: str):
        self.app_logger.warning(message)
    
    def analysis(self, message: str):
        self.analysis_logger.info(message)
    
    def performance(self, metric: str, value: float, unit: str = ""):
        self.perf_logger.info(f"{metric}: {value:.3f} {unit}")
    
    def session_start(self, scene_id: str = None):
        self.app_logger.info("="*60)
        self.app_logger.info("TRIAGE BRAIN SESSION STARTED")
        self.app_logger.info(f"Timestamp: {datetime.now()}")
        if scene_id:
            self.app_logger.info(f"Scene ID: {scene_id}")
        self.app_logger.info("="*60)

# Global logger instance
logger = TriageLogger()

class CacheManager:
    """Handles analysis result caching"""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_cache_path(self, scene_id: str) -> str:
        """Get cache file path for a scene"""
        return os.path.join(self.cache_dir, f"analysis_{scene_id}.pkl")
    
    def load_cached_analysis(self, scene_id: str) -> Optional[Dict]:
        """Load cached analysis for a scene"""
        cache_path = self.get_cache_path(scene_id)
        
        if not os.path.exists(cache_path):
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            
            logger.info(f"Loaded cached analysis for scene {scene_id}")
            return cached_data
            
        except Exception as e:
            logger.error(f"Failed to load cache for {scene_id}: {e}")
            return None
    
    def save_analysis(self, scene_id: str, analysis_data: Dict) -> bool:
        """Save analysis results to cache"""
        cache_path = self.get_cache_path(scene_id)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(analysis_data, f)
            
            logger.info(f"Cached analysis for scene {scene_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache analysis for {scene_id}: {e}")
            return False

class DataLoader:
    """Handles scene data loading from Argoverse format"""
    
    def __init__(self):
        self.scenes_df = None
        self.annotated_scene_ids = set()
    
    def load_annotated_scene_ids(self, jsonl_path: str) -> set:
        """Load set of annotated scene IDs from the annotation file"""
        try:
            annotated_clips = set()
            logger.info(f"Loading annotated clips from: {jsonl_path}")
            
            with open(jsonl_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        annotation = json.loads(line)
                        clip_name = annotation['clip']
                        # Store the full clip name (including .mp4)
                        annotated_clips.add(clip_name)
                        
                        # Debug: print first few entries
                        if line_num <= 5:
                            logger.info(f"  Clip {line_num}: {clip_name}")
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON on line {line_num}: {e}")
                        continue
            
            logger.info(f"Found {len(annotated_clips)} annotated clips")
            self.annotated_scene_ids = annotated_clips
            
            # Debug: print sample of annotated clips
            sample_clips = list(annotated_clips)[:10]
            logger.info(f"Sample annotated clips: {sample_clips}")
            
            return annotated_clips
            
        except Exception as e:
            logger.error(f"Failed to load annotated scene IDs: {e}")
            return set()
    
    def load_scenes_csv(self, csv_paths: List[str] = None) -> Optional[pd.DataFrame]:
        """Load scene data from CSV, with option to filter to annotated scenes"""
        
        if csv_paths is None:
            csv_paths = [
                "tools/data_exploration/scene_summary.csv",
                "data_exploration/scene_summary.csv", 
                "scene_summary.csv"
            ]
        
        # First load annotated clip names
        gold_path = CONFIG.get('gold_standard_path', '')
        if gold_path and os.path.exists(gold_path):
            logger.info(f"Loading gold standard annotations from: {gold_path}")
            self.load_annotated_scene_ids(gold_path)
        else:
            logger.warning(f"Gold standard file not found: {gold_path}")
        
        for csv_path in csv_paths:
            if os.path.exists(csv_path):
                try:
                    logger.info(f"Loading scenes from CSV: {csv_path}")
                    all_scenes_df = pd.read_csv(csv_path)
                    logger.info(f"Total scenes in CSV: {len(all_scenes_df)}")
                    
                    # Debug: print sample scene paths
                    sample_scenes = all_scenes_df.head(3)
                    for idx, row in sample_scenes.iterrows():
                        logger.info(f"  Sample scene {idx}: path={row['path']}, scene_id={row['scene_id']}")
                    
                    # If we have annotations, try to create a mapping
                    if self.annotated_scene_ids:
                        logger.info(f"Attempting to map {len(all_scenes_df)} scenes to {len(self.annotated_scene_ids)} annotated clips")
                        
                        # Option 1: Try to create a scene mapping file if it doesn't exist
                        mapping_file = "scene_mapping.json"
                        scene_mapping = self._load_or_create_scene_mapping(mapping_file, all_scenes_df)
                        
                        if scene_mapping:
                            # Filter using the mapping
                            filtered_scenes = []
                            for _, row in all_scenes_df.iterrows():
                                scene_id = str(row['scene_id'])
                                if scene_id in scene_mapping:
                                    annotated_clip = scene_mapping[scene_id]
                                    if annotated_clip in self.annotated_scene_ids:
                                        logger.info(f"  ✓ Mapped: {scene_id} -> {annotated_clip}")
                                        filtered_scenes.append(row)
                            
                            if filtered_scenes:
                                self.scenes_df = pd.DataFrame(filtered_scenes)
                                logger.info(f"✅ Filtered to {len(self.scenes_df)} annotated scenes using mapping")
                            else:
                                logger.warning("❌ No scenes matched using mapping, using all scenes")
                                self.scenes_df = all_scenes_df
                        else:
                            # Option 2: Manual verification needed - use all scenes for now
                            logger.warning("⚠️  No scene mapping available. You'll need to create a mapping file.")
                            logger.info("   To create mapping: manually map your annotated clip names to Argoverse scene UUIDs")
                            logger.info(f"   Example mapping format in {mapping_file}:")
                            logger.info('   {"429dedd9-2d51-385d-9661-b15f4027e34d": "your_clip_name.mp4"}')
                            
                            self.scenes_df = all_scenes_df
                    else:
                        logger.warning("No annotated clips found, using all scenes")
                        self.scenes_df = all_scenes_df
                    
                    logger.info(f"Final result: {len(self.scenes_df)} scenes available for analysis")
                    return self.scenes_df
                    
                except Exception as e:
                    logger.error(f"Failed to load {csv_path}: {e}")
        
        logger.error("No scene CSV file found")
        return None
    
    def _load_or_create_scene_mapping(self, mapping_file: str, scenes_df: pd.DataFrame) -> Optional[Dict[str, str]]:
        """Load existing scene mapping or create interactive mapping"""
        
        # Try to load existing mapping
        if os.path.exists(mapping_file):
            try:
                with open(mapping_file, 'r') as f:
                    mapping = json.load(f)
                logger.info(f"Loaded existing scene mapping with {len(mapping)} entries")
                return mapping
            except Exception as e:
                logger.error(f"Failed to load mapping file: {e}")
        
        logger.info(f"Scene mapping file '{mapping_file}' not found")
        logger.info("You have two options:")
        logger.info("1. Create a manual mapping file (recommended)")
        logger.info("2. Use all scenes without filtering")
        logger.info("")
        logger.info("To create mapping file, you need to manually map:")
        logger.info("- Argoverse scene UUIDs (like '429dedd9-2d51-385d-9661-b15f4027e34d')")
        logger.info("- To your annotation clip names (from the JSONL file)")
        logger.info("")
        logger.info("Example scene_mapping.json:")
        logger.info('{')
        sample_scene = scenes_df.iloc[0] if len(scenes_df) > 0 else None
        if sample_scene is not None:
            logger.info(f'  "{sample_scene["scene_id"]}": "your_corresponding_clip_name.mp4",')
        logger.info('  "another-uuid-here": "another_clip_name.mp4"')
        logger.info('}')
        
        return None
    
    def get_scene_info(self, scene_idx: int) -> Optional[pd.Series]:
        """Get scene information by index"""
        if self.scenes_df is None or scene_idx >= len(self.scenes_df):
            return None
        return self.scenes_df.iloc[scene_idx]
    
    def load_pose_data(self, pose_path: str) -> Optional[pd.DataFrame]:
        """Load pose data from feather file"""
        try:
            if not os.path.exists(pose_path):
                logger.error(f"Pose file not found: {pose_path}")
                return None
            
            pose_df = pd.read_feather(pose_path)
            logger.info(f"Loaded pose data: {len(pose_df)} records")
            return pose_df
            
        except Exception as e:
            logger.error(f"Failed to load pose data from {pose_path}: {e}")
            return None
    
    def get_camera_frames(self, scene_path: str, camera_view: str = "ring_front_center") -> List[str]:
        """Get list of camera frame files"""
        try:
            camera_path = os.path.join(scene_path, "sensors", "cameras", camera_view)
            logger.info(f"Looking for camera frames in: {camera_path}")
            
            if not os.path.exists(camera_path):
                logger.error(f"Camera path not found: {camera_path}")
                return []
            
            # Get all image files
            image_files = sorted([f for f in os.listdir(camera_path) 
                                if f.endswith(('.jpg', '.jpeg', '.png'))])
            
            if not image_files:
                logger.error(f"No image files found in {camera_path}")
                return []
            
            frame_paths = [os.path.join(camera_path, f) for f in image_files]
            logger.info(f"Found {len(frame_paths)} camera frames")
            
            # Debug: show first few frame paths
            sample_frames = frame_paths[:3]
            for i, frame_path in enumerate(sample_frames):
                logger.info(f"  Frame {i}: {frame_path}")
            
            return frame_paths
            
        except Exception as e:
            logger.error(f"Failed to get camera frames: {e}")
            return []

class ExportManager:
    """Handles export operations"""
    
    def __init__(self):
        pass
    
    def export_analysis_results(self, filename: str, scene_info: pd.Series, 
                               analysis_data: Dict, motion_summary: Dict) -> bool:
        """Export comprehensive analysis results"""
        try:
            results = {
                'export_metadata': {
                    'export_timestamp': datetime.now().isoformat(),
                    'scene_id': scene_info['scene_id'],
                    'scene_path': scene_info['path'],
                    'pose_path': scene_info['pose_path'],
                    'total_frames': int(scene_info['num_frames_total']),
                    'analysis_timestamp': analysis_data.get('analysis_timestamp', 'unknown')
                },
                'scene_summary': {
                    'duration_seconds': float(scene_info['num_frames_total']) * 0.1,
                    'motion_summary': motion_summary
                },
                'cluster_analysis': {
                    'top_clusters': analysis_data.get('top_clusters', []),
                    'all_clusters': analysis_data.get('all_clusters', []),
                    'total_analyzed_frames': len(analysis_data.get('frame_predictions', [])),
                    'cluster_summary': self._generate_cluster_summary(analysis_data.get('top_clusters', []))
                },
                'frame_level_predictions': {
                    'predictions': analysis_data.get('frame_predictions', []),
                    'risk_distribution': self._calculate_risk_distribution(analysis_data.get('frame_predictions', []))
                }
            }
            
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Exported analysis results to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export results: {e}")
            return False
    
    def _generate_cluster_summary(self, top_clusters: List[Dict]) -> Dict:
        """Generate summary statistics for clusters"""
        if not top_clusters:
            return {}
        
        behaviors = [c['classification'] for c in top_clusters]
        risk_levels = [c['dominant_risk_level'] for c in top_clusters]
        durations = [c['duration_seconds'] for c in top_clusters]
        
        return {
            'total_clusters': len(top_clusters),
            'behavior_distribution': {b: behaviors.count(b) for b in set(behaviors)},
            'risk_distribution': {r: risk_levels.count(r) for r in set(risk_levels)},
            'total_risky_duration': sum(durations),
            'avg_cluster_duration': sum(durations) / len(durations) if durations else 0,
            'max_risk_score': max([c['final_score'] for c in top_clusters]) if top_clusters else 0
        }
    
    def _calculate_risk_distribution(self, frame_predictions: List[Dict]) -> Dict:
        """Calculate distribution of risk levels across all frames"""
        if not frame_predictions:
            return {}
        
        risk_levels = [p['risk_level'] for p in frame_predictions]
        total_frames = len(frame_predictions)
        
        return {
            'total_frames_analyzed': total_frames,
            'risk_distribution': {
                level: {'count': risk_levels.count(level), 
                       'percentage': (risk_levels.count(level) / total_frames) * 100}
                for level in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
            },
            'avg_confidence': sum([p['confidence'] for p in frame_predictions]) / total_frames,
            'avg_risk_score': sum([p['risk_score'] for p in frame_predictions]) / total_frames
        }

# Global instances
cache_manager = CacheManager()
data_loader = DataLoader()
export_manager = ExportManager()

def get_config(key: str) -> Any:
    """Get configuration value"""
    keys = key.split('.')
    value = CONFIG
    for k in keys:
        value = value.get(k, {})
    return value