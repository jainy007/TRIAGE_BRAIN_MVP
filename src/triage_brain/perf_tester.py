#!/usr/bin/env python3
"""
Enhanced Ensemble Performance Tester - Tests both individual models and ensemble decisions
"""

import json
import numpy as np
import pandas as pd
import time
import torch
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
from pathlib import Path
import sys
import os

# Add triage brain models to path
sys.path.append('src/triage_brain')

class EnhancedEnsemblePerformanceTester:
    """Enhanced performance tester for ensemble system"""
    
    def __init__(self):
        self.test_results = []
        self.performance_metrics = {}
        self.ground_truth_mapping = self._create_ground_truth_mapping()
        
        # Initialize ensemble system
        self.ensemble_system = None
        
        # Individual models for comparison
        self.advanced_model = None
        self.simple_model = None
        self.rule_model = None
        self.dl_model = None
        self.dl_behavior_classes = None
        
    def _create_ground_truth_mapping(self) -> Dict[str, str]:
        """Map annotation comments to expected behaviors"""
        return {
            # Traffic control
            'stop sign overshoot': 'overshoot',
            'stop overshoot': 'overshoot', 
            'stop sine overshoot': 'overshoot',
            'left turn overshoot nearmiss': 'overshoot',
            
            # Near misses
            'near miss': 'nearmiss',
            'risky launch - near miss': 'nearmiss',
            'narrow near miss': 'nearmiss',
            'oversteer near miss': 'nearmiss',
            'oversteering near miss': 'nearmiss',
            'leading car backed up -near miss': 'nearmiss',
            'in parking lane - near miss': 'nearmiss',
            'near miss - men standing by parked car to right': 'nearmiss',
            'near miss - man entering vehicle to the right': 'nearmiss',
            
            # Pedestrians
            'man crossing': 'pedestrian',
            'pedestrian crossing': 'pedestrian',
            'pedestrian running on left walkway': 'pedestrian',
            'women on ego lane': 'pedestrian',
            'man with cradle ahead': 'pedestrian',
            'man crossing with dog ahead': 'pedestrian',
            'pedestrian crossed ahead': 'pedestrian',
            'man in horizon': 'pedestrian',
            
            # Bicycles
            'bicycle': 'bicycle',
            'bicycle crossed': 'bicycle',
            'letting bicycle pass from right': 'bicycle',
            'right turn hesitation - letting bicycle pass from right': 'bicycle',
            
            # Traffic cones/obstacles
            'traffic cones avoided': 'obstacles',
            'traffic cones': 'obstacles',
            'traffic cones in horizon': 'obstacles',
            'overwhelming traffic cones all around': 'obstacles',
            'obstacle - car on road': 'obstacles',
            
            # Steering issues
            'oversteering': 'oversteering',
            'turn oversteering': 'oversteering',
            'oversteer unnessarily': 'oversteering',
            'left turn oversteering': 'oversteering',
            'right turn oversteering - near miss': 'oversteering',
            
            # Hesitation
            'protected left hesitation': 'hesitation',
            'right turn hesitation': 'hesitation',
            'unnecessary hesitation': 'hesitation',
            'unnecessary hesitation to take unprotected left': 'hesitation',
            'intersection hesitation': 'hesitation',
            'following hesitation': 'hesitation',
            
            # Stop behaviors
            'unnecessary halt': 'stop_behavior',
            'nervous stop': 'stop_behavior',
            'stop sign crawl': 'stop_behavior',
            
            # Turning
            'perfect right steer': 'turning',
            'nervous right turn': 'turning',
            'left turn and lane change to right in quick succession': 'turning',
            
            # Lane changes
            'narrow pass': 'lane_change',
            'narrow pass- courier vehicle parked on curb to left': 'lane_change',
            'narrow pass - needed to swevel to the left lane a bit': 'lane_change',
            'narrow pass - needed to swevel to the left a bit': 'lane_change',
            'unnessary swevel to the right': 'lane_change',
            
            # Parking
            'car parallel parking ahead': 'parking',
            'single lane diverges to dual lane, disregards lead vehicle and overtakes': 'parking',
            
            # Road conditions
            'change in road color': 'road_conditions',
            'road color change': 'road_conditions',
            'dark to bright occlusion': 'road_conditions',
            'blending horizon color with red light': 'road_conditions',
            
            # Vehicle interactions
            'misalligned leading vehicle': 'vehicle_interaction',
            'leading vehicle': 'vehicle_interaction',
            'police leading vehicle - no siren': 'vehicle_interaction',
            'vehicles overtaking from right': 'vehicle_interaction',
            'stopped car on right lane': 'vehicle_interaction',
            'leading vehicle issues': 'vehicle_interaction',
            'blind right turn car in front': 'vehicle_interaction',
            'bus coming on lane from right': 'vehicle_interaction',
            '3 vehicle near miss - confusion': 'vehicle_interaction',
            
            # Other/unknown
            'unknown object ahead': 'other',
            'toll gate': 'other',
            'bump ignored': 'other',
            'stop sign no overshoot': 'other',
            'unprotected left with car parked ahead': 'other'
        }
    
    def load_ensemble_system(self):
        """Load the complete ensemble system"""
        
        print("🔧 Loading Enhanced Ensemble System...")
        
        try:
            # Import your existing ensemble system
            from ensemble_triage_brain import EnhancedTriageEnsemble
            
            self.ensemble_system = EnhancedTriageEnsemble()
            
            # Load all models into ensemble
            model_paths = {
                'advanced': 'assets/models/advanced_triage_brain.pkl',
                'simple': 'assets/models/simple_ml_triage.pkl',
                'rule': 'practical_triage_brain.json',
                'dl': 'best_fixed_model.pth'
            }
            
            self.ensemble_system.load_existing_models(model_paths)
            
            print("✅ Enhanced Ensemble System loaded")
            return True
            
        except Exception as e:
            print(f"❌ Ensemble system loading failed: {e}")
            
            # Fallback: Load individual models for comparison
            self._load_individual_models()
            return False
    
    def _load_individual_models(self):
        """Fallback: Load individual models separately"""
        
        print("🔄 Loading individual models as fallback...")
        
        # Load Advanced ML
        try:
            import joblib
            self.advanced_model = joblib.load("assets/models/advanced_triage_brain.pkl")
            print("✅ Advanced ML model loaded")
        except Exception as e:
            print(f"❌ Advanced ML failed: {e}")
        
        # Load Simple ML
        try:
            import joblib
            self.simple_model = joblib.load("assets/models/simple_ml_triage.pkl")
            print("✅ Simple ML model loaded")
        except Exception as e:
            print(f"❌ Simple ML failed: {e}")
        
        # Load Rule-based
        try:
            with open("practical_triage_brain.json", 'r') as f:
                self.rule_model = json.load(f)
            print("✅ Rule-based model loaded")
        except Exception as e:
            print(f"❌ Rule-based failed: {e}")
        
        # Load Deep Learning
        try:
            self.dl_model, self.dl_behavior_classes = self._load_dl_model_dynamic()
            print("✅ Deep Learning model loaded")
        except Exception as e:
            print(f"❌ Deep Learning failed: {e}")
    
    def _load_dl_model_dynamic(self):
        """Load DL model with dynamic size detection"""
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load the saved state dict to inspect dimensions
        state_dict = torch.load("best_fixed_model.pth", map_location=device)
        
        # Detect dimensions
        classifier_weight_shape = state_dict['classifier.3.weight'].shape
        num_behaviors = classifier_weight_shape[0]
        
        conv_weight_shape = state_dict['temporal_conv.0.weight'].shape
        num_features = conv_weight_shape[1]
        
        print(f"🔍 Detected DL model: {num_features} features, {num_behaviors} behaviors")
        
        # Create model with detected dimensions
        class DynamicTemporalNet(torch.nn.Module):
            def __init__(self, input_features, num_behaviors, hidden_dim=48):
                super().__init__()
                
                self.temporal_conv = torch.nn.Sequential(
                    torch.nn.Conv1d(input_features, 24, kernel_size=3, padding=1),
                    torch.nn.BatchNorm1d(24),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.2),
                    
                    torch.nn.Conv1d(24, 48, kernel_size=3, padding=1),
                    torch.nn.BatchNorm1d(48),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.2)
                )
                
                self.lstm = torch.nn.LSTM(
                    input_size=48,
                    hidden_size=hidden_dim,
                    num_layers=1,
                    bidirectional=True,
                    batch_first=True
                )
                
                self.attention = torch.nn.Linear(hidden_dim * 2, 1)
                
                self.classifier = torch.nn.Sequential(
                    torch.nn.Linear(hidden_dim * 2, 64),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.4),
                    torch.nn.Linear(64, num_behaviors)
                )
                
                self.risk_regressor = torch.nn.Sequential(
                    torch.nn.Linear(hidden_dim * 2, 32),
                    torch.nn.ReLU(),
                    torch.nn.Linear(32, 1),
                    torch.nn.Sigmoid()
                )
            
            def forward(self, x):
                batch_size, seq_len, features = x.shape
                
                x_conv = x.transpose(1, 2)
                conv_features = self.temporal_conv(x_conv)
                conv_features = conv_features.transpose(1, 2)
                
                lstm_out, _ = self.lstm(conv_features)
                
                attention_scores = self.attention(lstm_out)
                attention_weights = torch.nn.functional.softmax(attention_scores, dim=1)
                attended = torch.sum(lstm_out * attention_weights, dim=1)
                
                behavior_logits = self.classifier(attended)
                risk_score = self.risk_regressor(attended).squeeze(-1)
                
                return behavior_logits, risk_score
        
        # Create and load model
        model = DynamicTemporalNet(num_features, num_behaviors)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        # Create behavior class names based on your mapping
        behavior_classes = [
            'bicycle', 'hesitation', 'lane_change', 'nearmiss', 'obstacles',
            'other', 'overshoot', 'oversteering', 'parking', 'pedestrian',
            'road_conditions', 'stop_behavior', 'turning'
        ]
        
        return model, behavior_classes
    
    def load_test_data(self) -> List[Dict]:
        """Load test data with enhanced matching"""
        
        print("📁 Loading test data...")
        
        # Load annotations
        annotations = {}
        with open("assets/data/annotated_clips.jsonl", 'r') as f:
            for line in f:
                clip_data = json.loads(line)
                annotations[clip_data['clip']] = clip_data['segments']
        
        # Load feature vectors
        feature_vectors = []
        feature_paths = [
            "assets/data/feature_vectors.jsonl",
            "/home/jainy007/PEM/triage_brain/assets/data/feature_vectors.jsonl"
        ]
        
        for path in feature_paths:
            if os.path.exists(path):
                print(f"📁 Found features at: {path}")
                with open(path, 'r') as f:
                    for line in f:
                        feature_vectors.append(json.loads(line))
                break
        
        # Create test cases with enhanced matching
        test_cases = []
        
        for feature_vector in feature_vectors:
            clip_name = feature_vector['clip_name']
            start_frame = feature_vector['start_frame']
            end_frame = feature_vector['end_frame']
            
            if clip_name in annotations:
                for annotation in annotations[clip_name]:
                    # More flexible frame matching
                    frame_overlap = self._calculate_frame_overlap(
                        (start_frame, end_frame),
                        (annotation['start'], annotation['end'])
                    )
                    
                    if frame_overlap > 0.3:  # At least 30% overlap
                        comment = annotation['comment']
                        expected_behavior = self._map_comment_to_expected(comment)
                        
                        test_case = {
                            'clip_name': clip_name,
                            'start_frame': start_frame,
                            'end_frame': end_frame,
                            'comment': comment,
                            'expected_behavior': expected_behavior,
                            'features': feature_vector,
                            'frame_overlap': frame_overlap
                        }
                        
                        test_cases.append(test_case)
                        break
        
        print(f"✅ Created {len(test_cases)} test cases")
        
        # Show expected behavior distribution
        expected_distribution = Counter([tc['expected_behavior'] for tc in test_cases])
        print(f"📊 Expected behavior distribution:")
        for behavior, count in expected_distribution.most_common():
            print(f"   {behavior}: {count}")
        
        return test_cases
    
    def _calculate_frame_overlap(self, range1: Tuple[int, int], range2: Tuple[int, int]) -> float:
        """Calculate overlap percentage between two frame ranges"""
        
        start1, end1 = range1
        start2, end2 = range2
        
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        
        if overlap_start > overlap_end:
            return 0.0
        
        overlap_length = overlap_end - overlap_start + 1
        total_length = max(end1 - start1 + 1, end2 - start2 + 1)
        
        return overlap_length / total_length
    
    def _map_comment_to_expected(self, comment: str) -> str:
        """Map annotation comment to expected behavior"""
        
        comment_lower = comment.lower().strip()
        
        # Exact matches first
        if comment_lower in self.ground_truth_mapping:
            return self.ground_truth_mapping[comment_lower]
        
        # Partial matching with priority order
        for key_phrase, behavior in self.ground_truth_mapping.items():
            if key_phrase in comment_lower:
                return behavior
        
        return 'other'
    
    def test_ensemble_system(self, test_cases: List[Dict]) -> List[Dict]:
        """Test the complete ensemble system"""
        
        print(f"🧪 Testing Ensemble System on {len(test_cases)} cases...")
        
        results = []
        
        for i, test_case in enumerate(test_cases):
            if i % 10 == 0:
                print(f"   Progress: {i}/{len(test_cases)} ({100*i/len(test_cases):.1f}%)")
            
            try:
                features_dict = test_case['features']
                expected_behavior = test_case['expected_behavior']
                
                # Create features array for ensemble
                feature_values = [
                    features_dict.get('jerk_rms', 0),
                    features_dict.get('velocity_mean', 0),
                    features_dict.get('velocity_std', 0),
                    features_dict.get('velocity_min', 0),
                    features_dict.get('velocity_max', 0),
                    features_dict.get('acceleration_mean', 0),
                    features_dict.get('acceleration_std', 0),
                    features_dict.get('acceleration_min', 0),
                    features_dict.get('acceleration_max', 0),
                    features_dict.get('jerk_mean', 0),
                    features_dict.get('jerk_std', 0),
                    features_dict.get('max_deceleration', 0),
                    features_dict.get('deceleration_events', 0),
                    2,  # placeholder
                    features_dict.get('acceleration_reversals', 0),
                    features_dict.get('motion_smoothness', 0),
                    features_dict.get('jerk_per_second', 0),
                    features_dict.get('accel_changes_per_second', 0),
                    features_dict.get('duration_s', 0),
                    -1.35  # placeholder
                ]
                
                features_array = np.array(feature_values)
                
                # Test ensemble system
                start_time = time.time()
                
                if self.ensemble_system:
                    # Use full ensemble system
                    ensemble_result = self.ensemble_system.analyze_segment_enhanced(
                        features_array, features_dict
                    )
                    
                    inference_time = time.time() - start_time
                    
                    result = {
                        'test_case': test_case,
                        'ensemble_result': ensemble_result,
                        'inference_time': inference_time * 1000,  # ms
                        'test_type': 'ensemble'
                    }
                else:
                    # Fallback: Test individual models
                    individual_predictions = self._predict_individual_models(features_dict, features_array)
                    
                    inference_time = time.time() - start_time
                    
                    result = {
                        'test_case': test_case,
                        'individual_predictions': individual_predictions,
                        'inference_time': inference_time * 1000,  # ms
                        'test_type': 'individual'
                    }
                
                results.append(result)
                
            except Exception as e:
                print(f"   ❌ Error in test case {i}: {e}")
        
        print(f"✅ Completed {len(results)} test cases")
        return results
    
    def _predict_individual_models(self, features_dict: Dict, features_array: np.ndarray) -> Dict:
        """Get predictions from individual models (fallback)"""
        
        predictions = {}
        
        # Advanced ML
        if self.advanced_model:
            try:
                pipeline = self.advanced_model['pipeline']
                label_encoder = self.advanced_model['label_encoder']
                
                if features_array.ndim == 1:
                    features_array = features_array.reshape(1, -1)
                
                feature_names = self.advanced_model['feature_names']
                if features_array.shape[1] < len(feature_names):
                    padding = np.zeros((features_array.shape[0], len(feature_names) - features_array.shape[1]))
                    features_array = np.hstack([features_array, padding])
                elif features_array.shape[1] > len(feature_names):
                    features_array = features_array[:, :len(feature_names)]
                
                features_array = np.nan_to_num(features_array, nan=0.0, posinf=999, neginf=-999)
                
                probabilities = pipeline.predict_proba(features_array)[0]
                predicted_idx = np.argmax(probabilities)
                predicted_class = label_encoder.classes_[predicted_idx]
                confidence = probabilities[predicted_idx]
                
                predictions['advanced_ml'] = {
                    'predicted_behavior': predicted_class,
                    'confidence': float(confidence)
                }
            except Exception as e:
                predictions['advanced_ml'] = {"error": str(e)}
        
        # Add other individual models here if needed...
        
        return predictions
    
    def analyze_ensemble_performance(self, results: List[Dict]) -> Dict:
        """Analyze ensemble system performance"""
        
        print("\n📊 ANALYZING ENSEMBLE PERFORMANCE...")
        
        if not results:
            return {}
        
        test_type = results[0].get('test_type', 'unknown')
        
        if test_type == 'ensemble':
            return self._analyze_ensemble_results(results)
        else:
            return self._analyze_individual_results(results)
    
    def _analyze_ensemble_results(self, results: List[Dict]) -> Dict:
        """Analyze ensemble system results"""
        
        metrics = {
            'total_cases': len(results),
            'binary_filter_performance': {'interesting_detected': 0, 'total_interesting': 0},
            'ensemble_accuracy': {'correct': 0, 'total': 0},
            'risk_assessment': defaultdict(int),
            'confidence_distribution': [],
            'strategy_usage': defaultdict(int),
            'inference_times': []
        }
        
        # Analyze each result
        for result in results:
            expected = result['test_case']['expected_behavior']
            ensemble_result = result['ensemble_result']
            
            # Binary filter analysis
            binary_filter = ensemble_result.get('binary_filter', {})
            is_interesting_expected = expected != 'other'
            is_interesting_detected = binary_filter.get('is_interesting', False)
            
            if is_interesting_expected:
                metrics['binary_filter_performance']['total_interesting'] += 1
                if is_interesting_detected:
                    metrics['binary_filter_performance']['interesting_detected'] += 1
            
            # Ensemble prediction analysis
            ensemble_analysis = ensemble_result.get('ensemble_analysis', {})
            predicted = ensemble_analysis.get('ensemble_prediction', 'unknown')
            confidence = ensemble_analysis.get('confidence', 0.0)
            strategy = ensemble_analysis.get('strategy', 'unknown')
            
            metrics['ensemble_accuracy']['total'] += 1
            
            # Check if prediction is correct
            if self._is_prediction_correct(predicted, expected):
                metrics['ensemble_accuracy']['correct'] += 1
            
            # Collect other metrics
            metrics['confidence_distribution'].append(confidence)
            metrics['strategy_usage'][strategy] += 1
            
            # Risk assessment
            risk_level = ensemble_result.get('risk_assessment', {}).get('risk_level', 'UNKNOWN')
            metrics['risk_assessment'][risk_level] += 1
            
            # Inference time
            metrics['inference_times'].append(result.get('inference_time', 0))
        
        # Calculate final metrics
        binary_performance = metrics['binary_filter_performance']
        if binary_performance['total_interesting'] > 0:
            binary_recall = binary_performance['interesting_detected'] / binary_performance['total_interesting']
        else:
            binary_recall = 0.0
        
        ensemble_accuracy = metrics['ensemble_accuracy']['correct'] / metrics['ensemble_accuracy']['total']
        avg_confidence = np.mean(metrics['confidence_distribution'])
        avg_inference_time = np.mean(metrics['inference_times'])
        
        return {
            'test_type': 'ensemble',
            'total_cases': metrics['total_cases'],
            'binary_filter_recall': binary_recall,
            'ensemble_accuracy': ensemble_accuracy,
            'average_confidence': avg_confidence,
            'average_inference_time_ms': avg_inference_time,
            'strategy_distribution': dict(metrics['strategy_usage']),
            'risk_level_distribution': dict(metrics['risk_assessment']),
            'detailed_metrics': metrics
        }
    
    def _analyze_individual_results(self, results: List[Dict]) -> Dict:
        """Analyze individual model results (fallback)"""
        
        # Implementation for individual model analysis
        return {
            'test_type': 'individual',
            'total_cases': len(results),
            'note': 'Individual model analysis - ensemble system not available'
        }
    
    def _is_prediction_correct(self, predicted: str, expected: str) -> bool:
        """Check if prediction is correct with flexible matching"""
        
        if predicted == expected:
            return True
        
        # Group similar behaviors for evaluation
        similar_groups = [
            ['nearmiss', 'overshoot', 'oversteering'],  # Dangerous behaviors
            ['pedestrian', 'bicycle'],  # External entities
            ['hesitation', 'stop_behavior'],  # Stop/slow behaviors
            ['lane_change', 'turning'],  # Lane changes
            ['obstacles', 'vehicle_interaction'],  # Obstacle avoidance
            ['road_conditions', 'other']  # Environmental
        ]
        
        for group in similar_groups:
            if predicted in group and expected in group:
                return True
        
        return False
    
    def print_ensemble_report(self, analysis: Dict):
        """Print comprehensive ensemble performance report"""
        
        if not analysis:
            print("❌ No analysis results to display")
            return
        
        print(f"\n🎯 ENSEMBLE PERFORMANCE REPORT")
        print("=" * 60)
        
        if analysis['test_type'] == 'ensemble':
            print(f"Total Test Cases: {analysis['total_cases']}")
            print(f"Binary Filter Recall: {analysis['binary_filter_recall']:.1%}")
            print(f"Ensemble Accuracy: {analysis['ensemble_accuracy']:.1%}")
            print(f"Average Confidence: {analysis['average_confidence']:.3f}")
            print(f"Average Inference Time: {analysis['average_inference_time_ms']:.1f}ms")
            
            print(f"\n📊 Strategy Usage:")
            for strategy, count in analysis['strategy_distribution'].items():
                percentage = count / analysis['total_cases'] * 100
                print(f"   {strategy}: {count} ({percentage:.1f}%)")
            
            print(f"\n⚠️ Risk Level Distribution:")
            for risk_level, count in analysis['risk_level_distribution'].items():
                percentage = count / analysis['total_cases'] * 100
                print(f"   {risk_level}: {count} ({percentage:.1f}%)")
            
        else:
            print(f"Fallback mode - {analysis.get('note', 'Unknown issue')}")
    
    def run_comprehensive_test(self) -> Dict:
        """Run comprehensive ensemble performance test"""
        
        print("🧪 RUNNING COMPREHENSIVE ENSEMBLE PERFORMANCE TEST")
        print("=" * 70)
        
        # Load ensemble system
        ensemble_loaded = self.load_ensemble_system()
        
        # Load test data
        test_cases = self.load_test_data()
        if not test_cases:
            return {}
        
        # Run tests
        results = self.test_ensemble_system(test_cases)
        
        # Analyze results
        analysis = self.analyze_ensemble_performance(results)
        
        return analysis

def main():
    """Run the comprehensive ensemble performance test"""
    
    tester = EnhancedEnsemblePerformanceTester()
    analysis = tester.run_comprehensive_test()
    
    if analysis:
        tester.print_ensemble_report(analysis)
        
        # Save results
        with open("ensemble_performance_results.json", 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: ensemble_performance_results.json")
    
    print(f"\n🎉 ENSEMBLE PERFORMANCE TESTING COMPLETE!")

if __name__ == "__main__":
    main()