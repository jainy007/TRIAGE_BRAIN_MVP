#!/usr/bin/env python3
"""
Model Performance Tester - Tests all models with correct class mapping
"""

import json
import numpy as np
import pandas as pd
import time
import torch
from typing import Dict, List
from collections import Counter, defaultdict
import sys
import os

sys.path.append('src/triage_brain')

class ModelPerformanceTester:
    """Performance tester for all 4 models with correct class mapping"""
    
    def __init__(self):
        # Map annotation comments to ACTUAL model classes (not our wishful thinking)
        self.ground_truth_mapping = {
            # Stop signs -> overshoot
            'stop sign overshoot': 'overshoot',
            'stop overshoot': 'overshoot',
            'stop sine overshoot': 'overshoot',
            
            # Near misses -> nearmiss  
            'near miss': 'nearmiss',
            'risky launch - near miss': 'nearmiss',
            'narrow near miss': 'nearmiss',
            'leading car backed up -near miss': 'nearmiss',
            'oversteering near miss': 'nearmiss',
            'oversteer near miss': 'nearmiss',
            'left turn overshoot nearmiss': 'nearmiss',
            
            # Pedestrians -> pedestrian
            'man crossing': 'pedestrian',
            'pedestrian crossing': 'pedestrian',
            'pedestrian running': 'pedestrian',
            'women on ego lane': 'pedestrian',
            'man in horizon': 'pedestrian',
            'man at curb': 'pedestrian',
            
            # Bicycles -> bicycle
            'bicycle': 'bicycle',
            'bicycle crossed': 'bicycle',
            'letting bicycle pass': 'bicycle',
            
            # Traffic control -> traffic_control
            'stop sign': 'traffic_control',
            'traffic light': 'traffic_control',
            'red light': 'traffic_control',
            'stop sign crawl': 'traffic_control',
            
            # Steering issues -> oversteering
            'oversteering': 'oversteering',
            'turn oversteering': 'oversteering',
            'perfect right steer': 'oversteering',  # Could be other, but steering related
            
            # Hesitation behaviors -> hesitation
            'hesitation': 'hesitation',
            'unnecessary halt': 'hesitation',
            'nervous stop': 'hesitation',
            'protected left hesitation': 'hesitation',
            'unnecessary hesitation': 'hesitation',
            'right turn hesitation': 'hesitation',
            
            # Vehicle interactions -> vehicle_interaction
            'leading vehicle': 'vehicle_interaction',
            'stopped car': 'vehicle_interaction',
            'bus coming': 'vehicle_interaction',
            'misalligned leading vehicle': 'vehicle_interaction',
            'leading car backed up': 'vehicle_interaction',
            'police leading vehicle': 'vehicle_interaction',
            'vehicles overtaking': 'vehicle_interaction',
            'car parallel parking': 'vehicle_interaction',
            'car on road': 'vehicle_interaction',
            'vehicle parked': 'vehicle_interaction',
            
            # Road conditions -> road_conditions
            'road color': 'road_conditions',
            'change in road color': 'road_conditions',
            'dark to bright occlusion': 'road_conditions',
            'blending horizon color': 'road_conditions',
            'toll gate': 'road_conditions',
            
            # Everything else -> other
            'unknown object ahead': 'other',
            'traffic cones': 'other',  # Models don't have specific cone class
            'obstacle': 'other',
            'confusion': 'other',
            'bump ignored': 'other'
        }
        
        # Initialize models
        self.advanced_model = None
        self.simple_model = None
        self.rule_model = None
        self.dl_model = None
        self.dl_behavior_classes = None
        
    def load_models(self) -> Dict[str, bool]:
        """Load all models individually"""
        
        print("🔧 Loading models for performance testing...")
        
        loading_status = {
            'advanced_ml': False,
            'simple_ml': False,
            'rule_based': False,
            'deep_learning': False
        }
        
        # Advanced ML
        try:
            import joblib
            self.advanced_model = joblib.load("assets/models/advanced_triage_brain.pkl")
            loading_status['advanced_ml'] = True
            print("✅ Advanced ML model loaded")
            print(f"   Classes: {list(self.advanced_model['label_encoder'].classes_)}")
        except Exception as e:
            print(f"❌ Advanced ML failed: {e}")
        
        # Simple ML
        try:
            import joblib
            self.simple_model = joblib.load("assets/models/simple_ml_triage.pkl")
            loading_status['simple_ml'] = True
            print("✅ Simple ML model loaded")
            print(f"   Classes: {list(self.simple_model['label_encoder'].classes_)}")
        except Exception as e:
            print(f"❌ Simple ML failed: {e}")
        
        # Rule-based
        try:
            with open("practical_triage_brain.json", 'r') as f:
                self.rule_model = json.load(f)
            loading_status['rule_based'] = True
            print("✅ Rule-based model loaded")
            print(f"   Rules: {list(self.rule_model['behavioral_rules'].keys())}")
        except Exception as e:
            print(f"❌ Rule-based failed: {e}")
        
        # Deep Learning (load dynamically)
        try:
            self.dl_model, self.dl_behavior_classes = self._load_dl_model_dynamic()
            loading_status['deep_learning'] = True
            print("✅ Deep Learning model loaded")
            print(f"   Classes: {self.dl_behavior_classes}")
        except Exception as e:
            print(f"❌ Deep Learning failed: {e}")
        
        return loading_status
    
    def _load_dl_model_dynamic(self):
        """Load DL model with dynamic size detection"""
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load state dict to detect dimensions
        state_dict = torch.load("best_fixed_model.pth", map_location=device)
        
        # Detect dimensions
        classifier_weight_shape = state_dict['classifier.3.weight'].shape
        num_behaviors = classifier_weight_shape[0]
        
        conv_weight_shape = state_dict['temporal_conv.0.weight'].shape
        num_features = conv_weight_shape[1]
        
        print(f"🔍 DL Model: {num_features} features, {num_behaviors} behaviors")
        
        # Create model
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
        
        # Map DL classes to standard classes (assume same as other models for now)
        standard_classes = ['bicycle', 'hesitation', 'nearmiss', 'other', 'overshoot', 
                          'oversteering', 'pedestrian', 'road_conditions', 'traffic_control', 
                          'vehicle_interaction']
        
        # Use first N classes or create generic names
        if num_behaviors <= len(standard_classes):
            behavior_classes = standard_classes[:num_behaviors]
        else:
            behavior_classes = [f"behavior_{i}" for i in range(num_behaviors)]
        
        return model, behavior_classes
    
    def load_test_data(self) -> List[Dict]:
        """Load test data with flexible paths"""
        
        print("📁 Loading test data...")
        
        # Find annotation file
        annotation_paths = [
            "annotated_clips.jsonl",
            "/home/jainy007/PEM/triage_brain/annotated_clips.jsonl",
            "/home/jainy007/PEM/mvp3_0/annotated_clips.jsonl"
        ]
        
        annotations = {}
        for path in annotation_paths:
            if os.path.exists(path):
                print(f"📁 Found annotations: {path}")
                with open(path, 'r') as f:
                    for line in f:
                        clip_data = json.loads(line)
                        annotations[clip_data['clip']] = clip_data['segments']
                break
        else:
            print("❌ annotated_clips.jsonl not found")
            return []
        
        # Find feature file
        feature_paths = [
            "assets/data/feature_vectors.jsonl",
            "/home/jainy007/PEM/triage_brain/assets/data/feature_vectors.jsonl"
        ]
        
        feature_vectors = []
        for path in feature_paths:
            if os.path.exists(path):
                print(f"📁 Found features: {path}")
                with open(path, 'r') as f:
                    for line in f:
                        feature_vectors.append(json.loads(line))
                break
        else:
            print("❌ feature_vectors.jsonl not found")
            return []
        
        # Create test cases
        test_cases = []
        
        for feature_vector in feature_vectors:
            clip_name = feature_vector['clip_name']
            start_frame = feature_vector['start_frame']
            end_frame = feature_vector['end_frame']
            
            if clip_name in annotations:
                for annotation in annotations[clip_name]:
                    if (abs(annotation['start'] - start_frame) <= 5 and
                        abs(annotation['end'] - end_frame) <= 5):
                        
                        comment = annotation['comment']
                        expected_behavior = self._map_comment_to_expected(comment)
                        
                        test_case = {
                            'clip_name': clip_name,
                            'start_frame': start_frame,
                            'end_frame': end_frame,
                            'comment': comment,
                            'expected_behavior': expected_behavior,
                            'features': feature_vector
                        }
                        
                        test_cases.append(test_case)
                        break
        
        print(f"✅ Created {len(test_cases)} test cases")
        return test_cases
    
    def _map_comment_to_expected(self, comment: str) -> str:
        """Map annotation comment to expected model class"""
        
        comment_lower = comment.lower().strip()
        
        # Find best match
        for key_phrase, behavior in self.ground_truth_mapping.items():
            if key_phrase in comment_lower:
                return behavior
        
        # Default fallback
        return 'other'
    
    def predict_all_models(self, features_dict: Dict, features_array: np.ndarray) -> Dict:
        """Get predictions from all available models"""
        
        predictions = {}
        inference_times = {}
        
        # Advanced ML
        if self.advanced_model:
            start_time = time.time()
            try:
                pipeline = self.advanced_model['pipeline']
                label_encoder = self.advanced_model['label_encoder']
                feature_names = self.advanced_model['feature_names']
                
                # Create feature array
                feature_values = [features_dict.get(name, 0.0) for name in feature_names]
                X = np.array(feature_values).reshape(1, -1)
                X = np.nan_to_num(X, nan=0.0, posinf=999, neginf=-999)
                
                probabilities = pipeline.predict_proba(X)[0]
                predicted_idx = np.argmax(probabilities)
                predicted_class = label_encoder.classes_[predicted_idx]
                confidence = probabilities[predicted_idx]
                
                predictions['advanced_ml'] = {
                    'predicted_behavior': predicted_class,
                    'confidence': float(confidence)
                }
            except Exception as e:
                predictions['advanced_ml'] = {"error": str(e)}
            
            inference_times['advanced_ml'] = (time.time() - start_time) * 1000
        
        # Simple ML
        if self.simple_model:
            start_time = time.time()
            try:
                scaler = self.simple_model['scaler']
                classifier = self.simple_model['classifier']
                label_encoder = self.simple_model['label_encoder']
                feature_names = self.simple_model['feature_names']
                
                feature_values = [features_dict.get(name, 0.0) for name in feature_names]
                X = np.array(feature_values).reshape(1, -1)
                X_scaled = scaler.transform(X)
                
                probabilities = classifier.predict_proba(X_scaled)[0]
                predicted_idx = np.argmax(probabilities)
                predicted_class = label_encoder.classes_[predicted_idx]
                confidence = probabilities[predicted_idx]
                
                predictions['simple_ml'] = {
                    'predicted_behavior': predicted_class,
                    'confidence': float(confidence)
                }
            except Exception as e:
                predictions['simple_ml'] = {"error": str(e)}
            
            inference_times['simple_ml'] = (time.time() - start_time) * 1000
        
        # Rule-based
        if self.rule_model:
            start_time = time.time()
            try:
                behavioral_rules = self.rule_model['behavioral_rules']
                sample = pd.Series(features_dict)
                rule_predictions = {}
                
                for behavior, rule in behavioral_rules.items():
                    confidence = 0.0
                    conditions = rule.get('conditions', {})
                    
                    # Duration check
                    if 'duration' in conditions and 'duration_s' in sample:
                        duration = sample['duration_s']
                        duration_range = conditions['duration']
                        if duration_range['min'] <= duration <= duration_range['max']:
                            confidence += 0.3
                    
                    # Jerk check
                    if 'jerk_intensity' in conditions and 'jerk_rms' in sample:
                        jerk = sample['jerk_rms']
                        jerk_info = conditions['jerk_intensity']
                        if abs(jerk - jerk_info['threshold']) < jerk_info.get('std', 10) * 1.5:
                            confidence += 0.25
                    
                    rule_predictions[behavior] = confidence
                
                if rule_predictions:
                    best_behavior = max(rule_predictions.items(), key=lambda x: x[1])
                    predictions['rule_based'] = {
                        'predicted_behavior': best_behavior[0],
                        'confidence': best_behavior[1]
                    }
                else:
                    predictions['rule_based'] = {
                        'predicted_behavior': 'other',
                        'confidence': 0.0
                    }
            except Exception as e:
                predictions['rule_based'] = {"error": str(e)}
            
            inference_times['rule_based'] = (time.time() - start_time) * 1000
        
        # Deep Learning
        if self.dl_model:
            start_time = time.time()
            try:
                # Create feature sequence
                dl_features = [
                    'velocity_mean', 'velocity_std', 'velocity_min', 'velocity_max', 'velocity_range',
                    'acceleration_mean', 'acceleration_std', 'acceleration_min', 'acceleration_max', 'acceleration_range',
                    'jerk_mean', 'jerk_std', 'jerk_min', 'jerk_max', 'jerk_rms',
                    'max_deceleration', 'motion_smoothness', 'acceleration_reversals',
                    'velocity_zero_crossings', 'deceleration_events', 'jerk_per_second',
                    'accel_changes_per_second', 'distance_traveled', 'duration_s'
                ]
                
                feature_values = []
                for feature_name in dl_features:
                    value = features_dict.get(feature_name, 0.0)
                    if np.isnan(value) or np.isinf(value):
                        value = 0.0
                    feature_values.append(float(value))
                
                # Create sequence (repeat frame 8 times)
                sequence_length = 8
                feature_matrix = [feature_values] * sequence_length
                
                device = next(self.dl_model.parameters()).device
                features_tensor = torch.FloatTensor(feature_matrix).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    behavior_logits, risk_score = self.dl_model(features_tensor)
                    
                    probs = torch.nn.functional.softmax(behavior_logits, dim=-1)
                    predicted_idx = torch.argmax(probs).item()
                    confidence = torch.max(probs).item()
                    
                    if predicted_idx < len(self.dl_behavior_classes):
                        predicted_behavior = self.dl_behavior_classes[predicted_idx]
                    else:
                        predicted_behavior = 'other'
                    
                    predictions['deep_learning'] = {
                        'predicted_behavior': predicted_behavior,
                        'confidence': float(confidence),
                        'risk_score': float(risk_score.item())
                    }
            except Exception as e:
                predictions['deep_learning'] = {"error": str(e)}
            
            inference_times['deep_learning'] = (time.time() - start_time) * 1000
        
        return predictions, inference_times
    
    def run_comprehensive_test(self) -> Dict:
        """Run comprehensive performance test"""
        
        print("🧪 RUNNING COMPREHENSIVE MODEL PERFORMANCE TEST")
        print("=" * 65)
        
        # Load models
        loading_status = self.load_models()
        loaded_models = [model for model, status in loading_status.items() if status]
        
        if not loaded_models:
            print("❌ No models loaded")
            return {}
        
        print(f"✅ Testing {len(loaded_models)} models: {', '.join(loaded_models)}")
        
        # Load test data
        test_cases = self.load_test_data()
        if not test_cases:
            return {}
        
        print(f"🎯 Running tests on {len(test_cases)} cases...")
        
        # Initialize metrics
        metrics = {
            model: {
                'correct': 0,
                'total': 0,
                'confidences': [],
                'inference_times': [],
                'predictions': [],
                'errors': 0
            }
            for model in loaded_models
        }
        
        # Run tests
        for i, test_case in enumerate(test_cases):
            if i % 10 == 0:
                print(f"   Progress: {i}/{len(test_cases)} ({100*i/len(test_cases):.1f}%)")
            
            try:
                features_dict = test_case['features']
                expected = test_case['expected_behavior']
                
                # Create features array for advanced ML
                feature_values = [
                    features_dict.get('jerk_rms', 0),
                    features_dict.get('velocity_mean', 0),
                    features_dict.get('velocity_std', 0),
                    features_dict.get('max_deceleration', 0),
                    features_dict.get('duration_s', 0),
                    features_dict.get('motion_smoothness', 0),
                    features_dict.get('acceleration_reversals', 0),
                    features_dict.get('deceleration_events', 0),
                    features_dict.get('jerk_per_second', 0),
                    features_dict.get('accel_changes_per_second', 0)
                ]
                
                features_array = np.array(feature_values)
                
                # Get predictions
                predictions, inference_times = self.predict_all_models(features_dict, features_array)
                
                # Analyze each model
                for model_name in loaded_models:
                    if model_name in predictions:
                        pred = predictions[model_name]
                        
                        if 'error' in pred:
                            metrics[model_name]['errors'] += 1
                            continue
                        
                        predicted = pred.get('predicted_behavior', 'other')
                        confidence = pred.get('confidence', 0.0)
                        
                        metrics[model_name]['total'] += 1
                        metrics[model_name]['confidences'].append(confidence)
                        metrics[model_name]['predictions'].append(predicted)
                        
                        if model_name in inference_times:
                            metrics[model_name]['inference_times'].append(inference_times[model_name])
                        
                        # Check accuracy
                        if predicted == expected:
                            metrics[model_name]['correct'] += 1
                
            except Exception as e:
                print(f"   ❌ Error in test case {i}: {e}")
        
        print(f"✅ Completed testing on {len(test_cases)} cases")
        
        # Calculate final metrics
        summary = {}
        for model_name, model_metrics in metrics.items():
            total = model_metrics['total']
            if total > 0:
                accuracy = model_metrics['correct'] / total
                avg_confidence = np.mean(model_metrics['confidences']) if model_metrics['confidences'] else 0
                avg_inference_time = np.mean(model_metrics['inference_times']) if model_metrics['inference_times'] else 0
                
                summary[model_name] = {
                    'accuracy': accuracy,
                    'correct_predictions': model_metrics['correct'],
                    'total_predictions': total,
                    'average_confidence': avg_confidence,
                    'average_inference_time_ms': avg_inference_time,
                    'error_count': model_metrics['errors'],
                    'prediction_distribution': Counter(model_metrics['predictions'])
                }
        
        # Calculate optimal weights
        optimal_weights = self._calculate_optimal_weights(summary)
        
        return {
            'summary': summary,
            'optimal_weights': optimal_weights,
            'test_cases_count': len(test_cases),
            'loaded_models': loaded_models
        }
    
    def _calculate_optimal_weights(self, summary: Dict) -> Dict:
        """Calculate optimal weights based on performance"""
        
        models = ['advanced_ml', 'simple_ml', 'rule_based', 'deep_learning']
        scores = {}
        
        for model in models:
            if model not in summary:
                scores[model] = 0.0
                continue
            
            metrics = summary[model]
            
            # Composite score
            accuracy_score = metrics['accuracy'] * 0.7
            confidence_score = metrics['average_confidence'] * 0.2
            speed_score = (1000.0 / max(metrics['average_inference_time_ms'], 1)) * 0.1
            
            scores[model] = accuracy_score + confidence_score + speed_score
        
        # Normalize
        total_score = sum(scores.values())
        if total_score > 0:
            optimal_weights = {model: score / total_score for model, score in scores.items()}
        else:
            optimal_weights = {model: 0.25 for model in models}
        
        return optimal_weights
    
    def generate_report(self, analysis: Dict):
        """Generate performance report"""
        
        if not analysis:
            print("❌ No analysis data to report")
            return
        
        summary = analysis['summary']
        optimal_weights = analysis['optimal_weights']
        
        print(f"\n🎯 MODEL PERFORMANCE SUMMARY:")
        print(f"{'Model':<15} {'Accuracy':<10} {'Confidence':<11} {'Speed(ms)':<10} {'Errors'}")
        print("-" * 65)
        
        for model, metrics in summary.items():
            accuracy = f"{metrics['accuracy']:.1%}"
            confidence = f"{metrics['average_confidence']:.3f}"
            speed = f"{metrics['average_inference_time_ms']:.1f}"
            errors = str(metrics['error_count'])
            
            print(f"{model:<15} {accuracy:<10} {confidence:<11} {speed:<10} {errors}")
        
        print(f"\n⚖️ OPTIMAL WEIGHTS:")
        current_weights = {'deep_learning': 0.4, 'advanced_ml': 0.3, 'simple_ml': 0.2, 'rule_based': 0.1}
        
        print(f"{'Model':<15} {'Current':<10} {'Optimal':<10} {'Change'}")
        print("-" * 50)
        
        for model in ['deep_learning', 'advanced_ml', 'simple_ml', 'rule_based']:
            current = current_weights.get(model, 0.0)
            optimal = optimal_weights.get(model, 0.0)
            change = optimal - current
            
            print(f"{model:<15} {current:<10.1%} {optimal:<10.1%} {change:+.2f}")
        
        # Show prediction distributions
        print(f"\n📊 PREDICTION DISTRIBUTIONS:")
        for model, metrics in summary.items():
            print(f"\n{model.upper()}:")
            dist = metrics['prediction_distribution']
            for behavior, count in dist.most_common(5):
                print(f"   {behavior:<20}: {count}")
        
        # Save detailed report
        with open("model_performance_report.json", 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print(f"\n💾 Detailed report saved to: model_performance_report.json")

def main():
    """Run performance testing"""
    
    tester = ModelPerformanceTester()
    analysis = tester.run_comprehensive_test()
    
    if analysis:
        tester.generate_report(analysis)
        print(f"\n🎉 PERFORMANCE TESTING COMPLETE!")
    else:
        print(f"❌ Performance testing failed")

if __name__ == "__main__":
    main()