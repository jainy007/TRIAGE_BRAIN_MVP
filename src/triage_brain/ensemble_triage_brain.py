#!/usr/bin/env python3
"""
Enhanced version of your existing ensemble system with binary filtering
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional
import joblib
from pathlib import Path

class EnhancedTriageEnsemble:
    """
    Enhanced version of your existing 4-model ensemble with binary filtering
    """
    
    def __init__(self):
        # Load your existing models
        self.advanced_model = None
        self.simple_model = None  
        self.rule_model = None
        self.dl_model = None
        
        # New: Binary filter stage
        self.binary_filter = None
        
        # Enhanced weights with binary filtering
        self.model_weights = {
            'deep_learning': 0.35,
            'advanced_ml': 0.35,
            'simple_ml': 0.20,
            'rule_based': 0.10
        }
        
        # Binary filter thresholds
        self.binary_thresholds = {
            'motion_intensity': 35.0,    # jerk_rms threshold
            'deceleration': 10.0,        # braking intensity
            'duration': 2.0,             # minimum duration for interest
            'ensemble_confidence': 0.3    # minimum ensemble confidence
        }
        
    def load_existing_models(self, model_paths: Dict[str, str]):
        """Load your existing trained models"""
        
        try:
            # Load Advanced ML model
            if Path(model_paths.get('advanced', '')).exists():
                self.advanced_model = joblib.load(model_paths['advanced'])
                print("✅ Advanced ML model loaded")
            
            # Load Simple ML model  
            if Path(model_paths.get('simple', '')).exists():
                self.simple_model = joblib.load(model_paths['simple'])
                print("✅ Simple ML model loaded")
            
            # Load Rule-based model
            if Path(model_paths.get('rule', '')).exists():
                with open(model_paths['rule'], 'r') as f:
                    self.rule_model = json.load(f)
                print("✅ Rule-based model loaded")
            
            # Load DL model (your existing best_fixed_model.pth)
            if Path(model_paths.get('dl', '')).exists():
                self._load_dl_model(model_paths['dl'])
                print("✅ Deep Learning model loaded")
                
        except Exception as e:
            print(f"⚠️ Error loading models: {e}")
    
    def _load_dl_model(self, dl_path: str):
        """Load your existing DL model"""
        try:
            # Use your existing DL model architecture
            class DynamicTemporalNet(torch.nn.Module):
                def __init__(self, input_features=24, num_behaviors=13, hidden_dim=48):
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
            
            # Load your existing model
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            state_dict = torch.load(dl_path, map_location=device)
            
            # Determine model dimensions from state dict
            conv_weight_shape = state_dict['temporal_conv.0.weight'].shape
            classifier_weight_shape = state_dict['classifier.3.weight'].shape
            
            input_features = conv_weight_shape[1]
            num_behaviors = classifier_weight_shape[0]
            
            self.dl_model = DynamicTemporalNet(input_features, num_behaviors)
            self.dl_model.load_state_dict(state_dict)
            self.dl_model.to(device)
            self.dl_model.eval()
            
        except Exception as e:
            print(f"❌ DL model loading failed: {e}")
            self.dl_model = None
    
    def binary_interest_filter(self, features_dict: Dict) -> Dict:
        """
        Binary filter to determine if segment is 'interesting' 
        Uses your proven feature patterns + simple thresholds
        """
        
        interest_score = 0.0
        reasons = []
        
        # Motion intensity check
        jerk_rms = features_dict.get('jerk_rms', 0)
        if jerk_rms > self.binary_thresholds['motion_intensity']:
            interest_score += 0.3
            reasons.append(f"High motion intensity ({jerk_rms:.1f})")
        
        # Deceleration intensity  
        max_decel = abs(features_dict.get('max_deceleration', 0))
        if max_decel > self.binary_thresholds['deceleration']:
            interest_score += 0.25
            reasons.append(f"Hard braking ({max_decel:.1f})")
        
        # Duration filter (too brief segments are usually noise)
        duration = features_dict.get('duration_s', 0)
        if duration > self.binary_thresholds['duration']:
            interest_score += 0.2
        
        # Motion complexity
        accel_reversals = features_dict.get('acceleration_reversals', 0)
        if duration > 0 and accel_reversals / duration > 8:  # High indecision rate
            interest_score += 0.15
            reasons.append("High indecision rate")
        
        # Motion smoothness (low = rough/interesting)
        smoothness = features_dict.get('motion_smoothness', 1.0)
        if smoothness < 0.05:
            interest_score += 0.1
            reasons.append("Rough motion")
        
        is_interesting = interest_score >= 0.4  # Threshold for "interesting"
        
        return {
            'is_interesting': is_interesting,
            'interest_score': interest_score,
            'reasons': reasons,
            'motion_stats': {
                'jerk_rms': jerk_rms,
                'max_deceleration': max_decel,
                'duration_s': duration,
                'smoothness': smoothness
            }
        }
    
    def predict_with_existing_models(self, features_array: np.ndarray, features_dict: Dict) -> Dict:
        """
        Use your existing 4-model ensemble prediction logic
        (Adapting from your EnhancedEnsembleTriageBrain)
        """
        
        predictions = {}
        
        # Advanced ML prediction
        if self.advanced_model:
            try:
                pipeline = self.advanced_model['pipeline']
                label_encoder = self.advanced_model['label_encoder']
                
                if features_array.ndim == 1:
                    features_array = features_array.reshape(1, -1)
                
                # Handle feature dimension mismatch
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
                    'confidence': float(confidence),
                    'model': 'advanced_ml'
                }
            except Exception as e:
                predictions['advanced_ml'] = {'error': f'Advanced ML failed: {e}'}
        
        # Simple ML prediction
        if self.simple_model:
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
                    'confidence': float(confidence),
                    'model': 'simple_ml'
                }
            except Exception as e:
                predictions['simple_ml'] = {'error': f'Simple ML failed: {e}'}
        
        # Rule-based prediction
        if self.rule_model:
            try:
                behavioral_rules = self.rule_model['behavioral_rules']
                sample = pd.Series(features_dict)
                rule_predictions = {}
                
                for behavior, rule in behavioral_rules.items():
                    confidence = 0.0
                    conditions = rule['conditions']
                    
                    # Duration check
                    if 'duration' in conditions and 'duration_s' in sample:
                        duration = sample['duration_s']
                        duration_range = conditions['duration']
                        if duration_range['min'] <= duration <= duration_range['max']:
                            confidence += 0.3
                    
                    # Jerk intensity check  
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
                        'confidence': best_behavior[1],
                        'model': 'rule_based'
                    }
                else:
                    predictions['rule_based'] = {
                        'predicted_behavior': 'unknown',
                        'confidence': 0.0,
                        'model': 'rule_based'
                    }
            except Exception as e:
                predictions['rule_based'] = {'error': f'Rule-based failed: {e}'}
        
        # Deep Learning prediction
        if self.dl_model:
            try:
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
                
                # Create sequence (repeat 8 times for temporal model)
                sequence = [feature_values] * 8
                
                device = next(self.dl_model.parameters()).device
                features_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    behavior_logits, risk_score = self.dl_model(features_tensor)
                    probs = F.softmax(behavior_logits, dim=-1)
                    predicted_idx = torch.argmax(probs).item()
                    confidence = torch.max(probs).item()
                    
                    # Map to behavior (you'll need to define this mapping based on your training)
                    behavior_classes = [
                        'bicycle', 'hesitation', 'lane_change', 'nearmiss', 'obstacles',
                        'other', 'overshoot', 'oversteering', 'parking', 'pedestrian',
                        'road_conditions', 'stop_behavior', 'turning'
                    ]
                    
                    if predicted_idx < len(behavior_classes):
                        predicted_behavior = behavior_classes[predicted_idx]
                    else:
                        predicted_behavior = 'other'
                    
                    predictions['deep_learning'] = {
                        'predicted_behavior': predicted_behavior,
                        'confidence': float(confidence),
                        'risk_score': float(risk_score.item()),
                        'model': 'deep_learning'
                    }
            except Exception as e:
                predictions['deep_learning'] = {'error': f'Deep Learning failed: {e}'}
        
        return predictions
    
    def enhanced_ensemble_decision(self, individual_predictions: Dict, binary_filter_result: Dict) -> Dict:
        """
        Enhanced ensemble decision incorporating binary filter and your existing logic
        """
        
        # Filter out failed predictions
        valid_predictions = {k: v for k, v in individual_predictions.items() if 'error' not in v}
        
        if not valid_predictions:
            return {
                'ensemble_prediction': 'error',
                'confidence': 0.0,
                'strategy': 'all_models_failed',
                'binary_filter': binary_filter_result
            }
        
        # If binary filter says "not interesting", lower ensemble confidence
        interest_penalty = 1.0 if binary_filter_result['is_interesting'] else 0.7
        
        # Weighted voting (your existing logic)
        behavior_scores = {}
        total_weight = 0
        
        for model_name, pred in valid_predictions.items():
            if pred['confidence'] >= 0.2:  # Lower threshold since we have binary filter
                behavior = pred['predicted_behavior']
                weight = self.model_weights.get(model_name, 0.0)
                confidence = pred['confidence']
                
                base_score = weight * confidence * interest_penalty
                
                if behavior in behavior_scores:
                    behavior_scores[behavior] += base_score
                else:
                    behavior_scores[behavior] = base_score
                
                total_weight += weight
        
        if not behavior_scores:
            # Fallback to highest confidence
            best_pred = max(valid_predictions.values(), key=lambda x: x['confidence'])
            return {
                'ensemble_prediction': best_pred['predicted_behavior'],
                'confidence': best_pred['confidence'] * interest_penalty,
                'strategy': 'fallback_highest_confidence',
                'binary_filter': binary_filter_result,
                'individual_predictions': individual_predictions
            }
        
        # Find winning behavior
        best_behavior = max(behavior_scores.items(), key=lambda x: x[1])
        ensemble_confidence = best_behavior[1] / total_weight if total_weight > 0 else 0
        
        return {
            'ensemble_prediction': best_behavior[0],
            'confidence': ensemble_confidence,
            'strategy': 'enhanced_weighted_voting_with_binary_filter',
            'binary_filter': binary_filter_result,
            'behavior_scores': behavior_scores,
            'individual_predictions': individual_predictions,
            'interest_penalty_applied': interest_penalty < 1.0
        }
    
    def analyze_segment_enhanced(self, features_array: np.ndarray, features_dict: Dict) -> Dict:
        """
        Complete enhanced analysis using binary filter + existing 4-model ensemble
        """
        
        # Stage 1: Binary interest filter
        binary_result = self.binary_interest_filter(features_dict)
        
        # Stage 2: Full ensemble prediction (run regardless, but use binary result for weighting)
        individual_predictions = self.predict_with_existing_models(features_array, features_dict)
        
        # Stage 3: Enhanced ensemble decision
        ensemble_result = self.enhanced_ensemble_decision(individual_predictions, binary_result)
        
        # Risk assessment (your existing logic enhanced)
        risk_level = self._assess_enhanced_risk(ensemble_result, features_dict, binary_result)
        
        return {
            'binary_filter': binary_result,
            'ensemble_analysis': ensemble_result,
            'risk_assessment': {
                'risk_level': risk_level,
                'contributing_factors': self._get_risk_factors(features_dict, binary_result)
            },
            'recommended_actions': self._get_enhanced_recommendations(ensemble_result, binary_result, risk_level)
        }
    
    def _assess_enhanced_risk(self, ensemble_result: Dict, features_dict: Dict, binary_result: Dict) -> str:
        """Enhanced risk assessment"""
        
        risk_points = 0
        
        # Binary filter risk factors
        if binary_result['is_interesting']:
            risk_points += len(binary_result['reasons'])
        
        # Behavioral risk
        behavior = ensemble_result['ensemble_prediction']
        confidence = ensemble_result['confidence']
        
        high_risk_behaviors = ['nearmiss', 'overshoot', 'oversteering']
        if behavior in high_risk_behaviors and confidence > 0.3:
            risk_points += 3
        
        # Motion intensity
        jerk = features_dict.get('jerk_rms', 0)
        if jerk > 45:
            risk_points += 2
        elif jerk > 35:
            risk_points += 1
        
        # Deceleration intensity
        decel = abs(features_dict.get('max_deceleration', 0))
        if decel > 15:
            risk_points += 2
        elif decel > 10:
            risk_points += 1
        
        # Duration (very long events are riskier)
        duration = features_dict.get('duration_s', 0)
        if duration > 15:
            risk_points += 1
        
        # Classify
        if risk_points >= 6:
            return 'CRITICAL'
        elif risk_points >= 4:
            return 'HIGH'
        elif risk_points >= 2:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _get_risk_factors(self, features_dict: Dict, binary_result: Dict) -> List[str]:
        """Get list of contributing risk factors"""
        
        factors = []
        
        # From binary filter
        factors.extend(binary_result['reasons'])
        
        # Additional motion analysis
        jerk = features_dict.get('jerk_rms', 0)
        if jerk > 50:
            factors.append(f"Extremely rough motion (jerk: {jerk:.1f})")
        
        smoothness = features_dict.get('motion_smoothness', 1.0)
        if smoothness < 0.02:
            factors.append(f"Very erratic trajectory (smoothness: {smoothness:.3f})")
        
        return factors
    
    def _get_enhanced_recommendations(self, ensemble_result: Dict, binary_result: Dict, risk_level: str) -> List[str]:
        """Enhanced recommendations"""
        
        actions = []
        
        # Risk-based actions
        if risk_level in ['CRITICAL', 'HIGH']:
            actions.append("🚨 Immediate review required")
            actions.append("📹 Watch full video segment")
        
        # Binary filter insights
        if binary_result['is_interesting']:
            actions.append(f"🎯 Detected interesting behavior: {', '.join(binary_result['reasons'])}")
        else:
            actions.append("ℹ️ Segment appears normal - low priority")
        
        # Ensemble insights
        behavior = ensemble_result['ensemble_prediction']
        confidence = ensemble_result['confidence']
        
        if confidence > 0.7:
            actions.append(f"✅ High confidence {behavior} detection")
        elif confidence < 0.3:
            actions.append("⚠️ Low confidence - manual review recommended")
        
        # Model agreement
        strategy = ensemble_result.get('strategy', '')
        if 'fallback' in strategy:
            actions.append("🤔 Model disagreement - check multiple perspectives")
        
        return actions

def main():
    """Test the enhanced ensemble system"""
    
    # Initialize enhanced ensemble
    ensemble = EnhancedTriageEnsemble()
    
    # Load your existing models
    model_paths = {
        'advanced': 'assets/models/advanced_triage_brain.pkl',
        'simple': 'assets/models/simple_ml_triage.pkl',
        'rule': 'practical_triage_brain.json',
        'dl': 'best_fixed_model.pth'
    }
    
    try:
        ensemble.load_existing_models(model_paths)
    except:
        print("⚠️ Models not found - this is a demonstration")
        return
    
    # Test with your feature data
    features_dict = {
        'jerk_rms': 48.5,
        'velocity_mean': -2.1,
        'max_deceleration': -18.0,
        'duration_s': 5.2,
        'motion_smoothness': 0.021,
        'acceleration_reversals': 45,
        'velocity_std': 1.8,
        'deceleration_events': 15,
        'jerk_per_second': 148.0,
        'accel_changes_per_second': 100.0
    }
    
    features_array = np.array([
        48.5, -2.1, 1.8, -50, 50, -0.5, 12.0, -20, 20,
        -6.0, 37.0, 37.1, -18.0, 15, 2, 45, 0.021, 148.0, 100.0, 5.2, -1.35
    ])
    
    # Analyze segment
    result = ensemble.analyze_segment_enhanced(features_array, features_dict)
    
    print("🎯 ENHANCED ENSEMBLE RESULTS")
    print("=" * 60)
    
    # Binary filter results
    binary = result['binary_filter']
    print(f"Binary Filter: {'🎯 INTERESTING' if binary['is_interesting'] else '😴 Normal'}")
    print(f"Interest Score: {binary['interest_score']:.2f}")
    if binary['reasons']:
        print(f"Reasons: {', '.join(binary['reasons'])}")
    
    # Ensemble results
    ensemble_analysis = result['ensemble_analysis']
    print(f"\nEnsemble Prediction: {ensemble_analysis['ensemble_prediction']}")
    print(f"Confidence: {ensemble_analysis['confidence']:.3f}")
    print(f"Strategy: {ensemble_analysis['strategy']}")
    
    # Risk assessment
    risk = result['risk_assessment']
    print(f"\nRisk Level: {risk['risk_level']}")
    if risk['contributing_factors']:
        print(f"Risk Factors: {', '.join(risk['contributing_factors'])}")
    
    # Recommendations
    print(f"\nRecommended Actions:")
    for action in result['recommended_actions']:
        print(f"  • {action}")

if __name__ == "__main__":
    main()