#!/usr/bin/env python3
"""
Practical Triage Brain - A hybrid approach combining rule-based detection
with simple ML for small datasets.
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import Counter

class PracticalTriageBrain:
    """
    A practical implementation of Triage Brain that works well with small datasets
    by combining rule-based detection with simple statistical analysis.
    """
    
    def __init__(self):
        self.behavioral_rules = {}
        self.feature_thresholds = {}
        self.anomaly_thresholds = {}
        self.training_stats = {}

    def load_model(self, filepath: str) -> None:
        """Load a trained model from disk."""
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        self.behavioral_rules = model_data['behavioral_rules']
        self.anomaly_thresholds = model_data['anomaly_thresholds']
        self.training_stats = model_data['training_stats']
        
        print(f" Model loaded from {filepath}")

    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load labeled feature vectors."""
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        
        df = pd.DataFrame(data)
        # Clean infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        print(f"Loaded {len(df)} labeled segments")
        return df
    
    def create_behavioral_rules(self, df: pd.DataFrame) -> Dict:
        """Create rule-based classifiers for each behavior type."""
        
        rules = {}
        
        # Group by behavior type and analyze patterns
        for behavior in df['primary_label'].unique():
            if behavior == 'other':
                continue
                
            behavior_data = df[df['primary_label'] == behavior]
            
            if len(behavior_data) < 2:
                continue
            
            # Calculate statistical thresholds for key features
            rule_conditions = {}
            
            # Duration patterns
            durations = behavior_data['duration_s'].dropna()
            if len(durations) > 0:
                rule_conditions['duration'] = {
                    'min': float(durations.quantile(0.25)),
                    'max': float(durations.quantile(0.75)),
                    'mean': float(durations.mean())
                }
            
            # Jerk patterns (motion roughness)
            jerk_rms = behavior_data['jerk_rms'].dropna()
            if len(jerk_rms) > 0:
                rule_conditions['jerk_intensity'] = {
                    'threshold': float(jerk_rms.mean()),
                    'std': float(jerk_rms.std()) if len(jerk_rms) > 1 else 0.0
                }
            
            # Deceleration patterns
            max_decel = behavior_data['max_deceleration'].dropna()
            if len(max_decel) > 0:
                rule_conditions['braking_intensity'] = {
                    'threshold': float(abs(max_decel.mean())),
                    'strong_braking': float(abs(max_decel.quantile(0.75)))
                }
            
            # Motion smoothness
            smoothness = behavior_data['motion_smoothness'].dropna()
            if len(smoothness) > 0:
                rule_conditions['smoothness'] = {
                    'threshold': float(smoothness.mean()),
                    'std': float(smoothness.std()) if len(smoothness) > 1 else 0.0
                }
            
            # Acceleration patterns (indecision)
            accel_reversals = behavior_data['acceleration_reversals'].dropna()
            durations_for_reversals = behavior_data['duration_s'].dropna()
            if len(accel_reversals) > 0 and len(durations_for_reversals) > 0:
                reversals_per_sec = accel_reversals / durations_for_reversals
                rule_conditions['indecision_rate'] = {
                    'threshold': float(reversals_per_sec.mean()),
                    'high_indecision': float(reversals_per_sec.quantile(0.75))
                }
            
            rules[behavior] = {
                'sample_count': len(behavior_data),
                'conditions': rule_conditions,
                'example_comments': behavior_data['comment'].tolist()[:3]
            }
        
        self.behavioral_rules = rules
        return rules
    
    def create_anomaly_detector(self, df: pd.DataFrame) -> Dict:
        """Create simple statistical anomaly detection."""
        
        # Key features for anomaly detection
        anomaly_features = [
            'jerk_rms', 'velocity_std', 'acceleration_reversals', 
            'duration_s', 'motion_smoothness'
        ]
        
        thresholds = {}
        
        for feature in anomaly_features:
            if feature in df.columns:
                values = df[feature].replace([np.inf, -np.inf], np.nan).dropna()
                
                if len(values) > 5:  # Need enough samples
                    mean_val = values.mean()
                    std_val = values.std()
                    
                    # Define anomaly as beyond 2 standard deviations
                    thresholds[feature] = {
                        'mean': float(mean_val),
                        'std': float(std_val),
                        'upper_threshold': float(mean_val + 2 * std_val),
                        'lower_threshold': float(mean_val - 2 * std_val)
                    }
        
        self.anomaly_thresholds = thresholds
        
        # Test on training data to see what gets flagged
        anomalies_found = []
        for idx, row in df.iterrows():
            anomaly_score = self.calculate_anomaly_score(row)
            if anomaly_score > 2.0:  # High anomaly score
                anomalies_found.append({
                    'index': int(idx),
                    'comment': row['comment'],
                    'anomaly_score': float(anomaly_score),
                    'clip_name': row['clip_name']
                })
        
        # Sort by anomaly score
        anomalies_found.sort(key=lambda x: x['anomaly_score'], reverse=True)
        
        return {
            'thresholds': thresholds,
            'anomalies_in_training': anomalies_found[:10],
            'total_anomalies': len(anomalies_found)
        }
    
    def calculate_anomaly_score(self, sample: pd.Series) -> float:
        """Calculate anomaly score for a single sample."""
        
        total_score = 0.0
        feature_count = 0
        
        for feature, threshold_info in self.anomaly_thresholds.items():
            if feature in sample and not pd.isna(sample[feature]):
                value = sample[feature]
                mean_val = threshold_info['mean']
                std_val = threshold_info['std']
                
                if std_val > 0:
                    # Z-score based anomaly scoring
                    z_score = abs(value - mean_val) / std_val
                    total_score += z_score
                    feature_count += 1
        
        return total_score / max(feature_count, 1)
    
    def classify_behavior(self, sample: pd.Series) -> Dict:
        """Classify a behavior sample using rule-based approach."""
        
        predictions = {}
        
        for behavior, rule in self.behavioral_rules.items():
            confidence = 0.0
            matching_conditions = 0
            total_conditions = 0
            
            conditions = rule['conditions']
            
            # Check duration pattern
            if 'duration' in conditions and 'duration_s' in sample and not pd.isna(sample['duration_s']):
                duration = sample['duration_s']
                duration_range = conditions['duration']
                
                if duration_range['min'] <= duration <= duration_range['max']:
                    confidence += 0.3
                    matching_conditions += 1
                total_conditions += 1
            
            # Check jerk intensity
            if 'jerk_intensity' in conditions and 'jerk_rms' in sample and not pd.isna(sample['jerk_rms']):
                jerk = sample['jerk_rms']
                jerk_info = conditions['jerk_intensity']
                
                # Close to mean jerk for this behavior
                if abs(jerk - jerk_info['threshold']) < jerk_info['std'] * 1.5:
                    confidence += 0.25
                    matching_conditions += 1
                total_conditions += 1
            
            # Check braking patterns
            if 'braking_intensity' in conditions and 'max_deceleration' in sample and not pd.isna(sample['max_deceleration']):
                decel = abs(sample['max_deceleration'])
                braking_info = conditions['braking_intensity']
                
                if decel >= braking_info['threshold'] * 0.8:  # Within 80% of typical braking
                    confidence += 0.2
                    matching_conditions += 1
                total_conditions += 1
            
            # Check smoothness
            if 'smoothness' in conditions and 'motion_smoothness' in sample and not pd.isna(sample['motion_smoothness']):
                smoothness = sample['motion_smoothness']
                smooth_info = conditions['smoothness']
                
                if abs(smoothness - smooth_info['threshold']) < smooth_info['std'] * 1.5:
                    confidence += 0.15
                    matching_conditions += 1
                total_conditions += 1
            
            # Check indecision patterns
            if 'indecision_rate' in conditions and 'acceleration_reversals' in sample and 'duration_s' in sample:
                if not pd.isna(sample['acceleration_reversals']) and not pd.isna(sample['duration_s']):
                    reversals_per_sec = sample['acceleration_reversals'] / max(sample['duration_s'], 0.1)
                    indecision_info = conditions['indecision_rate']
                    
                    if reversals_per_sec >= indecision_info['threshold'] * 0.7:
                        confidence += 0.1
                        matching_conditions += 1
                    total_conditions += 1
            
            # Normalize confidence by number of conditions checked
            if total_conditions > 0:
                confidence = confidence / total_conditions * (matching_conditions / total_conditions)
            
            predictions[behavior] = confidence
        
        # Find best match
        if predictions:
            best_behavior = max(predictions.items(), key=lambda x: x[1])
            return {
                'predicted_behavior': best_behavior[0],
                'confidence': best_behavior[1],
                'all_scores': predictions
            }
        else:
            return {
                'predicted_behavior': 'unknown',
                'confidence': 0.0,
                'all_scores': {}
            }
    
    def analyze_segment(self, sample: pd.Series) -> Dict:
        """Perform complete analysis of a driving segment."""
        
        # Behavioral classification
        behavior_result = self.classify_behavior(sample)
        
        # Anomaly detection
        anomaly_score = self.calculate_anomaly_score(sample)
        is_anomaly = anomaly_score > 2.0
        
        # Extract key characteristics
        characteristics = []
        
        if 'jerk_rms' in sample and not pd.isna(sample['jerk_rms']):
            if sample['jerk_rms'] > 45:
                characteristics.append('high_jerk')
            elif sample['jerk_rms'] < 30:
                characteristics.append('smooth')
        
        if 'max_deceleration' in sample and not pd.isna(sample['max_deceleration']):
            if abs(sample['max_deceleration']) > 15:
                characteristics.append('hard_braking')
        
        if 'duration_s' in sample and not pd.isna(sample['duration_s']):
            if sample['duration_s'] > 15:
                characteristics.append('long_event')
            elif sample['duration_s'] < 2:
                characteristics.append('brief_event')
        
        return {
            'behavior_classification': behavior_result,
            'anomaly_analysis': {
                'anomaly_score': anomaly_score,
                'is_anomaly': is_anomaly
            },
            'characteristics': characteristics,
            'risk_level': self._assess_risk_level(sample, behavior_result, anomaly_score)
        }
    
    def _assess_risk_level(self, sample: pd.Series, behavior_result: Dict, anomaly_score: float) -> str:
        """Assess risk level based on behavior and motion patterns."""
        
        risk_factors = 0
        
        # High anomaly score
        if anomaly_score > 2.5:
            risk_factors += 2
        elif anomaly_score > 2.0:
            risk_factors += 1
        
        # Dangerous behaviors
        dangerous_behaviors = ['nearmiss', 'overshoot', 'risky_behavior']
        if behavior_result['predicted_behavior'] in dangerous_behaviors:
            risk_factors += 2
        
        # High jerk or hard braking
        if 'jerk_rms' in sample and not pd.isna(sample['jerk_rms']) and sample['jerk_rms'] > 45:
            risk_factors += 1
        
        if 'max_deceleration' in sample and not pd.isna(sample['max_deceleration']):
            if abs(sample['max_deceleration']) > 18:
                risk_factors += 1
        
        # Long duration events
        if 'duration_s' in sample and not pd.isna(sample['duration_s']) and sample['duration_s'] > 20:
            risk_factors += 1
        
        # Classify risk level
        if risk_factors >= 4:
            return 'HIGH'
        elif risk_factors >= 2:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def train(self, data_file: str) -> Dict:
        """Train the practical triage brain."""
        
        print("🧠 TRAINING PRACTICAL TRIAGE BRAIN")
        print("=" * 50)
        
        # Load data
        df = self.load_data(data_file)
        
        # Create behavioral rules
        print("\n🎯 Creating Behavioral Rules...")
        behavioral_rules = self.create_behavioral_rules(df)
        
        print(f"Created rules for {len(behavioral_rules)} behaviors:")
        for behavior, rule in behavioral_rules.items():
            print(f"  {behavior:<20}: {rule['sample_count']} samples")
        
        # Create anomaly detector
        print(f"\n🔍 Training Anomaly Detector...")
        anomaly_results = self.create_anomaly_detector(df)
        
        print(f"Anomaly detection setup complete:")
        print(f"  Features monitored: {len(anomaly_results['thresholds'])}")
        print(f"  Training anomalies: {anomaly_results['total_anomalies']}")
        
        # Test on training data
        print(f"\n🧪 Testing on Training Data...")
        correct_predictions = 0
        total_predictions = 0
        
        for idx, row in df.iterrows():
            if row['primary_label'] != 'other':
                result = self.classify_behavior(row)
                if result['predicted_behavior'] == row['primary_label'] and result['confidence'] > 0.3:
                    correct_predictions += 1
                total_predictions += 1
        
        accuracy = correct_predictions / max(total_predictions, 1)
        
        # Save training results
        training_summary = {
            'dataset_size': len(df),
            'behavioral_rules': behavioral_rules,
            'anomaly_detection': anomaly_results,
            'training_accuracy': accuracy,
            'behaviors_covered': len(behavioral_rules)
        }
        
        self.training_stats = training_summary
        
        print(f"\n✅ Training Complete!")
        print(f"Dataset: {len(df)} segments")
        print(f"Behaviors: {len(behavioral_rules)} rule sets created")
        print(f"Training accuracy: {accuracy:.1%}")
        
        return training_summary
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        model_data = {
            'behavioral_rules': self.behavioral_rules,
            'anomaly_thresholds': self.anomaly_thresholds,
            'training_stats': self.training_stats
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        print(f"✅ Model saved to {filepath}")

def main():
    """Train and test the practical triage brain."""
    
    # Initialize and train
    ptb = PracticalTriageBrain()
    results = ptb.train("assets/data/feature_vectors_labeled.jsonl")
    
    # Save model
    ptb.save_model("practical_triage_brain.json")
    
    # Save results
    with open("practical_triage_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💡 PRACTICAL INSIGHTS:")
    
    # Show top behavioral patterns
    sorted_behaviors = sorted(results['behavioral_rules'].items(), 
                             key=lambda x: x[1]['sample_count'], reverse=True)
    
    print(f"Top behavioral patterns:")
    for behavior, rule in sorted_behaviors[:5]:
        count = rule['sample_count']
        print(f"  {behavior:<15}: {count} samples")
        
        # Show key conditions
        conditions = rule['conditions']
        if 'duration' in conditions:
            dur = conditions['duration']['mean']
            print(f"    Typical duration: {dur:.1f}s")
        if 'jerk_intensity' in conditions:
            jerk = conditions['jerk_intensity']['threshold']
            print(f"    Jerk intensity: {jerk:.1f}")
    
    print(f"\n🚨 Most anomalous training segments:")
    for anomaly in results['anomaly_detection']['anomalies_in_training'][:3]:
        score = anomaly['anomaly_score']
        comment = anomaly['comment']
        print(f"  Score {score:.1f}: {comment}")

if __name__ == "__main__":
    main()