#!/usr/bin/env python3
"""
Simple but effective ML Triage Brain - optimized for small datasets
"""

import json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score
from collections import Counter

class SimpleMLTriageBrain:
    """Simple but effective ML model for behavioral classification"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.classifier = RandomForestClassifier(
            n_estimators=50,
            max_depth=6,
            min_samples_split=3,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        self.feature_names = []
        self.feature_importance = {}
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load and clean training data"""
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        
        df = pd.DataFrame(data)
        
        # Core features for classification
        self.feature_names = [
            'velocity_mean', 'velocity_std', 'jerk_rms', 'max_deceleration',
            'acceleration_reversals', 'motion_smoothness', 'duration_s',
            'deceleration_events', 'jerk_per_second', 'accel_changes_per_second'
        ]
        
        # Keep only rows with these features
        df_clean = df.dropna(subset=self.feature_names)
        
        # Handle infinite values
        for col in self.feature_names:
            df_clean[col] = df_clean[col].replace([np.inf, -np.inf], df_clean[col].median())
        
        print(f"Loaded {len(df_clean)} samples with {len(self.feature_names)} features")
        return df_clean
    
    def train(self, data_file: str) -> dict:
        """Train the ML model"""
        print("🎯 Training Simple ML Triage Brain...")
        
        # Load data
        df = self.load_data(data_file)
        
        # Prepare features
        X = df[self.feature_names].values
        y = df['primary_label'].values
        
        # Filter classes with enough samples
        label_counts = Counter(y)
        valid_labels = [label for label, count in label_counts.items() if count >= 2]
        
        print(f"Training on {len(valid_labels)} classes:")
        for label in valid_labels:
            print(f"  {label}: {label_counts[label]} samples")
        
        # Filter to valid labels
        valid_mask = pd.Series(y).isin(valid_labels)
        X_filtered = X[valid_mask]
        y_filtered = y[valid_mask]
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y_filtered)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_filtered)
        
        # Train model
        self.classifier.fit(X_scaled, y_encoded)
        
        # Cross-validation
        cv_scores = cross_val_score(self.classifier, X_scaled, y_encoded, cv=3)
        
        # Feature importance
        self.feature_importance = dict(zip(self.feature_names, self.classifier.feature_importances_))
        self.feature_importance = dict(sorted(self.feature_importance.items(), 
                                            key=lambda x: x[1], reverse=True))
        
        results = {
            'cv_accuracy': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'n_classes': len(valid_labels),
            'n_samples': len(X_filtered),
            'feature_importance': self.feature_importance
        }
        
        print(f"✅ Training complete:")
        print(f"   CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print(f"   Classes: {len(valid_labels)}")
        
        return results
    
    def predict(self, features_dict: dict) -> dict:
        """Predict behavior from feature dictionary"""
        # Extract features in correct order
        feature_values = []
        for feature_name in self.feature_names:
            feature_values.append(features_dict.get(feature_name, 0.0))
        
        X = np.array(feature_values).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # Predict
        probabilities = self.classifier.predict_proba(X_scaled)[0]
        predicted_idx = np.argmax(probabilities)
        predicted_class = self.label_encoder.classes_[predicted_idx]
        confidence = probabilities[predicted_idx]
        
        # Risk assessment
        risk_level = self._assess_risk(features_dict, predicted_class, confidence)
        
        return {
            'predicted_behavior': predicted_class,
            'confidence': float(confidence),
            'risk_level': risk_level,
            'probabilities': {
                class_name: float(prob) 
                for class_name, prob in zip(self.label_encoder.classes_, probabilities)
            }
        }
    
    def _assess_risk(self, features: dict, behavior: str, confidence: float) -> str:
        """Simple risk assessment"""
        jerk = features.get('jerk_rms', 0)
        deceleration = abs(features.get('max_deceleration', 0))
        
        # High risk behaviors
        if behavior in ['nearmiss', 'overshoot'] and confidence > 0.3:
            return 'HIGH'
        
        # High motion intensity
        if jerk > 45 and deceleration > 15:
            return 'HIGH'
        
        # Medium risk
        if jerk > 40 or deceleration > 12:
            return 'MEDIUM'
        
        return 'LOW'
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        model_data = {
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'classifier': self.classifier,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance
        }
        joblib.dump(model_data, filepath)
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.classifier = model_data['classifier']
        self.feature_names = model_data['feature_names']
        self.feature_importance = model_data['feature_importance']
        print(f"✅ Model loaded from {filepath}")

def main():
    """Train and save the simple ML model"""
    tb = SimpleMLTriageBrain()
    
    # Train
    results = tb.train("assets/data/feature_vectors_labeled.jsonl")
    
    # Save model
    tb.save_model("assets/models/simple_ml_triage.pkl")
    
    # Show top features
    print(f"\n🔝 Top Features:")
    for i, (feature, importance) in enumerate(list(tb.feature_importance.items())[:5]):
        print(f"  {i+1}. {feature}: {importance:.3f}")
    
    # Test prediction
    test_features = {
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
    
    result = tb.predict(test_features)
    print(f"\n🧪 Test Prediction:")
    print(f"   Behavior: {result['predicted_behavior']}")
    print(f"   Confidence: {result['confidence']:.3f}")
    print(f"   Risk: {result['risk_level']}")

if __name__ == "__main__":
    main()