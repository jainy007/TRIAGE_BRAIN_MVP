#!/usr/bin/env python3
"""
Improved Triage Brain - Random Forest optimized for small datasets
with proper feature engineering and regularization techniques.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter
import pickle
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import joblib

class ImprovedTriageBrain:
    """
    Advanced Triage Brain with Random Forest optimized for small datasets.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.label_encoder = LabelEncoder()
        self.feature_selector = SelectKBest(f_classif, k=10)  # Select top features
        self.classifier = None
        self.pipeline = None
        self.anomaly_detector = None
        self.feature_importance = {}
        self.training_stats = {}
        self.feature_names = []
        
    def _default_config(self) -> Dict:
        """Optimized configuration for small datasets"""
        return {
            'random_state': 42,
            'test_size': 0.25,
            'cv_folds': 3,  # Reduced for small dataset
            'min_samples_per_class': 2,
            'anomaly_contamination': 0.1,
            'feature_selection': True,
            'n_estimators': 100,
            'max_depth': 8,  # Prevent overfitting
            'min_samples_split': 5,
            'min_samples_leaf': 3,
            'bootstrap': True,
            'class_weight': 'balanced'  # Handle class imbalance
        }
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load and prepare training data with enhanced preprocessing"""
        print(f"Loading training data from {file_path}...")
        
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        
        df = pd.DataFrame(data)
        print(f"Loaded {len(df)} labeled segments")
        
        # Enhanced data cleaning
        df = self._enhanced_data_cleaning(df)
        
        return df
    
    def _enhanced_data_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced data cleaning with outlier handling and feature engineering"""
        
        # Handle infinite and NaN values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Core motion features
        motion_features = [
            'velocity_mean', 'velocity_std', 'velocity_min', 'velocity_max',
            'acceleration_mean', 'acceleration_std', 'acceleration_min', 'acceleration_max',
            'jerk_mean', 'jerk_std', 'jerk_rms',
            'max_deceleration', 'deceleration_events',
            'velocity_zero_crossings', 'acceleration_reversals',
            'motion_smoothness', 'jerk_per_second', 'accel_changes_per_second',
            'duration_s', 'distance_traveled'
        ]
        
        # Keep only available features
        available_features = [f for f in motion_features if f in df.columns]
        
        # Fill NaN values with median for numerical stability
        for feature in available_features:
            if df[feature].dtype in ['float64', 'int64']:
                median_val = df[feature].median()
                df[feature] = df[feature].fillna(median_val)
        
        # Engineer additional features
        df = self._engineer_features(df)
        
        # Remove rows with too many missing values
        df_clean = df.dropna(subset=available_features[:10], thresh=8)  # At least 8 of 10 features
        
        print(f"Data cleaning: {len(df)} → {len(df_clean)} segments")
        print(f"Using {len(available_features)} base features + engineered features")
        
        return df_clean
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer additional features to improve model performance"""
        
        # Ratio features (often more informative than absolutes)
        if 'jerk_rms' in df.columns and 'duration_s' in df.columns:
            df['jerk_intensity_per_second'] = df['jerk_rms'] / (df['duration_s'] + 0.1)
        
        if 'velocity_std' in df.columns and 'velocity_mean' in df.columns:
            df['velocity_variability_ratio'] = df['velocity_std'] / (abs(df['velocity_mean']) + 0.1)
        
        if 'acceleration_reversals' in df.columns and 'duration_s' in df.columns:
            df['indecision_intensity'] = df['acceleration_reversals'] / (df['duration_s'] + 0.1)
        
        # Motion complexity features
        if 'jerk_std' in df.columns and 'jerk_rms' in df.columns:
            df['motion_consistency'] = df['jerk_rms'] / (df['jerk_std'] + 0.1)
        
        # Safety features
        if 'max_deceleration' in df.columns:
            df['braking_intensity'] = abs(df['max_deceleration'])
        
        # Duration categorization (sometimes categorical is better than continuous)
        if 'duration_s' in df.columns:
            df['duration_category'] = pd.cut(df['duration_s'], 
                                           bins=[0, 3, 8, 15, float('inf')], 
                                           labels=['brief', 'normal', 'long', 'extended'],
                                           include_lowest=True).astype(str)
            
            # One-hot encode duration categories
            duration_dummies = pd.get_dummies(df['duration_category'], prefix='duration')
            df = pd.concat([df, duration_dummies], axis=1)
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Extract and prepare features with automatic selection"""
        
        # Get all numeric features
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove non-feature columns
        exclude_columns = ['start_frame', 'end_frame', 'total_frames', 'fps', 'label_count']
        feature_columns = [col for col in numeric_columns if col not in exclude_columns]
        
        print(f"Using {len(feature_columns)} engineered features for training:")
        
        # Select features present in the dataframe
        available_features = [col for col in feature_columns if col in df.columns]
        
        for i, feat in enumerate(available_features[:15]):  # Show first 15
            print(f"  {i+1:2d}. {feat}")
        if len(available_features) > 15:
            print(f"  ... and {len(available_features)-15} more")
        
        # Extract feature matrix
        X = df[available_features].fillna(0).values
        
        # Handle any remaining infinite values
        X = np.nan_to_num(X, nan=0.0, posinf=999, neginf=-999)
        
        self.feature_names = available_features
        
        return X, available_features
    
    def train_model(self, df: pd.DataFrame) -> Dict:
        """Train Random Forest with hyperparameter optimization"""
        print("\n🎯 Training Advanced Random Forest Model...")
        
        # Prepare features and labels
        X, feature_names = self.prepare_features(df)
        y = df['primary_label'].values
        
        # Filter classes with minimum samples
        label_counts = Counter(y)
        valid_labels = [label for label, count in label_counts.items() 
                       if count >= self.config['min_samples_per_class']]
        
        # Filter dataset to valid labels
        valid_mask = pd.Series(y).isin(valid_labels)
        X_filtered = X[valid_mask]
        y_filtered = y[valid_mask]
        
        print(f"Training on {len(valid_labels)} behavior classes:")
        for label in valid_labels:
            count = label_counts[label]
            print(f"  {label:<15}: {count:>2} samples")
        
        if len(y_filtered) < 10:
            print("❌ Too few samples for reliable ML training")
            return {"error": "Insufficient data"}
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y_filtered)
        
        # Create pipeline with preprocessing and model
        self.pipeline = Pipeline([
            ('scaler', self.scaler),
            ('feature_selection', self.feature_selector if self.config['feature_selection'] else 'passthrough'),
            ('classifier', RandomForestClassifier(
                n_estimators=self.config['n_estimators'],
                max_depth=self.config['max_depth'],
                min_samples_split=self.config['min_samples_split'],
                min_samples_leaf=self.config['min_samples_leaf'],
                bootstrap=self.config['bootstrap'],
                class_weight=self.config['class_weight'],
                random_state=self.config['random_state']
            ))
        ])
        
        # Hyperparameter optimization for small datasets
        if len(y_filtered) > 15:  # Only if we have enough samples
            param_grid = {
                'classifier__n_estimators': [50, 100],
                'classifier__max_depth': [6, 8, 10],
                'classifier__min_samples_split': [3, 5],
                'feature_selection__k': [8, 10, 12] if self.config['feature_selection'] else [10]
            }
            
            grid_search = GridSearchCV(
                self.pipeline, 
                param_grid, 
                cv=min(3, len(valid_labels)),  # At least as many folds as classes
                scoring='accuracy',
                n_jobs=-1
            )
            
            grid_search.fit(X_filtered, y_encoded)
            self.pipeline = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            self.pipeline.fit(X_filtered, y_encoded)
        
        # Evaluate model
        cv_scores = cross_val_score(
            self.pipeline, X_filtered, y_encoded,
            cv=min(self.config['cv_folds'], len(valid_labels)),
            scoring='accuracy'
        )
        
        # Get feature importance
        if hasattr(self.pipeline.named_steps['classifier'], 'feature_importances_'):
            # Handle feature selection
            if self.config['feature_selection']:
                selected_features = self.pipeline.named_steps['feature_selection'].get_support()
                selected_feature_names = [feature_names[i] for i, selected in enumerate(selected_features) if selected]
                importances = self.pipeline.named_steps['classifier'].feature_importances_
            else:
                selected_feature_names = feature_names
                importances = self.pipeline.named_steps['classifier'].feature_importances_
            
            self.feature_importance = dict(zip(selected_feature_names, importances))
            self.feature_importance = dict(sorted(self.feature_importance.items(), 
                                                key=lambda x: x[1], reverse=True))
        
        # Train-test split for detailed evaluation (only if enough samples)
        if len(y_filtered) >= 8:  # Need at least 8 samples for meaningful split
            X_train, X_test, y_train, y_test = train_test_split(
                X_filtered, y_encoded, 
                test_size=min(0.3, max(0.2, len(y_filtered) * 0.25)),  # Adaptive test size
                random_state=self.config['random_state'],
                stratify=y_encoded if len(np.unique(y_encoded)) > 1 else None
            )
            
            test_predictions = self.pipeline.predict(X_test)
            test_accuracy = accuracy_score(y_test, test_predictions)
            
            # Generate classification report
            class_names = self.label_encoder.classes_
            
            # Get unique classes in test set to avoid mismatch
            unique_test_classes = np.unique(np.concatenate([y_test, test_predictions]))
            test_class_names = [class_names[i] for i in unique_test_classes]
            
            try:
                class_report = classification_report(y_test, test_predictions, 
                                               labels=unique_test_classes,
                                               target_names=test_class_names, 
                                               output_dict=True, 
                                               zero_division=0)
            except Exception as e:
                print(f"Warning: Could not generate detailed classification report: {e}")
                class_report = {"accuracy": test_accuracy}
        else:
            print("Dataset too small for train-test split, using CV scores only")
            test_accuracy = cv_scores.mean()
            class_report = {"cv_accuracy": test_accuracy}
        
        results = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_accuracy': test_accuracy,
            'feature_importance': self.feature_importance,
            'classification_report': class_report,
            'valid_classes': valid_labels,
            'total_samples': len(X_filtered),
            'feature_names': feature_names,
            'selected_features': len(self.feature_importance) if self.feature_importance else len(feature_names)
        }
        
        self.training_stats['classifier'] = results
        
        print(f"✅ Random Forest trained:")
        print(f"   CV accuracy:       {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print(f"   Test accuracy:     {test_accuracy:.3f}")
        print(f"   Features selected: {results['selected_features']}")
        
        return results
    
    def train_anomaly_detector(self, df: pd.DataFrame) -> Dict:
        """Train anomaly detection with improved parameters"""
        print("\n🔍 Training Anomaly Detector...")
        
        X, _ = self.prepare_features(df)
        
        # Use the fitted scaler from the main pipeline
        if self.pipeline:
            X_scaled = self.pipeline.named_steps['scaler'].transform(X)
        else:
            X_scaled = self.scaler.fit_transform(X)
        
        # Train Isolation Forest
        self.anomaly_detector = IsolationForest(
            contamination=self.config['anomaly_contamination'],
            random_state=self.config['random_state'],
            n_estimators=100
        )
        
        anomaly_labels = self.anomaly_detector.fit_predict(X_scaled)
        anomaly_scores = self.anomaly_detector.score_samples(X_scaled)
        
        # Analyze results
        n_anomalies = np.sum(anomaly_labels == -1)
        
        # Identify most anomalous segments
        anomaly_indices = np.where(anomaly_labels == -1)[0]
        anomalous_segments = []
        
        for idx in anomaly_indices:
            anomalous_segments.append({
                'index': int(idx),
                'comment': df.iloc[idx]['comment'],
                'anomaly_score': float(anomaly_scores[idx]),
                'clip_name': df.iloc[idx]['clip_name']
            })
        
        anomalous_segments.sort(key=lambda x: x['anomaly_score'])
        
        results = {
            'total_segments': len(X),
            'anomalous_segments': int(n_anomalies),
            'anomaly_rate': float(n_anomalies / len(X)),
            'most_anomalous': anomalous_segments[:5]
        }
        
        self.training_stats['anomaly_detector'] = results
        
        print(f"✅ Anomaly detector trained:")
        print(f"   Anomalous segments: {n_anomalies}/{len(X)} ({100*n_anomalies/len(X):.1f}%)")
        
        return results
    
    def predict(self, features: np.ndarray) -> Dict:
        """Predict behavior class and anomaly score for new features"""
        if self.pipeline is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Ensure features is 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Handle missing features by padding with zeros
        if features.shape[1] < len(self.feature_names):
            padding = np.zeros((features.shape[0], len(self.feature_names) - features.shape[1]))
            features = np.hstack([features, padding])
        elif features.shape[1] > len(self.feature_names):
            features = features[:, :len(self.feature_names)]
        
        # Replace any inf/nan values
        features = np.nan_to_num(features, nan=0.0, posinf=999, neginf=-999)
        
        # Primary classification
        class_probs = self.pipeline.predict_proba(features)[0]
        predicted_class_idx = np.argmax(class_probs)
        predicted_class = self.label_encoder.classes_[predicted_class_idx]
        confidence = class_probs[predicted_class_idx]
        
        # Anomaly detection
        anomaly_score = self.anomaly_detector.score_samples(features)[0] if self.anomaly_detector else 0.0
        is_anomaly = self.anomaly_detector.predict(features)[0] == -1 if self.anomaly_detector else False
        
        return {
            'predicted_behavior': predicted_class,
            'confidence': float(confidence),
            'all_probabilities': {
                class_name: float(prob) 
                for class_name, prob in zip(self.label_encoder.classes_, class_probs)
            },
            'anomaly_score': float(anomaly_score),
            'is_anomaly': bool(is_anomaly)
        }
    
    def save_model(self, filepath: str) -> None:
        """Save trained model to disk using joblib for better sklearn compatibility"""
        model_data = {
            'pipeline': self.pipeline,
            'label_encoder': self.label_encoder,
            'anomaly_detector': self.anomaly_detector,
            'feature_importance': self.feature_importance,
            'feature_names': self.feature_names,
            'training_stats': self.training_stats,
            'config': self.config
        }
        
        joblib.dump(model_data, filepath)
        print(f"✅ Advanced model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load trained model from disk"""
        model_data = joblib.load(filepath)
        
        self.pipeline = model_data['pipeline']
        self.label_encoder = model_data['label_encoder']
        self.anomaly_detector = model_data['anomaly_detector']
        self.feature_importance = model_data['feature_importance']
        self.feature_names = model_data['feature_names']
        self.training_stats = model_data['training_stats']
        self.config = model_data['config']
        
        print(f"✅ Advanced model loaded from {filepath}")
    
    def train_full_pipeline(self, data_file: str) -> Dict:
        """Train the complete advanced Triage Brain pipeline"""
        print("🧠 TRAINING ADVANCED TRIAGE BRAIN")
        print("=" * 60)
        
        # Load data
        df = self.load_data(data_file)
        
        # Train classifier
        classifier_results = self.train_model(df)
        
        if 'error' in classifier_results:
            print("❌ Training failed due to insufficient data")
            return classifier_results
        
        # Train anomaly detector
        anomaly_results = self.train_anomaly_detector(df)
        
        # Generate summary report
        training_summary = {
            'dataset_size': len(df),
            'classifier': classifier_results,
            'anomaly_detector': anomaly_results,
            'top_features': list(self.feature_importance.keys())[:10] if self.feature_importance else []
        }
        
        print("\n🎯 TRAINING COMPLETE!")
        print(f"Dataset: {len(df)} segments")
        print(f"Classifier accuracy: {classifier_results['cv_mean']:.3f} ± {classifier_results['cv_std']:.3f}")
        print(f"Features selected: {classifier_results['selected_features']}")
        print(f"Anomaly detection: {anomaly_results['anomaly_rate']:.1%} flagged")
        
        if self.feature_importance:
            print(f"\n🔝 TOP PREDICTIVE FEATURES:")
            for i, (feature, importance) in enumerate(list(self.feature_importance.items())[:8]):
                print(f"  {i+1}. {feature:<25}: {importance:.3f}")
        
        return training_summary

def main():
    """Train and test the improved Random Forest model"""
    
    # Initialize
    tb = ImprovedTriageBrain()
    
    # Train
    results = tb.train_full_pipeline("assets/data/feature_vectors_labeled.jsonl")
    
    if 'error' not in results:
        # Save model
        tb.save_model("assets/models/advanced_triage_brain.pkl")
        
        # Save training results
        with open("outputs/reports/advanced_training_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to outputs/reports/advanced_training_results.json")
        
        # Test the model
        print(f"\n🧪 Testing model on sample data...")
        test_features = np.array([
            48.5, -2.1, 1.8, -50, 50, -0.5, 12.0, -20, 20,
            -6.0, 37.0, 37.1, -18.0, 15, 2, 45, 0.021, 148.0, 100.0, 5.2, -1.35
        ])
        
        try:
            result = tb.predict(test_features)
            print(f"✅ Test prediction successful:")
            print(f"   Behavior: {result['predicted_behavior']}")
            print(f"   Confidence: {result['confidence']:.3f}")
            print(f"   Anomaly: {result['is_anomaly']}")
        except Exception as e:
            print(f"❌ Test prediction failed: {e}")

if __name__ == "__main__":
    main()