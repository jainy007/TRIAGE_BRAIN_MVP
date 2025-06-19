#!/usr/bin/env python3
"""
Ensemble Engine Checkpoint Test - Fix step by step
Run this independently to verify each checkpoint before touching main.py
"""

import os
import sys
import pandas as pd
import numpy as np

# GLOBAL PATHS - NO MORE BULLSHIT
PROJECT_ROOT = "/home/jainy007/PEM/triage_brain"
V2_PATH = os.path.join(PROJECT_ROOT, "src", "triage_brain_v2")
MODEL_PREFIX = os.path.join(PROJECT_ROOT, "triage_brain_model")

print("🔧 ENSEMBLE ENGINE CHECKPOINT TEST")
print("=" * 50)

# CHECKPOINT 1: PATH VERIFICATION
print("\n📍 CHECKPOINT 1: PATH VERIFICATION")
print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"Exists: {os.path.exists(PROJECT_ROOT)}")

print(f"V2_PATH: {V2_PATH}")
print(f"Exists: {os.path.exists(V2_PATH)}")

print(f"MODEL_PREFIX: {MODEL_PREFIX}")
model_files = [
    f"{MODEL_PREFIX}_xgboost.pkl",
    f"{MODEL_PREFIX}_cnn_attention.pkl", 
    f"{MODEL_PREFIX}_autoencoder.pkl",
    f"{MODEL_PREFIX}_svm_rbf.pkl"
]

for model_file in model_files:
    print(f"  {os.path.basename(model_file)}: {os.path.exists(model_file)}")

if not all(os.path.exists(f) for f in model_files):
    print("❌ CHECKPOINT 1 FAILED: Model files missing")
    print("Run this first: cd src/triage_brain_v2/ && python ensemble_triage_brain.py")
    exit(1)
else:
    print("✅ CHECKPOINT 1 PASSED: All paths and model files exist")

# CHECKPOINT 2: V2 IMPORT
print("\n📍 CHECKPOINT 2: V2 IMPORT")
if not os.path.exists(V2_PATH):
    print("❌ CHECKPOINT 2 FAILED: V2 path doesn't exist")
    exit(1)

sys.path.insert(0, V2_PATH)
print(f"Added to sys.path: {V2_PATH}")

try:
    from ensemble_triage_brain import EnsembleTriageBrain, label_segment, preprocess_data
    print("✅ CHECKPOINT 2 PASSED: V2 imports successful")
except ImportError as e:
    print(f"❌ CHECKPOINT 2 FAILED: Import error: {e}")
    exit(1)

# CHECKPOINT 3: MODEL LOADING
print("\n📍 CHECKPOINT 3: MODEL LOADING")
try:
    ensemble = EnsembleTriageBrain()
    print("EnsembleTriageBrain created")
    
    ensemble.load(MODEL_PREFIX)
    print("✅ CHECKPOINT 3 PASSED: Models loaded successfully")
    
    # Verify models are loaded
    print("Loaded models:")
    for model_name in ensemble.models:
        print(f"  - {model_name}: {type(ensemble.models[model_name])}")
        
except Exception as e:
    print(f"❌ CHECKPOINT 3 FAILED: Model loading error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# CHECKPOINT 4: FEATURE ORDER VERIFICATION
print("\n📍 CHECKPOINT 4: FEATURE ORDER VERIFICATION")

# Try to get feature order from XGBoost model
try:
    xgb_model = ensemble.models['xgboost']
    
    # Check different ways to get feature names
    features = None
    if hasattr(xgb_model.model, 'feature_names_in_'):
        features = xgb_model.model.feature_names_in_
        print(f"Features from XGBoost model: {len(features)}")
    elif hasattr(xgb_model.scaler, 'feature_names_in_'):
        features = xgb_model.scaler.feature_names_in_
        print(f"Features from StandardScaler: {len(features)}")
    
    if features is not None:
        print("✅ CHECKPOINT 4 PASSED: Feature order retrieved")
        V2_FEATURE_ORDER = list(features)
        print("First 5 features:", V2_FEATURE_ORDER[:5])
    else:
        print("⚠️ CHECKPOINT 4 WARNING: Using hardcoded feature order")
        V2_FEATURE_ORDER = [
            'duration_s', 'num_samples', 'sample_rate_hz',
            'velocity_mean', 'velocity_std', 'velocity_min', 'velocity_max', 'velocity_range',
            'acceleration_mean', 'acceleration_std', 'acceleration_min', 'acceleration_max', 'acceleration_range',
            'jerk_mean', 'jerk_std', 'jerk_min', 'jerk_max', 'jerk_rms',
            'max_deceleration', 'deceleration_events', 'velocity_zero_crossings', 'acceleration_reversals',
            'motion_smoothness', 'jerk_per_second', 'accel_changes_per_second', 'distance_traveled'
        ]
        
except Exception as e:
    print(f"❌ CHECKPOINT 4 FAILED: Feature order error: {e}")
    exit(1)

# CHECKPOINT 5: TEST PREDICTION
print("\n📍 CHECKPOINT 5: TEST PREDICTION")

try:
    # Create test data with exact feature order
    test_data = {}
    feature_defaults = {
        'duration_s': 2.0, 'num_samples': 20, 'sample_rate_hz': 10.0,
        'velocity_mean': 5.0, 'velocity_std': 1.5, 'velocity_min': 2.0, 'velocity_max': 8.0, 'velocity_range': 6.0,
        'acceleration_mean': 0.5, 'acceleration_std': 2.0, 'acceleration_min': -5.0, 'acceleration_max': 5.0, 'acceleration_range': 10.0,
        'jerk_mean': 0.0, 'jerk_std': 15.0, 'jerk_min': -25.0, 'jerk_max': 25.0, 'jerk_rms': 15.0,
        'max_deceleration': -5.0, 'deceleration_events': 3, 'velocity_zero_crossings': 0, 'acceleration_reversals': 8,
        'motion_smoothness': 0.02, 'jerk_per_second': 7.5, 'accel_changes_per_second': 4.0, 'distance_traveled': 10.0
    }
    
    # Create test data in exact order
    for feat in V2_FEATURE_ORDER:
        test_data[feat] = feature_defaults.get(feat, 1.0)
    
    # Add metadata
    test_data.update({
        'start_frame': 100,
        'end_frame': 120,
        'comment': 'test_prediction',
        'clip_name': 'test_clip',
        'scene_id': 'test_scene',
        'total_frames': 1000,
        'fps': 10.0
    })
    
    # Test prediction pipeline
    test_df = pd.DataFrame([test_data])
    print(f"Test dataframe shape: {test_df.shape}")
    print(f"Test dataframe columns: {list(test_df.columns)}")
    
    # Preprocess
    X, y, processed_data = preprocess_data(test_df)
    print(f"Preprocessed X shape: {X.shape}")
    
    # Predict
    predictions = ensemble.predict(X)
    prediction = predictions[0]
    
    print(f"✅ CHECKPOINT 5 PASSED: Test prediction = {prediction:.4f}")
    
    # Test label_segment
    label = label_segment(test_data['comment'])
    print(f"Label segment result: {label}")
    
except Exception as e:
    print(f"❌ CHECKPOINT 5 FAILED: Prediction error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# CHECKPOINT 6: BATCH PREDICTION TEST
print("\n📍 CHECKPOINT 6: BATCH PREDICTION TEST")

try:
    # Create multiple test samples
    batch_data = []
    for i in range(5):
        sample = test_data.copy()
        sample['start_frame'] = 100 + i * 20
        sample['end_frame'] = 120 + i * 20
        sample['comment'] = f'batch_test_{i}'
        # Vary some features
        sample['velocity_mean'] = 5.0 + i * 0.5
        sample['jerk_rms'] = 15.0 + i * 2.0
        batch_data.append(sample)
    
    batch_df = pd.DataFrame(batch_data)
    print(f"Batch dataframe shape: {batch_df.shape}")
    
    # Preprocess batch
    X_batch, y_batch, processed_batch = preprocess_data(batch_df)
    print(f"Batch preprocessed X shape: {X_batch.shape}")
    
    # Predict batch
    batch_predictions = ensemble.predict(X_batch)
    print(f"Batch predictions: {batch_predictions}")
    
    print("✅ CHECKPOINT 6 PASSED: Batch prediction successful")
    
except Exception as e:
    print(f"❌ CHECKPOINT 6 FAILED: Batch prediction error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n🎉 ALL CHECKPOINTS PASSED!")
print("✅ V2 ensemble is working correctly")
print("✅ Ready to integrate with GUI")
print("\nNext step: Update ensemble_engine.py with working code")

# Export working configuration for ensemble_engine.py
print("\n📋 WORKING CONFIGURATION:")
print(f"PROJECT_ROOT = '{PROJECT_ROOT}'")
print(f"V2_PATH = '{V2_PATH}'") 
print(f"MODEL_PREFIX = '{MODEL_PREFIX}'")
print(f"V2_FEATURE_ORDER = {V2_FEATURE_ORDER}")