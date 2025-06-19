import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost_model import XGBoostModel
from cnn_attention_model import CNNAttentionWrapper
from autoencoder_model import AutoencoderWrapper
from svm_rbf_model import SVMRBFModel
from sklearn.metrics import accuracy_score
import json
from io import StringIO
import torch
import torch.serialization
from sklearn.preprocessing import StandardScaler

# Allowlist required classes for safe unpickling
torch.serialization.add_safe_globals([StandardScaler, np.core.multiarray._reconstruct, np.ndarray])

def label_segment(comment):
    """Label segments as interesting (1) or normal (0) based on comment."""
    interesting_keywords = [
        'near miss', 'overshoot', 'hesitation', 'oversteer', 'unnecessary', 'risky', 
        'crossing', 'occlusion', 'obstacle', 'nervous', 'avoided', 'confusion'
    ]
    return 1 if any(keyword in comment.lower() for keyword in interesting_keywords) else 0

def preprocess_data(data):
    """Preprocess feature vectors."""
    data = data[data['duration_s'] > 0].copy()
    features = [col for col in data.columns if col not in ['start_frame', 'end_frame', 'comment', 'clip_name', 'scene_id', 'total_frames', 'fps']]
    X = data[features]
    X = X.fillna(0)
    y = np.array([label_segment(comment) for comment in data['comment']])
    return X, y, data

class EnsembleTriageBrain:
    def __init__(self):
        self.models = {
            'xgboost': XGBoostModel(weight=0.4),
            'cnn_attention': CNNAttentionWrapper(weight=0.15),
            'autoencoder': AutoencoderWrapper(weight=0.1),
            'svm_rbf': SVMRBFModel(weight=0.35)
        }
        self.weights = [0.4, 0.15, 0.1, 0.35]

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        for name, model in self.models.items():
            print(f"Training {name}...")
            if name == 'autoencoder':
                model.train(X_train)
            else:
                model.train(X_train, y_train)
            metrics = model.evaluate(X_test, y_test)
            print(f"{name} metrics: {metrics}")
        return X_train, X_test, y_train, y_test

    def predict(self, X):
        predictions = np.zeros(len(X))
        for model in self.models.values():
            predictions += model.predict(X)
        return (predictions > 0.5).astype(int)

    def save(self, path_prefix):
        for name, model in self.models.items():
            model.save(f"{path_prefix}_{name}.pkl")

    def load(self, path_prefix):
        self.models['xgboost'] = XGBoostModel.load(f"{path_prefix}_xgboost.pkl")
        self.models['cnn_attention'] = CNNAttentionWrapper.load(f"{path_prefix}_cnn_attention.pkl")
        self.models['autoencoder'] = AutoencoderWrapper.load(f"{path_prefix}_autoencoder.pkl")
        self.models['svm_rbf'] = SVMRBFModel.load(f"{path_prefix}_svm_rbf.pkl")

if __name__ == "__main__":
    # Load and validate JSON
    with open('assets/data/feature_vectors.jsonl', 'r') as f:
        lines = f.readlines()
    valid_json = []
    for i, line in enumerate(lines):
        try:
            json.loads(line.strip())
            valid_json.append(line)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON at line {i+1}: {line.strip()[:50]}... Error: {e}")
    data = pd.read_json(StringIO(''.join(valid_json)), lines=True)
    
    # Preprocess data
    X, y, processed_data = preprocess_data(data)
    
    # Initialize and train ensemble
    ensemble = EnsembleTriageBrain()
    X_train, X_test, y_train, y_test = ensemble.train(X, y)
    
    # Save models
    ensemble.save("triage_brain_model")
    
    # Example prediction
    y_pred = ensemble.predict(X_test)
    print("Ensemble test accuracy:", accuracy_score(y_test, y_pred))
    
    # Save processed data for tester
    processed_data.to_json('processed_feature_vectors.jsonl', orient='records', lines=True)