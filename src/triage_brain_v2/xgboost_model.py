import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

class XGBoostModel:
    def __init__(self, weight=0.3):
        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            scale_pos_weight=2,  # Handle imbalance
            random_state=42
        )
        self.scaler = StandardScaler()
        self.weight = weight
        self.is_trained = False

    def preprocess(self, X):
        """Scale features and handle NaNs."""
        X = X.fillna(0)  # Impute NaNs with 0
        X_scaled = self.scaler.fit_transform(X) if not self.is_trained else self.scaler.transform(X)
        return X_scaled

    def train(self, X, y):
        """Train the model."""
        X_scaled = self.preprocess(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        return self

    def predict(self, X):
        """Predict probabilities for ensemble."""
        X_scaled = self.preprocess(X)
        return self.model.predict_proba(X_scaled)[:, 1] * self.weight

    def evaluate(self, X, y):
        """Evaluate model performance."""
        X_scaled = self.preprocess(X)
        y_pred = self.model.predict(X_scaled)
        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0)
        }

    def save(self, path):
        """Save model and scaler."""
        with open(path, 'wb') as f:
            pickle.dump({'model': self.model, 'scaler': self.scaler, 'weight': self.weight}, f)

    @staticmethod
    def load(path):
        """Load model and scaler."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = XGBoostModel(weight=data['weight'])
        model.model = data['model']
        model.scaler = data['scaler']
        model.is_trained = True
        return model

if __name__ == "__main__":
    # Example usage
    data = pd.read_json('feature_vectors.jsonl', lines=True)
    features = [col for col in data.columns if col not in ['start_frame', 'end_frame', 'comment', 'clip_name', 'scene_id', 'total_frames', 'fps']]
    X = data[features]
    # Dummy labels for testing
    y = np.random.randint(0, 2, len(X))
    model = XGBoostModel()
    model.train(X, y)
    print(model.evaluate(X, y))