import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import torch.serialization

# Allowlist required classes for safe unpickling
torch.serialization.add_safe_globals([StandardScaler, np.core.multiarray._reconstruct, np.ndarray])

class Attention(nn.Module):
    def __init__(self, feature_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(feature_dim, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        scores = self.attention(x)
        weights = self.softmax(scores)
        return (x * weights).sum(dim=1)

class CNNAttentionModel(nn.Module):
    def __init__(self, input_dim, weight=0.15):
        super(CNNAttentionModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.attention = Attention(16)
        self.fc = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
        self.weight = weight

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.permute(0, 2, 1)  # For attention
        x = self.attention(x)
        x = self.fc(x)
        return self.sigmoid(x)

class CNNAttentionWrapper:
    def __init__(self, weight=0.15):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = StandardScaler()
        self.weight = weight
        self.is_trained = False

    def preprocess(self, X):
        X = X.fillna(0)
        X_scaled = self.scaler.fit_transform(X) if not self.is_trained else self.scaler.transform(X)
        return torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

    def train(self, X, y, epochs=50, batch_size=16):
        X_tensor = self.preprocess(X)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(self.device)
        self.model = CNNAttentionModel(input_dim=X.shape[1], weight=self.weight).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.BCELoss()

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for _ in range(epochs):
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
        self.is_trained = True
        return self

    def predict(self, X):
        X_tensor = self.preprocess(X)
        self.model.eval()
        with torch.no_grad():
            probs = self.model(X_tensor).cpu().numpy().flatten()
        return probs * self.weight

    def evaluate(self, X, y):
        X_tensor = self.preprocess(X)
        self.model.eval()
        with torch.no_grad():
            y_pred = (self.model(X_tensor).cpu().numpy() > 0.5).astype(int).flatten()
        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0)
        }

    def save(self, path):
        torch.save({
            'model_state': self.model.state_dict(),
            'scaler': self.scaler,
            'weight': self.weight
        }, path)

    @staticmethod
    def load(path):
        checkpoint = torch.load(path, weights_only=False)
        model = CNNAttentionWrapper(weight=checkpoint['weight'])
        model.model = CNNAttentionModel(input_dim=checkpoint['scaler'].mean_.shape[0], weight=checkpoint['weight']).to(model.device)
        model.model.load_state_dict(checkpoint['model_state'])
        model.scaler = checkpoint['scaler']
        model.is_trained = True
        return model

if __name__ == "__main__":
    data = pd.read_json('feature_vectors.jsonl', lines=True)
    features = [col for col in data.columns if col not in ['start_frame', 'end_frame', 'comment', 'clip_name', 'scene_id', 'total_frames', 'fps']]
    X = data[features]
    y = np.random.randint(0, 2, len(X))
    model = CNNAttentionWrapper()
    model.train(X, y)
    print(model.evaluate(X, y))