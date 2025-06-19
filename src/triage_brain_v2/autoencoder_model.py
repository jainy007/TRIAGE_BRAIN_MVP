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

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16)
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        return self.decoder(latent)

class AutoencoderWrapper:
    def __init__(self, weight=0.1):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = StandardScaler()
        self.weight = weight
        self.is_trained = False
        self.recon_error_threshold = None

    def preprocess(self, X):
        X = X.fillna(0)
        X_scaled = self.scaler.fit_transform(X) if not self.is_trained else self.scaler.transform(X)
        return torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

    def train(self, X, y=None, epochs=100, batch_size=16):
        X_tensor = self.preprocess(X)
        self.model = Autoencoder(input_dim=X.shape[1]).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        dataset = torch.utils.data.TensorDataset(X_tensor, X_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for _ in range(epochs):
            for X_batch, _ in loader:
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, X_batch)
                loss.backward()
                optimizer.step()

        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(X_tensor)
            errors = ((X_tensor - reconstructed) ** 2).mean(dim=1).cpu().numpy()
            self.recon_error_threshold = np.percentile(errors, 70)  # Top 30% as anomaly
        self.is_trained = True
        return self

    def predict(self, X):
        X_tensor = self.preprocess(X)
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(X_tensor)
            errors = ((X_tensor - reconstructed) ** 2).mean(dim=1).cpu().numpy()
        scores = (errors - errors.min()) / (errors.max() - errors.min() + 1e-6)
        return scores * self.weight

    def evaluate(self, X, y):
        X_tensor = self.preprocess(X)
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(X_tensor)
            errors = ((X_tensor - reconstructed) ** 2).mean(dim=1).cpu().numpy()

        threshold = self.recon_error_threshold or np.percentile(errors, 70)
        y_pred = (errors > threshold).astype(int)
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
            'weight': self.weight,
            'recon_error_threshold': self.recon_error_threshold
        }, path)

    @staticmethod
    def load(path):
        checkpoint = torch.load(path, weights_only=False)

        model = AutoencoderWrapper(weight=checkpoint['weight'])
        input_dim = checkpoint['scaler'].mean_.shape[0]
        model.model = Autoencoder(input_dim=input_dim).to(model.device)
        model.model.load_state_dict(checkpoint['model_state'])
        model.scaler = checkpoint['scaler']
        model.recon_error_threshold = checkpoint.get('recon_error_threshold')
        model.is_trained = True
        return model