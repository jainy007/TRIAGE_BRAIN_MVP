#!/usr/bin/env python3
"""
Fixed DL Trainer - Tensor shape and gradient issues resolved
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
import os
from typing import Dict, List, Optional
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from collections import Counter

def get_optimal_device():
    """Automatically detect best available device"""
    
    if not torch.cuda.is_available():
        print("🖥️  CUDA not available, using CPU")
        return 'cpu'
    
    try:
        device = 'cuda'
        test_tensor = torch.tensor([1.0]).to(device)
        test_tensor = test_tensor + 1
        
        print(f"✅ GPU compatible: {torch.cuda.get_device_name(0)}")
        return device
        
    except RuntimeError as e:
        if "no kernel image is available" in str(e):
            print("❌ GPU not compatible with PyTorch version")
            print("🖥️  Falling back to CPU training")
            return 'cpu'
        else:
            raise e

# Enhanced Behavior Taxonomy
class EnhancedBehaviorTaxonomy:
    def __init__(self):
        self.behavior_mapping = {
            'stop_sign_overshoot': ['stop sign overshoot', 'stop overshoot', 'stop sine overshoot'],
            'traffic_cone_avoidance': ['traffic cones', 'traffic cones avoided', 'traffic cones in horizon'],
            'pedestrian_near_miss': ['man crossing', 'pedestrian crossing', 'women on ego lane', 'pedestrian running'],
            'vehicle_near_miss': ['near miss', 'risky launch - near miss', 'narrow near miss'],
            'bicycle_interaction': ['bicycle', 'bicycle crossed', 'letting bicycle pass'],
            'oversteering': ['oversteering', 'turn oversteering', 'oversteer near miss'],
            'intersection_hesitation': ['protected left hesitation', 'right turn hesitation', 'unnecessary hesitation'],
            'parked_vehicle_avoidance': ['car parallel parking', 'parked car', 'courier vehicle parked'],
            'following_hesitation': ['unnecessary halt', 'nervous stop'],
            'leading_vehicle_issues': ['misalligned leading vehicle', 'leading car backed up'],
            'visibility_issues': ['dark to bright occlusion', 'blending horizon color'],
            'aggressive_overtaking': ['disregards lead vehicle and overtakes', 'vehicles overtaking'],
            'lane_deviation': ['swevel to the left', 'swevel to the right', 'narrow pass'],
            'emergency_vehicle': ['police leading vehicle'],
            'unknown_behavior': ['unknown object ahead']
        }
        
        self.risk_levels = {
            'stop_sign_overshoot': 'HIGH', 'traffic_cone_avoidance': 'MEDIUM',
            'pedestrian_near_miss': 'CRITICAL', 'vehicle_near_miss': 'CRITICAL',
            'bicycle_interaction': 'MEDIUM', 'oversteering': 'HIGH',
            'intersection_hesitation': 'MEDIUM', 'parked_vehicle_avoidance': 'MEDIUM',
            'following_hesitation': 'LOW', 'leading_vehicle_issues': 'MEDIUM',
            'visibility_issues': 'MEDIUM', 'aggressive_overtaking': 'HIGH',
            'lane_deviation': 'MEDIUM', 'emergency_vehicle': 'MEDIUM',
            'unknown_behavior': 'MEDIUM'
        }
    
    def map_comment_to_behavior(self, comment: str) -> str:
        comment_lower = comment.lower().strip()
        
        for behavior, keywords in self.behavior_mapping.items():
            for keyword in keywords:
                if keyword.lower() in comment_lower:
                    return behavior
        
        return 'unknown_behavior'
    
    def get_risk_level(self, behavior: str) -> str:
        return self.risk_levels.get(behavior, 'MEDIUM')

# Fixed Dataset class
class SimpleFrameDataset(Dataset):
    """Simplified dataset to avoid complex frame extraction issues"""
    
    def __init__(self, feature_vectors_file: str, annotations_file: str, sequence_length: int = 8):
        self.sequence_length = sequence_length
        self.taxonomy = EnhancedBehaviorTaxonomy()
        
        # Load existing feature vectors
        self.segments = self._load_segments(feature_vectors_file)
        self.annotations = self._load_annotations(annotations_file)
        
        # Enhanced labeling
        self._enhance_labels()
        
        # Create sequences
        self.sequences = self._create_sequences()
        
        # Setup encoders
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self._fit_encoders()
        
        print(f"📊 Dataset Created:")
        print(f"   Segments loaded: {len(self.segments)}")
        print(f"   Training sequences: {len(self.sequences)}")
    
    def _load_segments(self, file_path: str) -> List[Dict]:
        segments = []
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    segments.append(json.loads(line))
        except FileNotFoundError:
            print(f"❌ Feature file not found: {file_path}")
            return []
        return segments
    
    def _load_annotations(self, file_path: str) -> Dict:
        annotations = {}
        with open(file_path, 'r') as f:
            for line in f:
                clip_data = json.loads(line)
                clip_name = clip_data['clip']
                annotations[clip_name] = clip_data['segments']
        return annotations
    
    def _enhance_labels(self):
        """Add enhanced behavior labels to segments"""
        
        for segment in self.segments:
            comment = segment.get('comment', '')
            behavior_class = self.taxonomy.map_comment_to_behavior(comment)
            risk_level = self.taxonomy.get_risk_level(behavior_class)
            
            segment['behavior_class'] = behavior_class
            segment['risk_level'] = risk_level
    
    def _create_sequences(self) -> List[Dict]:
        """Create training sequences from segments"""
        
        # Group by clip and behavior
        clips = {}
        for segment in self.segments:
            clip_name = segment['clip_name']
            behavior = segment['behavior_class']
            key = f"{clip_name}_{behavior}"
            
            if key not in clips:
                clips[key] = []
            clips[key].append(segment)
        
        sequences = []
        
        for key, segments in clips.items():
            # Ensure we have enough segments
            while len(segments) < self.sequence_length:
                segments.extend(segments[:self.sequence_length - len(segments)])
            
            # Create overlapping sequences
            for i in range(0, len(segments) - self.sequence_length + 1, max(1, self.sequence_length // 2)):
                sequence_segments = segments[i:i + self.sequence_length]
                
                # Use majority behavior
                behaviors = [s['behavior_class'] for s in sequence_segments]
                sequence_behavior = max(set(behaviors), key=behaviors.count)
                
                sequences.append({
                    'segments': sequence_segments,
                    'behavior': sequence_behavior,
                    'clip_key': key
                })
        
        return sequences
    
    def _fit_encoders(self):
        """Fit encoders on data"""
        
        # Motion feature names from your data
        self.feature_names = [
            'velocity_mean', 'velocity_std', 'velocity_min', 'velocity_max', 'velocity_range',
            'acceleration_mean', 'acceleration_std', 'acceleration_min', 'acceleration_max', 'acceleration_range',
            'jerk_mean', 'jerk_std', 'jerk_min', 'jerk_max', 'jerk_rms',
            'max_deceleration', 'motion_smoothness', 'acceleration_reversals',
            'velocity_zero_crossings', 'deceleration_events', 'jerk_per_second',
            'accel_changes_per_second', 'distance_traveled', 'duration_s'
        ]
        
        # Collect behaviors and features
        behaviors = [seq['behavior'] for seq in self.sequences]
        all_features = []
        
        for seq in self.sequences:
            for segment in seq['segments']:
                feature_vector = []
                for feature_name in self.feature_names:
                    value = segment.get(feature_name, 0.0)
                    if np.isnan(value) or np.isinf(value):
                        value = 0.0
                    feature_vector.append(float(value))
                all_features.append(feature_vector)
        
        # Fit encoders
        if behaviors:
            unique_behaviors = sorted(list(set(behaviors)))
            self.label_encoder.fit(unique_behaviors)
            
            print(f"📊 Behavior Classes ({len(unique_behaviors)}):")
            for i, behavior in enumerate(unique_behaviors):
                count = behaviors.count(behavior)
                risk = self.taxonomy.get_risk_level(behavior)
                print(f"   {i:2d}: {behavior:<25} ({count:2d} sequences, {risk})")
        
        if all_features:
            self.scaler.fit(all_features)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # Extract feature matrix
        feature_matrix = []
        for segment in sequence['segments']:
            feature_vector = []
            for feature_name in self.feature_names:
                value = segment.get(feature_name, 0.0)
                if np.isnan(value) or np.isinf(value):
                    value = 0.0
                feature_vector.append(float(value))
            feature_matrix.append(feature_vector)
        
        # Scale features
        feature_matrix = np.array(feature_matrix)
        feature_matrix = self.scaler.transform(feature_matrix)
        features_tensor = torch.FloatTensor(feature_matrix)
        
        # Encode label
        behavior_idx = self.label_encoder.transform([sequence['behavior']])[0]
        label_tensor = torch.LongTensor([behavior_idx])[0]
        
        # Risk score
        risk_mapping = {'LOW': 0.1, 'MEDIUM': 0.4, 'HIGH': 0.7, 'CRITICAL': 0.9}
        risk_level = self.taxonomy.get_risk_level(sequence['behavior'])
        risk_score = risk_mapping.get(risk_level, 0.4)
        risk_tensor = torch.FloatTensor([risk_score])[0]
        
        return {
            'features': features_tensor,
            'label': label_tensor,
            'risk_score': risk_tensor,
            'info': sequence['clip_key']
        }

# Fixed Model
class FixedTemporalNet(nn.Module):
    """Fixed model with proper tensor shapes"""
    
    def __init__(self, input_features=24, sequence_length=8, num_behaviors=15, hidden_dim=48):
        super().__init__()
        
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(input_features, 24, kernel_size=3, padding=1),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Conv1d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.lstm = nn.LSTM(
            input_size=48,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_behaviors)
        )
        
        self.risk_regressor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, return_attention=False):
        batch_size, seq_len, features = x.shape
        
        # Temporal convolution
        x_conv = x.transpose(1, 2)
        conv_features = self.temporal_conv(x_conv)
        conv_features = conv_features.transpose(1, 2)
        
        # LSTM
        lstm_out, _ = self.lstm(conv_features)
        
        # Attention
        attention_scores = self.attention(lstm_out)
        attention_weights = F.softmax(attention_scores, dim=1)
        attended = torch.sum(lstm_out * attention_weights, dim=1)
        
        # Predictions
        behavior_logits = self.classifier(attended)
        risk_score = self.risk_regressor(attended).squeeze(-1)  # FIX: Remove last dimension
        
        if return_attention:
            return behavior_logits, risk_score, attention_weights.squeeze(-1)
        return behavior_logits, risk_score
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# Fixed Trainer
class FixedTrainer:
    """Fixed trainer with proper loss handling"""
    
    def __init__(self, model, device='cuda'):
        self.device = get_optimal_device()  # Auto-detect device
        self.model = model.to(self.device)
        
        self.classification_loss = nn.CrossEntropyLoss()
        self.regression_loss = nn.MSELoss()
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=0.001,
            weight_decay=1e-4
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.7
        )
    
    def train(self, dataloader, epochs=30):
        """Fixed training loop"""
        
        print(f"🎯 TRAINING ENHANCED DL MODEL")
        print(f"   Device: {self.device}")
        print(f"   Batches per epoch: {len(dataloader)}")
        print(f"   Epochs: {epochs}")
        
        best_loss = float('inf')
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            correct = 0
            total = 0
            
            for batch in dataloader:
                features = batch['features'].to(self.device)
                labels = batch['label'].to(self.device)
                risk_scores = batch['risk_score'].to(self.device)
                
                # Forward pass
                behavior_logits, risk_pred = self.model(features)
                
                # Losses - FIXED tensor shapes
                class_loss = self.classification_loss(behavior_logits, labels)
                
                # Ensure risk tensors have same shape
                if risk_pred.dim() == 0:  # scalar
                    risk_pred = risk_pred.unsqueeze(0)
                if risk_scores.dim() == 0:  # scalar
                    risk_scores = risk_scores.unsqueeze(0)
                
                risk_loss = self.regression_loss(risk_pred, risk_scores)
                
                # Combined loss
                total_batch_loss = 0.7 * class_loss + 0.3 * risk_loss
                
                # Ensure total_batch_loss requires grad
                if not total_batch_loss.requires_grad:
                    print(f"Warning: Loss doesn't require grad at epoch {epoch}")
                    continue
                
                # Backward pass
                self.optimizer.zero_grad()
                total_batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                # Statistics
                total_loss += total_batch_loss.item()
                _, predicted = torch.max(behavior_logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            # Epoch summary
            avg_loss = total_loss / len(dataloader)
            accuracy = 100 * correct / total
            
            print(f"   Epoch {epoch+1:2d}/{epochs}: "
                  f"Loss={avg_loss:.4f} "
                  f"Acc={accuracy:.1f}%")
            
            self.scheduler.step(avg_loss)
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.model.state_dict(), 'best_fixed_model.pth')
                print(f"   💾 New best model saved!")
        
        print(f"✅ Training complete! Best loss: {best_loss:.4f}")
    
    def test_traffic_cone_clip(self, dataset):
        """Test on traffic cone scenario"""
        
        print(f"\n🎯 TESTING ON TRAFFIC CONE SCENARIO")
        
        # Find traffic cone sequences
        cone_sequences = []
        for i, seq_info in enumerate(dataset.sequences):
            if 'traffic_cone' in seq_info['behavior']:
                cone_sequences.append((i, seq_info))
        
        if not cone_sequences:
            print("   ❌ No traffic cone sequences found")
            # Test first sequence instead
            if len(dataset.sequences) > 0:
                cone_sequences = [(0, dataset.sequences[0])]
        
        self.model.eval()
        
        for seq_idx, seq_info in cone_sequences[:2]:  # Test first 2
            data_item = dataset[seq_idx]
            
            with torch.no_grad():
                features = data_item['features'].unsqueeze(0).to(self.device)
                
                behavior_logits, risk_pred, attention = self.model(features, return_attention=True)
                
                probs = F.softmax(behavior_logits, dim=-1)
                predicted_idx = torch.argmax(probs).item()
                predicted_behavior = dataset.label_encoder.classes_[predicted_idx]
                confidence = torch.max(probs).item()
                risk_score = risk_pred.item()
                
                print(f"\n   📊 SEQUENCE ANALYSIS:")
                print(f"      Sequence: {seq_info['clip_key']}")
                print(f"      True behavior: {seq_info['behavior']}")
                print(f"      Predicted: {predicted_behavior}")
                print(f"      Confidence: {confidence:.3f}")
                print(f"      Risk score: {risk_score:.3f}")
                
                if predicted_behavior == seq_info['behavior']:
                    print(f"      ✅ CORRECT PREDICTION!")
                elif 'traffic_cone' in predicted_behavior:
                    print(f"      ✅ CORRECTLY IDENTIFIED CONE SCENARIO")

def main():
    """Main training pipeline"""
    
    print("🧠 FIXED DL TRAINING WITH YOUR MOTION DATA")
    print("=" * 60)
    
    # Create dataset from your existing feature vectors
    dataset = SimpleFrameDataset(
        feature_vectors_file="assets/data/feature_vectors.jsonl",
        annotations_file="assets/data/annotated_clips.jsonl",
        sequence_length=8
    )
    
    if len(dataset) == 0:
        print("❌ No sequences created")
        return
    
    # Data loader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True)
    
    # Create model
    num_behaviors = len(dataset.label_encoder.classes_)
    num_features = len(dataset.feature_names)
    
    model = FixedTemporalNet(
        input_features=num_features,
        sequence_length=8,
        num_behaviors=num_behaviors,
        hidden_dim=48
    )
    
    param_count = model.count_parameters()
    
    print(f"\n📊 MODEL CONFIGURATION:")
    print(f"   Input features: {num_features}")
    print(f"   Behavior classes: {num_behaviors}")
    print(f"   Parameters: {param_count:,}")
    print(f"   Training sequences: {len(dataset)}")
    
    # Train
    trainer = FixedTrainer(model)
    trainer.train(dataloader, epochs=40)
    
    # Test
    trainer.test_traffic_cone_clip(dataset)
    
    print(f"\n🎉 FIXED TRAINING COMPLETE!")
    print(f"   Model saved as: best_fixed_model.pth")

if __name__ == "__main__":
    main()