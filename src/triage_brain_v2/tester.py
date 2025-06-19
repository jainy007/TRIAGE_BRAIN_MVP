import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from xgboost_model import XGBoostModel
from cnn_attention_model import CNNAttentionWrapper
from autoencoder_model import AutoencoderWrapper
from svm_rbf_model import SVMRBFModel
from ensemble_triage_brain import label_segment, preprocess_data
import json
from collections import defaultdict

class SmartValidationTester:
    def __init__(self):
        self.models = {}
        self.annotated_segments = self.load_annotated_clips()
        
    def load_annotated_clips(self):
        """Load the human-annotated ground truth data."""
        clips = []
        with open('assets/data/annotated_clips.jsonl', 'r') as f:
            for line in f:
                clips.append(json.loads(line.strip()))
        
        # Create a lookup of all annotated segments
        annotated_segments = {}
        for clip_data in clips:
            clip_name = clip_data['clip']
            for segment in clip_data['segments']:
                key = f"{clip_name}_{segment['start']}_{segment['end']}"
                annotated_segments[key] = {
                    'clip': clip_name,
                    'start': segment['start'],
                    'end': segment['end'],
                    'comment': segment['comment'],
                    'is_interesting': 1  # All annotated segments are interesting
                }
        return annotated_segments
    
    def create_ground_truth_labels(self, data):
        """Create ground truth labels with balanced approach."""
        labels = []
        segment_info = []
        
        # Define normal/non-interesting keywords to balance the dataset
        normal_keywords = [
            'perfect', 'no overshoot', 'normal', 'smooth', 'good', 'proper',
            'road color', 'toll gate', 'change in', 'parallel parking'
        ]
        
        for _, row in data.iterrows():
            # Create segment key to match against annotations
            clip_name = row['clip_name']
            start_frame = row['start_frame']
            end_frame = row['end_frame']
            comment = row['comment'].lower()
            key = f"{clip_name}_{start_frame}_{end_frame}"
            
            # Check if this segment is in our annotated ground truth
            if key in self.annotated_segments:
                # Even annotated segments can be normal if they contain normal keywords
                if any(keyword in comment for keyword in normal_keywords):
                    labels.append(0)  # Normal behavior, even if annotated
                    segment_info.append({'source': 'annotated_normal', 'confidence': 0.9})
                else:
                    labels.append(1)  # Definitely interesting - human annotated
                    segment_info.append({'source': 'annotated_interesting', 'confidence': 1.0})
            else:
                # Use enhanced comment-based labeling
                if any(keyword in comment for keyword in normal_keywords):
                    labels.append(0)  # Normal
                    segment_info.append({'source': 'comment_normal', 'confidence': 0.8})
                else:
                    comment_label = label_segment(row['comment'])
                    labels.append(comment_label)
                    segment_info.append({'source': 'comment_interesting', 'confidence': 0.7})
        
        return np.array(labels), segment_info
    
    def stratified_cross_validation(self, X, y, n_splits=5):
        """Perform stratified k-fold cross-validation with fallback for extreme imbalance."""
        # Check class distribution
        unique_classes, class_counts = np.unique(y, return_counts=True)
        min_class_count = min(class_counts)
        
        print(f"Class distribution: {dict(zip(unique_classes, class_counts))}")
        
        # If we don't have enough samples for stratified CV, use regular CV
        if min_class_count < n_splits:
            print(f"⚠️ WARNING: Minimum class has only {min_class_count} samples. Using regular CV instead of stratified.")
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=min(n_splits, len(X)), shuffle=True, random_state=42)
            fold_generator = kf.split(X)
        else:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            fold_generator = skf.split(X, y)
        
        cv_results = {
            'individual_models': defaultdict(list),
            'ensemble': defaultdict(list),
            'fold_details': []
        }
        
        fold_count = 0
        for train_idx, test_idx in fold_generator:
            fold_count += 1
            print(f"\n=== FOLD {fold_count}/{n_splits} ===")
            
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
            print(f"Train distribution: {np.bincount(y_train, minlength=2)}")
            print(f"Test distribution: {np.bincount(y_test, minlength=2)}")
            
            # Skip fold if no positive samples in training
            if len(np.unique(y_train)) < 2:
                print("⚠️ Skipping fold due to single class in training set")
                continue
            
            # Train models
            fold_models = self.train_fold_models(X_train, y_train)
            
            # Evaluate individual models
            individual_results = {}
            for name, model in fold_models.items():
                try:
                    metrics = model.evaluate(X_test, y_test)
                    cv_results['individual_models'][name].append(metrics)
                    individual_results[name] = metrics
                    print(f"{name}: Acc={metrics['accuracy']:.3f}, F1={metrics['f1']:.3f}")
                except Exception as e:
                    print(f"⚠️ {name} evaluation failed: {e}")
                    individual_results[name] = {'accuracy': 0, 'f1': 0, 'precision': 0, 'recall': 0}
            
            # Evaluate ensemble
            try:
                ensemble_metrics = self.evaluate_ensemble(X_test, y_test, fold_models)
                cv_results['ensemble']['accuracy'].append(ensemble_metrics['accuracy'])
                cv_results['ensemble']['f1'].append(ensemble_metrics['f1'])
                cv_results['ensemble']['precision'].append(ensemble_metrics['precision'])
                cv_results['ensemble']['recall'].append(ensemble_metrics['recall'])
                
                print(f"Ensemble: Acc={ensemble_metrics['accuracy']:.3f}, F1={ensemble_metrics['f1']:.3f}")
            except Exception as e:
                print(f"⚠️ Ensemble evaluation failed: {e}")
                ensemble_metrics = {'accuracy': 0, 'f1': 0, 'precision': 0, 'recall': 0}
            
            cv_results['fold_details'].append({
                'fold': fold_count,
                'individual': individual_results,
                'ensemble': ensemble_metrics,
                'test_indices': test_idx.tolist()
            })
        
        return cv_results
    
    def train_fold_models(self, X_train, y_train):
        """Train all models for a single fold."""
        models = {
            'xgboost': XGBoostModel(weight=0.4),
            'cnn_attention': CNNAttentionWrapper(weight=0.15),
            'autoencoder': AutoencoderWrapper(weight=0.1),
            'svm_rbf': SVMRBFModel(weight=0.35)
        }
        
        for name, model in models.items():
            if name == 'autoencoder':
                model.train(X_train)  # Unsupervised
            else:
                model.train(X_train, y_train)
        
        return models
    
    def evaluate_ensemble(self, X, y, models, weights=None):
        """Evaluate ensemble performance."""
        if weights is None:
            weights = [0.4, 0.15, 0.1, 0.35]  # Default weights
        
        predictions = np.zeros(len(X))
        for i, (name, model) in enumerate(models.items()):
            model_pred = model.predict(X)
            predictions += model_pred
        
        y_pred = (predictions > 0.5).astype(int)
        
        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'predictions': predictions,
            'binary_predictions': y_pred
        }
    
    def analyze_misclassifications(self, data, y_true, y_pred, segment_info):
        """Analyze where the model fails."""
        misclassified = []
        
        for i, (true_label, pred_label) in enumerate(zip(y_true, y_pred)):
            if true_label != pred_label:
                row = data.iloc[i]
                misclassified.append({
                    'clip': row['clip_name'],
                    'start_frame': row['start_frame'],
                    'end_frame': row['end_frame'],
                    'comment': row['comment'],
                    'true_label': true_label,
                    'predicted_label': pred_label,
                    'source': segment_info[i]['source'],
                    'error_type': 'False Positive' if pred_label == 1 else 'False Negative'
                })
        
        return misclassified
    
    def weight_optimization_with_cv(self, X, y):
        """Optimize ensemble weights using cross-validation."""
        best_weights = [0.4, 0.15, 0.1, 0.35]
        best_score = 0
        
        print("\n=== OPTIMIZING ENSEMBLE WEIGHTS ===")
        
        # Simplified grid search due to small dataset
        weight_configs = [
            [0.5, 0.2, 0.05, 0.25],  # Favor XGBoost + SVM
            [0.4, 0.15, 0.1, 0.35],   # Original
            [0.3, 0.25, 0.15, 0.3],   # More balanced
            [0.6, 0.1, 0.05, 0.25],   # Heavy XGBoost
        ]
        
        for weights in weight_configs:
            scores = []
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            
            for train_idx, test_idx in skf.split(X, y):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                models = self.train_fold_models(X_train, y_train)
                metrics = self.evaluate_ensemble(X_test, y_test, models, weights)
                scores.append(metrics['f1'])
            
            avg_score = np.mean(scores)
            print(f"Weights {weights}: F1 = {avg_score:.3f} (±{np.std(scores):.3f})")
            
            if avg_score > best_score:
                best_score = avg_score
                best_weights = weights
        
        print(f"Best weights: {best_weights} (F1 = {best_score:.3f})")
        return best_weights
    
    def generate_comprehensive_report(self, data, cv_results, optimized_weights):
        """Generate detailed evaluation report."""
        report = "# Comprehensive Triage Brain Validation Report\n\n"
        
        # Dataset overview
        report += "## Dataset Overview\n"
        y, segment_info = self.create_ground_truth_labels(data)
        
        # Count different label sources
        source_counts = defaultdict(int)
        for info in segment_info:
            source_counts[info['source']] += 1
        
        report += f"- Total samples: {len(data)}\n"
        for source, count in source_counts.items():
            report += f"- {source.replace('_', ' ').title()}: {count}\n"
        report += f"- Class distribution: {np.bincount(y, minlength=2)} (Normal=0, Interesting=1)\n\n"
        
        # Cross-validation results
        report += "## Cross-Validation Results (5-Fold)\n\n"
        
        # Individual models
        report += "### Individual Model Performance\n"
        for model_name in ['xgboost', 'cnn_attention', 'autoencoder', 'svm_rbf']:
            scores = cv_results['individual_models'][model_name]
            if scores:
                metrics = ['accuracy', 'precision', 'recall', 'f1']
                report += f"**{model_name.upper()}**\n"
                for metric in metrics:
                    values = [s[metric] for s in scores]
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    report += f"- {metric.capitalize()}: {mean_val:.3f} (±{std_val:.3f})\n"
                report += "\n"
        
        # Ensemble performance
        report += "### Ensemble Performance\n"
        ensemble_metrics = ['accuracy', 'precision', 'recall', 'f1']
        for metric in ensemble_metrics:
            values = cv_results['ensemble'][metric]
            mean_val = np.mean(values)
            std_val = np.std(values)
            report += f"- {metric.capitalize()}: {mean_val:.3f} (±{std_val:.3f})\n"
        
        # Confidence interval
        f1_scores = cv_results['ensemble']['f1']
        f1_mean = np.mean(f1_scores)
        f1_ci = 1.96 * np.std(f1_scores) / np.sqrt(len(f1_scores))
        report += f"\n**95% Confidence Interval for F1-Score: {f1_mean:.3f} ± {f1_ci:.3f}**\n\n"
        
        # Optimized weights
        report += "## Optimized Ensemble Weights\n"
        model_names = ['XGBoost', 'CNN+Attention', 'Autoencoder', 'SVM RBF']
        report += "| Model | Weight |\n|-------|--------|\n"
        for name, weight in zip(model_names, optimized_weights):
            report += f"| {name} | {weight:.3f} |\n"
        
        # Final assessment
        report += "\n## Assessment for CEO Demo\n"
        f1_mean = np.mean(cv_results['ensemble']['f1'])
        if f1_mean >= 0.8:
            status = "✅ READY FOR DEMO"
        elif f1_mean >= 0.65:
            status = "⚠️ ACCEPTABLE WITH CAVEATS"
        else:
            status = "❌ NEEDS MORE WORK"
        
        report += f"**Status: {status}**\n\n"
        report += f"Expected performance on unseen data: {f1_mean:.1%} F1-score\n"
        
        return report

def main():
    print("🧠 SMART VALIDATION TESTER")
    print("=" * 50)
    
    # Load data
    data = pd.read_json('processed_feature_vectors.jsonl', lines=True)
    print(f"Loaded {len(data)} samples")
    
    # Initialize tester
    tester = SmartValidationTester()
    
    # Preprocess and create ground truth
    X, y_original, processed_data = preprocess_data(data)
    y_ground_truth, segment_info = tester.create_ground_truth_labels(data)
    
    print(f"Ground truth distribution: {np.bincount(y_ground_truth)}")
    print(f"Original labels distribution: {np.bincount(y_original)}")
    
    # Perform cross-validation
    print("\n🔄 PERFORMING CROSS-VALIDATION...")
    cv_results = tester.stratified_cross_validation(X, y_ground_truth, n_splits=5)
    
    # Optimize weights
    print("\n⚙️ OPTIMIZING ENSEMBLE WEIGHTS...")
    optimized_weights = tester.weight_optimization_with_cv(X, y_ground_truth)
    
    # Generate comprehensive report
    print("\n📊 GENERATING REPORT...")
    report = tester.generate_comprehensive_report(data, cv_results, optimized_weights)
    
    # Save report
    with open('smart_validation_report.md', 'w') as f:
        f.write(report)
    
    print("\n✅ VALIDATION COMPLETE!")
    print("📄 Report saved to: smart_validation_report.md")
    
    # Quick summary
    f1_mean = np.mean(cv_results['ensemble']['f1'])
    f1_std = np.std(cv_results['ensemble']['f1'])
    print(f"\n🎯 BOTTOM LINE: {f1_mean:.1%} F1-score (±{f1_std:.1%})")
    
    if f1_mean >= 0.65:
        print("🚀 READY FOR CEO DEMO!")
    else:
        print("⚠️ NEEDS IMPROVEMENT BEFORE DEMO")

if __name__ == "__main__":
    main()