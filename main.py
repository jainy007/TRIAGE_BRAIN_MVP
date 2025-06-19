#!/usr/bin/env python3
"""
Triage Brain - Main Runner
AV Validation Suite - Automated Behavioral Classification

Usage:
    python main.py extract-features --input annotated_clips.jsonl --pose-dir /path/to/pose/data
    python main.py train --input feature_vectors_labeled.jsonl
    python main.py analyze --input feature_vectors_labeled.jsonl  
    python main.py classify --model assets/models/practical_triage_brain.json --features motion_features.json
    python main.py pipeline --input annotated_clips.jsonl --pose-dir /path/to/pose/data
"""

import argparse
import sys
import json
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.feature_engineering.extract_features import extract_features_from_annotations
from src.feature_engineering.pose_extractor import load_pose_data
from src.triage_brain.practical_triage_brain import PracticalTriageBrain
from src.analysis.simple_triage_analysis import main as run_analysis
from src.analysis.classify_comments import classify_feature_data

class TriageBrainRunner:
    """Main runner for Triage Brain operations"""
    
    def __init__(self):
        self.config = self.load_config()
        
    def load_config(self):
        """Load configuration settings"""
        config_file = Path("configs/config.json")
        
        default_config = {
            "pose_data_dir": "/mnt/db/av_dataset/",
            "models_dir": "assets/models/",
            "data_dir": "assets/data/",
            "outputs_dir": "outputs/",
            "default_model": "practical_triage_brain.json"
        }
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def extract_features(self, input_file, pose_dir=None, output_file=None):
        """Extract motion features from annotated clips"""
        print("🔧 EXTRACTING MOTION FEATURES")
        print("=" * 50)
        
        pose_dir = pose_dir or self.config["pose_data_dir"]
        output_file = output_file or "outputs/features/feature_vectors.jsonl"
        
        # Ensure output directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        try:
            extract_features_from_annotations(input_file, pose_dir, output_file)
            print(f"✅ Features extracted to: {output_file}")
            return output_file
        except Exception as e:
            print(f"❌ Feature extraction failed: {e}")
            return None
    
    def classify_comments(self, input_file, output_file=None):
        """Classify human comments into standardized labels"""
        print("🏷️ CLASSIFYING COMMENTS")
        print("=" * 50)
        
        output_file = output_file or "outputs/features/feature_vectors_labeled.jsonl"
        
        # Ensure output directory exists  
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        try:
            classify_feature_data(input_file, output_file)
            print(f"✅ Comments classified and saved to: {output_file}")
            return output_file
        except Exception as e:
            print(f"❌ Comment classification failed: {e}")
            return None
    
    def train_model(self, input_file, model_output=None):
        """Train the Triage Brain model"""
        print("🧠 TRAINING TRIAGE BRAIN")
        print("=" * 50)
        
        model_output = model_output or f"{self.config['models_dir']}/practical_triage_brain.json"
        
        # Ensure output directory exists
        Path(model_output).parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Initialize and train
            tb = PracticalTriageBrain()
            results = tb.train(input_file)
            
            # Save model
            tb.save_model(model_output)
            
            # Save training results
            results_file = "outputs/reports/training_results.json"
            Path(results_file).parent.mkdir(parents=True, exist_ok=True)
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"✅ Model trained and saved to: {model_output}")
            print(f"📊 Training results saved to: {results_file}")
            
            return model_output
            
        except Exception as e:
            print(f"❌ Model training failed: {e}")
            return None
    
    def analyze_data(self, input_file):
        """Run comprehensive data analysis"""
        print("📊 ANALYZING BEHAVIORAL DATA")
        print("=" * 50)
        
        try:
            # This would call the analysis functions
            # For now, print instructions
            print(f"Run analysis on: {input_file}")
            print("Note: Run 'python src/analysis/simple_triage_analysis.py' directly for detailed analysis")
            return True
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            return False
    
    def classify_segment(self, model_file, features_input):
        """Classify a single driving segment"""
        print("🎯 CLASSIFYING DRIVING SEGMENT")
        print("=" * 50)
        
        try:
            # Load model
            tb = PracticalTriageBrain()
            tb.load_model(model_file)
            
            # Load features (expect JSON with motion features)
            if isinstance(features_input, str):
                with open(features_input, 'r') as f:
                    features = json.load(f)
            else:
                features = features_input
            
            # Convert to pandas Series for analysis
            import pandas as pd
            feature_series = pd.Series(features)
            
            # Analyze segment
            result = tb.analyze_segment(feature_series)
            
            print(f"🎯 Classification Results:")
            print(f"  Predicted Behavior: {result['behavior_classification']['predicted_behavior']}")
            print(f"  Confidence: {result['behavior_classification']['confidence']:.2f}")
            print(f"  Risk Level: {result['risk_level']}")
            print(f"  Anomaly Score: {result['anomaly_analysis']['anomaly_score']:.2f}")
            print(f"  Characteristics: {', '.join(result['characteristics'])}")
            
            return result
            
        except Exception as e:
            print(f"❌ Classification failed: {e}")
            return None
    
    def run_full_pipeline(self, input_file, pose_dir=None):
        """Run the complete Triage Brain pipeline"""
        print("🚀 RUNNING FULL TRIAGE BRAIN PIPELINE")
        print("=" * 60)
        
        # Step 1: Extract features
        print("\n📍 Step 1: Extracting Motion Features...")
        features_file = self.extract_features(input_file, pose_dir)
        
        if not features_file:
            print("❌ Pipeline failed at feature extraction")
            return False
        
        # Step 2: Classify comments
        print("\n📍 Step 2: Classifying Comments...")
        labeled_file = self.classify_comments(features_file)
        
        if not labeled_file:
            print("❌ Pipeline failed at comment classification")
            return False
        
        # Step 3: Train model
        print("\n📍 Step 3: Training Triage Brain...")
        model_file = self.train_model(labeled_file)
        
        if not model_file:
            print("❌ Pipeline failed at model training")
            return False
        
        # Step 4: Run analysis
        print("\n📍 Step 4: Running Analysis...")
        analysis_success = self.analyze_data(labeled_file)
        
        if not analysis_success:
            print("❌ Pipeline failed at analysis")
            return False
        
        print("\n🎉 PIPELINE COMPLETE!")
        print(f"📁 Model saved to: {model_file}")
        print(f"📁 Labeled data: {labeled_file}")
        print(f"📁 Check outputs/ directory for all results")
        
        return True
    
    def load_model(self, model_file=None):
        """Load a trained Triage Brain model"""
        model_file = model_file or f"{self.config['models_dir']}/{self.config['default_model']}"
        
        if not Path(model_file).exists():
            print(f"❌ Model file not found: {model_file}")
            return None
        
        try:
            tb = PracticalTriageBrain()
            tb.load_model(model_file)
            print(f"✅ Model loaded from: {model_file}")
            return tb
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return None

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="Triage Brain - AV Behavioral Classification")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Extract features command
    extract_parser = subparsers.add_parser('extract-features', help='Extract motion features from annotations')
    extract_parser.add_argument('--input', required=True, help='Input annotated clips file')
    extract_parser.add_argument('--pose-dir', help='Directory containing pose data')
    extract_parser.add_argument('--output', help='Output features file')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train Triage Brain model')
    train_parser.add_argument('--input', required=True, help='Input labeled features file')
    train_parser.add_argument('--output', help='Output model file')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze behavioral data')
    analyze_parser.add_argument('--input', required=True, help='Input features file')
    
    # Classify command
    classify_parser = subparsers.add_parser('classify', help='Classify a driving segment')
    classify_parser.add_argument('--model', help='Model file to use')
    classify_parser.add_argument('--features', required=True, help='Motion features JSON file')
    
    # Pipeline command
    pipeline_parser = subparsers.add_parser('pipeline', help='Run full pipeline')
    pipeline_parser.add_argument('--input', required=True, help='Input annotated clips file')
    pipeline_parser.add_argument('--pose-dir', help='Directory containing pose data')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize runner
    runner = TriageBrainRunner()
    
    # Execute command
    if args.command == 'extract-features':
        runner.extract_features(args.input, args.pose_dir, args.output)
        
    elif args.command == 'train':
        runner.train_model(args.input, args.output)
        
    elif args.command == 'analyze':
        runner.analyze_data(args.input)
        
    elif args.command == 'classify':
        runner.classify_segment(args.model, args.features)
        
    elif args.command == 'pipeline':
        runner.run_full_pipeline(args.input, args.pose_dir)

if __name__ == "__main__":
    main()