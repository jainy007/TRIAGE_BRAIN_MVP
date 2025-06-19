#!/usr/bin/env python3
"""
Analyze extracted motion features to understand behavioral signatures
and prepare data for Triage Brain model training.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter

def load_feature_data(file_path: str) -> pd.DataFrame:
    """Load feature vectors into a pandas DataFrame."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} feature vectors with {len(df.columns)} features each")
    return df

def analyze_feature_distributions(df: pd.DataFrame) -> Dict:
    """Analyze the distribution of motion features."""
    
    # Define feature groups
    motion_features = [
        'velocity_mean', 'velocity_std', 'velocity_min', 'velocity_max',
        'acceleration_mean', 'acceleration_std', 'acceleration_min', 'acceleration_max',
        'jerk_mean', 'jerk_std', 'jerk_min', 'jerk_max', 'jerk_rms'
    ]
    
    behavioral_features = [
        'max_deceleration', 'deceleration_events', 'velocity_zero_crossings',
        'acceleration_reversals', 'motion_smoothness', 'jerk_per_second',
        'accel_changes_per_second'
    ]
    
    temporal_features = [
        'duration_s', 'num_samples', 'sample_rate_hz', 'distance_traveled'
    ]
    
    analysis = {}
    
    # Analyze each feature group
    for group_name, features in [
        ('motion', motion_features),
        ('behavioral', behavioral_features), 
        ('temporal', temporal_features)
    ]:
        group_stats = {}
        for feature in features:
            if feature in df.columns:
                series = df[feature].replace([np.inf, -np.inf], np.nan).dropna()
                group_stats[feature] = {
                    'mean': series.mean(),
                    'std': series.std(),
                    'min': series.min(),
                    'max': series.max(),
                    'q25': series.quantile(0.25),
                    'q75': series.quantile(0.75),
                    'missing_count': df[feature].isna().sum()
                }
        analysis[group_name] = group_stats
    
    return analysis

def analyze_behavioral_patterns(df: pd.DataFrame) -> Dict:
    """Analyze patterns in behavioral comments."""
    
    # Comment analysis
    comment_stats = {
        'total_segments': len(df),
        'unique_comments': df['comment'].nunique(),
        'most_common': df['comment'].value_counts().head(10).to_dict(),
        'comment_lengths': {
            'mean': df['comment'].str.len().mean(),
            'min': df['comment'].str.len().min(),
            'max': df['comment'].str.len().max()
        }
    }
    
    # Duration analysis by behavior type
    duration_by_comment = df.groupby('comment')['duration_s'].agg(['mean', 'std', 'count']).round(2)
    
    # Motion intensity analysis
    df_clean = df.replace([np.inf, -np.inf], np.nan)
    
    motion_intensity = df_clean.groupby('comment').agg({
        'jerk_rms': 'mean',
        'velocity_std': 'mean', 
        'acceleration_reversals': 'mean',
        'motion_smoothness': 'mean'
    }).round(3)
    
    return {
        'comment_stats': comment_stats,
        'duration_by_behavior': duration_by_comment.to_dict(),
        'motion_intensity_by_behavior': motion_intensity.to_dict()
    }

def identify_behavioral_signatures(df: pd.DataFrame) -> Dict:
    """Identify distinctive motion signatures for different behaviors."""
    
    # Clean data
    df_clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['jerk_rms', 'velocity_std'])
    
    # Group by comment type (behaviors with 3+ samples)
    behavior_groups = df_clean.groupby('comment').filter(lambda x: len(x) >= 3)
    
    signatures = {}
    
    for behavior in behavior_groups['comment'].unique():
        behavior_data = behavior_groups[behavior_groups['comment'] == behavior]
        
        signature = {
            'sample_count': len(behavior_data),
            'avg_duration': behavior_data['duration_s'].mean(),
            'motion_profile': {
                'jerk_intensity': behavior_data['jerk_rms'].mean(),
                'velocity_variability': behavior_data['velocity_std'].mean(),
                'deceleration_strength': abs(behavior_data['max_deceleration'].mean()),
                'smoothness_score': behavior_data['motion_smoothness'].mean(),
                'stop_go_frequency': behavior_data['acceleration_reversals'].mean() / behavior_data['duration_s'].mean()
            },
            'key_characteristics': []
        }
        
        # Identify key characteristics
        if signature['motion_profile']['jerk_intensity'] > 40:
            signature['key_characteristics'].append('high_jerk')
        if signature['motion_profile']['deceleration_strength'] > 15:
            signature['key_characteristics'].append('hard_braking')
        if signature['motion_profile']['stop_go_frequency'] > 50:
            signature['key_characteristics'].append('indecisive')
        if signature['motion_profile']['smoothness_score'] < 0.025:
            signature['key_characteristics'].append('jerky_motion')
        
        signatures[behavior] = signature
    
    return signatures

def create_feature_correlation_analysis(df: pd.DataFrame) -> Dict:
    """Analyze correlations between motion features."""
    
    # Select numeric motion features
    motion_cols = [col for col in df.columns if any(x in col for x in 
                  ['velocity', 'acceleration', 'jerk', 'deceleration', 'smoothness'])]
    
    df_motion = df[motion_cols].replace([np.inf, -np.inf], np.nan).dropna()
    
    correlation_matrix = df_motion.corr()
    
    # Find strong correlations (>0.7 or <-0.7)
    strong_correlations = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_value = correlation_matrix.iloc[i, j]
            if abs(corr_value) > 0.7:
                strong_correlations.append({
                    'feature1': correlation_matrix.columns[i],
                    'feature2': correlation_matrix.columns[j],
                    'correlation': round(corr_value, 3)
                })
    
    return {
        'correlation_matrix': correlation_matrix.to_dict(),
        'strong_correlations': strong_correlations,
        'feature_count': len(motion_cols)
    }

def generate_analysis_report(input_file: str, output_dir: str = "analysis_output") -> None:
    """Generate comprehensive analysis report."""
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Load data
    df = load_feature_data(input_file)
    
    print("\n🔍 ANALYZING MOTION FEATURES...")
    
    # Run analyses
    feature_distributions = analyze_feature_distributions(df)
    behavioral_patterns = analyze_behavioral_patterns(df)
    behavioral_signatures = identify_behavioral_signatures(df)
    correlation_analysis = create_feature_correlation_analysis(df)
    
    # Compile comprehensive report
    full_report = {
        'dataset_summary': {
            'total_segments': len(df),
            'total_features': len(df.columns),
            'unique_behaviors': df['comment'].nunique(),
            'duration_range': [df['duration_s'].min(), df['duration_s'].max()],
            'clips_covered': df['clip_name'].nunique()
        },
        'feature_distributions': feature_distributions,
        'behavioral_patterns': behavioral_patterns,
        'behavioral_signatures': behavioral_signatures,
        'correlation_analysis': correlation_analysis
    }
    
    # Save detailed report
    with open(f"{output_dir}/motion_analysis_report.json", 'w') as f:
        json.dump(full_report, f, indent=2, default=str)
    
    # Print summary
    print(f"\n📊 MOTION FEATURE ANALYSIS SUMMARY:")
    print(f"Dataset: {len(df)} segments from {df['clip_name'].nunique()} clips")
    print(f"Behaviors: {df['comment'].nunique()} unique annotation types")
    print(f"Duration range: {df['duration_s'].min():.1f}s to {df['duration_s'].max():.1f}s")
    
    print(f"\n🎯 TOP BEHAVIORAL SIGNATURES:")
    sorted_behaviors = sorted(behavioral_signatures.items(), 
                            key=lambda x: x[1]['sample_count'], reverse=True)
    
    for behavior, signature in sorted_behaviors[:8]:
        count = signature['sample_count']
        duration = signature['avg_duration']
        jerk = signature['motion_profile']['jerk_intensity']
        characteristics = ', '.join(signature['key_characteristics']) or 'smooth'
        print(f"  {behavior[:35]:<35}: {count:>2} samples, {duration:>5.1f}s avg, jerk={jerk:>5.1f} [{characteristics}]")
    
    print(f"\n📈 MOTION FEATURE INSIGHTS:")
    motion_stats = feature_distributions['motion']
    print(f"  Velocity range: {motion_stats['velocity_mean']['min']:.1f} to {motion_stats['velocity_mean']['max']:.1f} m/s")
    print(f"  Jerk intensity: {motion_stats['jerk_rms']['min']:.1f} to {motion_stats['jerk_rms']['max']:.1f} m/s³")
    print(f"  Acceleration reversals: {feature_distributions['behavioral']['acceleration_reversals']['mean']:.1f} avg per segment")
    
    print(f"\n✅ Detailed analysis saved to: {output_dir}/motion_analysis_report.json")
    
    # Create CSV export for further analysis
    df.to_csv(f"{output_dir}/feature_vectors.csv", index=False)
    print(f"📄 CSV export saved to: {output_dir}/feature_vectors.csv")

if __name__ == "__main__":
    input_file = "feature_vectors.jsonl"
    generate_analysis_report(input_file)