#!/usr/bin/env python3
"""
Simple Triage Brain Analysis - Basic behavioral pattern detection
without heavy ML dependencies.
"""

import json
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

def load_labeled_data(file_path: str) -> pd.DataFrame:
    """Load labeled feature vectors."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} labeled segments")
    return df

def analyze_behavioral_signatures(df: pd.DataFrame) -> Dict:
    """Analyze motion signatures for each behavior type."""
    
    # Key motion features for analysis
    motion_features = [
        'velocity_mean', 'velocity_std', 
        'acceleration_mean', 'acceleration_std',
        'jerk_rms', 'max_deceleration',
        'acceleration_reversals', 'motion_smoothness',
        'duration_s'
    ]
    
    behavioral_signatures = {}
    
    # Analyze each primary label
    for label in df['primary_label'].unique():
        if label == 'other':
            continue
            
        label_data = df[df['primary_label'] == label]
        
        if len(label_data) < 2:  # Need at least 2 samples
            continue
        
        signature = {
            'sample_count': len(label_data),
            'avg_duration': label_data['duration_s'].mean(),
            'motion_profile': {}
        }
        
        # Calculate statistics for each motion feature
        for feature in motion_features:
            if feature in label_data.columns:
                values = label_data[feature].replace([np.inf, -np.inf], np.nan).dropna()
                if len(values) > 0:
                    signature['motion_profile'][feature] = {
                        'mean': float(values.mean()),
                        'std': float(values.std()) if len(values) > 1 else 0.0,
                        'min': float(values.min()),
                        'max': float(values.max())
                    }
        
        # Identify distinctive characteristics
        signature['characteristics'] = identify_characteristics(signature['motion_profile'])
        
        behavioral_signatures[label] = signature
    
    return behavioral_signatures

def identify_characteristics(motion_profile: Dict) -> List[str]:
    """Identify key characteristics based on motion profile."""
    characteristics = []
    
    # High jerk (jerky motion)
    if 'jerk_rms' in motion_profile:
        jerk_mean = motion_profile['jerk_rms']['mean']
        if jerk_mean > 45:
            characteristics.append('high_jerk')
        elif jerk_mean < 30:
            characteristics.append('smooth_motion')
    
    # Strong deceleration
    if 'max_deceleration' in motion_profile:
        decel_mean = abs(motion_profile['max_deceleration']['mean'])
        if decel_mean > 15:
            characteristics.append('hard_braking')
    
    # High acceleration reversals (indecisive)
    if 'acceleration_reversals' in motion_profile and 'duration_s' in motion_profile:
        reversals_per_sec = (motion_profile['acceleration_reversals']['mean'] / 
                           motion_profile['duration_s']['mean'])
        if reversals_per_sec > 60:
            characteristics.append('indecisive')
    
    # Low motion smoothness
    if 'motion_smoothness' in motion_profile:
        smoothness = motion_profile['motion_smoothness']['mean']
        if smoothness < 0.025:
            characteristics.append('jerky')
        elif smoothness > 0.035:
            characteristics.append('very_smooth')
    
    # High velocity variation
    if 'velocity_std' in motion_profile:
        vel_std = motion_profile['velocity_std']['mean']
        if vel_std > 2.0:
            characteristics.append('variable_speed')
    
    # Long duration events
    if 'duration_s' in motion_profile:
        duration = motion_profile['duration_s']['mean']
        if duration > 15:
            characteristics.append('long_event')
        elif duration < 2:
            characteristics.append('brief_event')
    
    return characteristics

def find_behavioral_separability(df: pd.DataFrame) -> Dict:
    """Find which features best separate different behaviors."""
    
    motion_features = [
        'velocity_mean', 'velocity_std', 'jerk_rms', 
        'acceleration_reversals', 'motion_smoothness', 'duration_s'
    ]
    
    separability_analysis = {}
    
    for feature in motion_features:
        if feature not in df.columns:
            continue
            
        # Calculate between-class vs within-class variance
        feature_data = df[feature].replace([np.inf, -np.inf], np.nan).dropna()
        
        # Group by primary label
        groups = []
        labels = []
        
        for label in df['primary_label'].unique():
            if label == 'other':
                continue
            label_data = df[df['primary_label'] == label][feature].replace([np.inf, -np.inf], np.nan).dropna()
            if len(label_data) >= 2:
                groups.append(label_data.values)
                labels.append(label)
        
        if len(groups) < 2:
            continue
        
        # Simple separability metric: ratio of between-group to within-group variance
        all_values = np.concatenate(groups)
        overall_mean = np.mean(all_values)
        
        # Between-group variance
        between_var = 0
        total_samples = 0
        for group in groups:
            group_mean = np.mean(group)
            between_var += len(group) * (group_mean - overall_mean) ** 2
            total_samples += len(group)
        between_var /= (len(groups) - 1)
        
        # Within-group variance
        within_var = 0
        total_within_samples = 0
        for group in groups:
            if len(group) > 1:
                within_var += np.sum((group - np.mean(group)) ** 2)
                total_within_samples += len(group) - 1
        
        if total_within_samples > 0:
            within_var /= total_within_samples
            separability_score = between_var / (within_var + 1e-8)  # Add small constant to avoid division by zero
        else:
            separability_score = 0
        
        separability_analysis[feature] = {
            'separability_score': float(separability_score),
            'between_variance': float(between_var),
            'within_variance': float(within_var),
            'num_groups': len(groups)
        }
    
    return separability_analysis

def create_behavior_rules(behavioral_signatures: Dict) -> Dict:
    """Create simple rule-based classification rules."""
    
    rules = {}
    
    for behavior, signature in behavioral_signatures.items():
        motion = signature['motion_profile']
        characteristics = signature['characteristics']
        
        rule_conditions = []
        
        # Create rules based on distinctive features
        if 'jerk_rms' in motion:
            jerk_mean = motion['jerk_rms']['mean']
            jerk_std = motion['jerk_rms']['std']
            if jerk_mean > 40:
                rule_conditions.append(f"jerk_rms > {jerk_mean - jerk_std:.1f}")
        
        if 'duration_s' in motion:
            duration_mean = motion['duration_s']['mean']
            if duration_mean > 10:
                rule_conditions.append(f"duration_s > {duration_mean * 0.8:.1f}")
            elif duration_mean < 3:
                rule_conditions.append(f"duration_s < {duration_mean * 1.2:.1f}")
        
        if 'max_deceleration' in motion:
            decel_mean = motion['max_deceleration']['mean']
            if abs(decel_mean) > 10:
                rule_conditions.append(f"abs(max_deceleration) > {abs(decel_mean) * 0.8:.1f}")
        
        if rule_conditions:
            rules[behavior] = {
                'conditions': rule_conditions,
                'sample_count': signature['sample_count'],
                'characteristics': characteristics
            }
    
    return rules

def analyze_label_distribution(df: pd.DataFrame) -> Dict:
    """Analyze the distribution and quality of labels."""
    
    # Primary label distribution
    primary_dist = df['primary_label'].value_counts().to_dict()
    
    # Multi-label analysis
    all_labels = []
    for labels_list in df['all_labels']:
        all_labels.extend(labels_list)
    
    all_label_dist = dict(Counter(all_labels))
    
    # Label co-occurrence
    co_occurrence = defaultdict(int)
    for labels_list in df['all_labels']:
        if len(labels_list) > 1:
            for i in range(len(labels_list)):
                for j in range(i+1, len(labels_list)):
                    pair = tuple(sorted([labels_list[i], labels_list[j]]))
                    co_occurrence[pair] += 1
    
    return {
        'primary_distribution': primary_dist,
        'all_labels_distribution': all_label_dist,
        'multi_label_segments': len([x for x in df['all_labels'] if len(x) > 1]),
        'common_co_occurrences': dict(sorted(co_occurrence.items(), key=lambda x: x[1], reverse=True)[:10])
    }

def main():
    """Run the simple triage brain analysis."""
    
    print("🧠 SIMPLE TRIAGE BRAIN ANALYSIS")
    print("=" * 50)
    
    # Load data
    df = load_labeled_data("feature_vectors_labeled.jsonl")
    
    print(f"\n📊 DATASET OVERVIEW:")
    print(f"Total segments: {len(df)}")
    print(f"Unique behaviors: {df['primary_label'].nunique()}")
    print(f"Duration range: {df['duration_s'].min():.1f}s to {df['duration_s'].max():.1f}s")
    
    # Label distribution analysis
    print(f"\n🏷️ LABEL DISTRIBUTION ANALYSIS:")
    label_analysis = analyze_label_distribution(df)
    
    print("Primary label distribution:")
    for label, count in sorted(label_analysis['primary_distribution'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {label:<20}: {count:>2} segments")
    
    print(f"\nMulti-label segments: {label_analysis['multi_label_segments']}")
    
    if label_analysis['common_co_occurrences']:
        print("Common label combinations:")
        for (label1, label2), count in list(label_analysis['common_co_occurrences'].items())[:5]:
            print(f"  {label1} + {label2}: {count}")
    
    # Behavioral signature analysis
    print(f"\n🎯 BEHAVIORAL SIGNATURE ANALYSIS:")
    signatures = analyze_behavioral_signatures(df)
    
    for behavior, sig in sorted(signatures.items(), key=lambda x: x[1]['sample_count'], reverse=True):
        count = sig['sample_count']
        duration = sig['avg_duration']
        chars = ', '.join(sig['characteristics']) if sig['characteristics'] else 'normal'
        
        print(f"\n{behavior} ({count} samples, {duration:.1f}s avg):")
        print(f"  Characteristics: {chars}")
        
        # Show key motion features
        motion = sig['motion_profile']
        if 'jerk_rms' in motion:
            print(f"  Jerk intensity: {motion['jerk_rms']['mean']:.1f} ± {motion['jerk_rms']['std']:.1f}")
        if 'motion_smoothness' in motion:
            print(f"  Motion smoothness: {motion['motion_smoothness']['mean']:.3f}")
        if 'acceleration_reversals' in motion:
            reversals_per_sec = motion['acceleration_reversals']['mean'] / duration
            print(f"  Indecision rate: {reversals_per_sec:.1f} reversals/sec")
    
    # Feature separability analysis
    print(f"\n🔬 FEATURE SEPARABILITY ANALYSIS:")
    separability = find_behavioral_separability(df)
    
    print("Best features for behavior discrimination:")
    sorted_features = sorted(separability.items(), key=lambda x: x[1]['separability_score'], reverse=True)
    
    for feature, analysis in sorted_features[:8]:
        score = analysis['separability_score']
        print(f"  {feature:<25}: {score:.2f}")
    
    # Generate simple classification rules
    print(f"\n📋 SIMPLE CLASSIFICATION RULES:")
    rules = create_behavior_rules(signatures)
    
    for behavior, rule in rules.items():
        if rule['sample_count'] >= 3:  # Only show rules for behaviors with enough samples
            print(f"\n{behavior}:")
            for condition in rule['conditions']:
                print(f"  - {condition}")
    
    # Save analysis results
    analysis_results = {
        'dataset_summary': {
            'total_segments': len(df),
            'unique_behaviors': df['primary_label'].nunique(),
            'duration_range': [float(df['duration_s'].min()), float(df['duration_s'].max())]
        },
        'label_analysis': {
            'primary_distribution': label_analysis['primary_distribution'],
            'all_labels_distribution': label_analysis['all_labels_distribution'],
            'multi_label_segments': label_analysis['multi_label_segments'],
            # Convert tuple keys to strings for JSON serialization
            'common_co_occurrences': {f"{k[0]}+{k[1]}": v for k, v in label_analysis['common_co_occurrences'].items()}
        },
        'behavioral_signatures': signatures,
        'feature_separability': separability,
        'classification_rules': rules
    }
    
    with open('simple_triage_analysis_results.json', 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)
    
    print(f"\n✅ Analysis complete! Results saved to simple_triage_analysis_results.json")
    
    # Summary insights
    print(f"\n💡 KEY INSIGHTS:")
    print(f"• Most discriminative feature: {sorted_features[0][0]} (score: {sorted_features[0][1]['separability_score']:.2f})")
    print(f"• Most common behavior: {max(label_analysis['primary_distribution'].items(), key=lambda x: x[1])[0]}")
    print(f"• {len([b for b in signatures.values() if 'high_jerk' in b['characteristics']])} behaviors show high jerk patterns")
    print(f"• {len([b for b in signatures.values() if 'hard_braking' in b['characteristics']])} behaviors involve hard braking")

if __name__ == "__main__":
    main()