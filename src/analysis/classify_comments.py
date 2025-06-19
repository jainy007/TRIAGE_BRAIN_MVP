#!/usr/bin/env python3
"""
Classify human annotation comments into standardized single-word labels.
This creates consistent labels for ML training and behavioral analysis.
"""

import json
import re
from collections import Counter
from typing import Dict, List, Set

def define_classification_rules() -> Dict[str, Dict]:
    """
    Define classification rules mapping comment patterns to standardized labels.
    Each rule has keywords, patterns, and exclusions.
    """
    
    classification_rules = {
        'overshoot': {
            'keywords': ['overshoot', 'oversho'],
            'patterns': [r'.*overshoot.*', r'.*stop.*overshoot.*'],
            'description': 'Vehicle overshoots intended stopping point'
        },
        'nearmiss': {
            'keywords': ['near miss', 'nearmiss', 'near-miss'],
            'patterns': [r'.*near\s*miss.*'],
            'description': 'Close call with potential collision'
        },
        'hesitation': {
            'keywords': ['hesitation', 'hesitate', 'unnecessary halt', 'halt'],
            'patterns': [r'.*hesitat.*', r'.*unnecessary.*halt.*'],
            'description': 'Indecisive or overly cautious behavior'
        },
        'oversteering': {
            'keywords': ['oversteer', 'oversteering'],
            'patterns': [r'.*oversteer.*'],
            'description': 'Excessive steering input or correction'
        },
        'pedestrian': {
            'keywords': ['pedestrian', 'man crossing', 'women', 'man', 'people'],
            'patterns': [r'.*pedestrian.*', r'.*man\s+(crossing|in\s+horizon).*', r'.*women.*'],
            'description': 'Interaction with pedestrians'
        },
        'bicycle': {
            'keywords': ['bicycle', 'bike'],
            'patterns': [r'.*bicycle.*', r'.*bike.*'],
            'description': 'Interaction with cyclists'
        },
        'traffic_control': {
            'keywords': ['stop sign', 'traffic', 'red light'],
            'patterns': [r'.*stop\s+sign.*', r'.*traffic.*light.*', r'.*red\s+light.*'],
            'description': 'Traffic control device interaction'
        },
        'obstacles': {
            'keywords': ['traffic cones', 'cones', 'obstacle', 'car on road'],
            'patterns': [r'.*traffic\s+cones.*', r'.*cones.*', r'.*obstacle.*'],
            'description': 'Static obstacles or road furniture'
        },
        'vehicle_interaction': {
            'keywords': ['leading vehicle', 'car', 'bus', 'vehicle', 'overtaking'],
            'patterns': [r'.*leading\s+(vehicle|car).*', r'.*bus.*', r'.*overtaking.*', r'.*car.*ahead.*'],
            'description': 'Interaction with other vehicles'
        },
        'lane_change': {
            'keywords': ['lane change', 'swevel', 'merge'],
            'patterns': [r'.*lane\s+change.*', r'.*swevel.*', r'.*merge.*'],
            'description': 'Lane changing or merging maneuvers'
        },
        'turn': {
            'keywords': ['turn', 'left turn', 'right turn'],
            'patterns': [r'.*\b(left|right)\s+turn.*'],
            'description': 'Turning maneuvers'
        },
        'narrow_pass': {
            'keywords': ['narrow pass', 'narrow'],
            'patterns': [r'.*narrow\s+pass.*', r'.*narrow.*'],
            'description': 'Navigating through tight spaces'
        },
        'visibility': {
            'keywords': ['occlusion', 'dark to bright', 'blind'],
            'patterns': [r'.*occlusion.*', r'.*dark\s+to\s+bright.*', r'.*blind.*'],
            'description': 'Visibility or lighting challenges'
        },
        'road_conditions': {
            'keywords': ['road color', 'road surface'],
            'patterns': [r'.*road\s+(color|surface).*'],
            'description': 'Road surface or marking changes'
        },
        'parking': {
            'keywords': ['parking', 'parked'],
            'patterns': [r'.*parking.*', r'.*parked.*'],
            'description': 'Parking-related scenarios'
        },
        'risky_behavior': {
            'keywords': ['risky', 'nervous', 'confusion'],
            'patterns': [r'.*risky.*', r'.*nervous.*', r'.*confusion.*'],
            'description': 'Dangerous or erratic driving behavior'
        }
    }
    
    return classification_rules

def classify_comment(comment: str, rules: Dict[str, Dict]) -> List[str]:
    """
    Classify a single comment into one or more categories.
    
    Args:
        comment: The annotation comment text
        rules: Classification rules dictionary
    
    Returns:
        List of matching category labels
    """
    comment_lower = comment.lower().strip()
    matches = []
    
    for category, rule in rules.items():
        # Check keyword matches
        for keyword in rule['keywords']:
            if keyword.lower() in comment_lower:
                matches.append(category)
                break
        
        # Check pattern matches if no keyword match
        if category not in matches:
            for pattern in rule['patterns']:
                if re.match(pattern, comment_lower):
                    matches.append(category)
                    break
    
    # If no matches, classify as 'other'
    if not matches:
        matches = ['other']
    
    return matches

def analyze_comments_distribution(feature_data: List[Dict]) -> Dict:
    """
    Analyze the distribution of comments and classifications.
    """
    comments = [item['comment'] for item in feature_data]
    comment_counter = Counter(comments)
    
    rules = define_classification_rules()
    all_classifications = []
    
    for comment in comments:
        classifications = classify_comment(comment, rules)
        all_classifications.extend(classifications)
    
    classification_counter = Counter(all_classifications)
    
    return {
        'total_segments': len(comments),
        'unique_comments': len(comment_counter),
        'most_common_comments': comment_counter.most_common(10),
        'classification_distribution': classification_counter.most_common(),
        'rules_used': len(rules)
    }

def classify_feature_data(input_file: str, output_file: str) -> None:
    """
    Add classification labels to existing feature data.
    
    Args:
        input_file: Path to feature_vectors.jsonl
        output_file: Path to save enhanced feature vectors with classifications
    """
    # Load feature data
    feature_data = []
    with open(input_file, 'r') as f:
        for line in f:
            feature_data.append(json.loads(line))
    
    print(f"Loaded {len(feature_data)} feature vectors")
    
    # Load classification rules
    rules = define_classification_rules()
    
    # Classify each comment
    classified_data = []
    classification_stats = Counter()
    
    for item in feature_data:
        comment = item['comment']
        classifications = classify_comment(comment, rules)
        
        # Add classifications to the feature vector
        enhanced_item = item.copy()
        enhanced_item['primary_label'] = classifications[0]  # First match as primary
        enhanced_item['all_labels'] = classifications
        enhanced_item['label_count'] = len(classifications)
        
        classified_data.append(enhanced_item)
        classification_stats.update(classifications)
    
    # Save enhanced data
    with open(output_file, 'w') as f:
        for item in classified_data:
            f.write(json.dumps(item) + '\n')
    
    # Generate analysis report
    analysis = analyze_comments_distribution(feature_data)
    
    print(f"\n📊 CLASSIFICATION ANALYSIS:")
    print(f"Total segments: {analysis['total_segments']}")
    print(f"Unique comments: {analysis['unique_comments']}")
    print(f"Classification categories: {analysis['rules_used']}")
    
    print(f"\n🏷️ LABEL DISTRIBUTION:")
    for label, count in classification_stats.most_common():
        percentage = 100 * count / len(feature_data)
        print(f"  {label:<20}: {count:>3} ({percentage:>5.1f}%)")
    
    print(f"\n📝 MOST COMMON ORIGINAL COMMENTS:")
    for comment, count in analysis['most_common_comments'][:8]:
        print(f"  {count:>2}x: {comment[:60]}")
    
    print(f"\n✅ Enhanced feature vectors saved to: {output_file}")
    
    # Create a separate labels summary file
    labels_summary = {
        'classification_rules': rules,
        'label_distribution': dict(classification_stats),
        'total_segments': len(feature_data),
        'labels_created': len(classification_stats)
    }
    
    labels_file = output_file.replace('.jsonl', '_labels_summary.json')
    with open(labels_file, 'w') as f:
        json.dump(labels_summary, f, indent=2)
    
    print(f"📋 Labels summary saved to: {labels_file}")

if __name__ == "__main__":
    # Classify the extracted features
    input_file = "feature_vectors.jsonl"
    output_file = "feature_vectors_labeled.jsonl"
    
    classify_feature_data(input_file, output_file)