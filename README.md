# Triage Brain - AV Behavioral Classification

**Automated behavioral analysis and risk detection for autonomous vehicle validation.**

## Overview

Triage Brain automatically identifies and classifies risky driving behaviors from AV sensor data. It extracts motion signatures from pose data and uses rule-based classification to detect:

- **Near misses** - Close collision calls
- **Overshoots** - Stop sign violations  
- **Hesitation** - Indecisive behavior
- **Pedestrian interactions** - Human safety events
- **Vehicle interactions** - Traffic scenarios
- **Anomalous behaviors** - Novel failure modes

## Project Structure

```
mvp3_0/
├── main.py                    # Main CLI runner
├── configs/
│   └── config.json           # Configuration settings
├── src/
│   ├── triage_brain/         # Core classification logic
│   ├── feature_engineering/  # Motion feature extraction
│   └── analysis/             # Data analysis tools
├── assets/
│   ├── models/              # Trained models
│   └── data/                # Input datasets
├── outputs/
│   ├── features/            # Extracted features
│   ├── analysis/            # Analysis results
│   └── reports/             # Training reports
├── tools/                   # Utility scripts
├── tests/                   # Unit tests
└── annotation_gui/          # Manual annotation interface
```

## Quick Start

### 1. Run Full Pipeline
```bash
python main.py pipeline --input assets/data/annotated_clips.jsonl --pose-dir /mnt/db/av_dataset/
```

### 2. Individual Steps

**Extract Features:**
```bash
python main.py extract-features --input assets/data/annotated_clips.jsonl --pose-dir /mnt/db/av_dataset/
```

**Train Model:**
```bash
python main.py train --input outputs/features/feature_vectors_labeled.jsonl
```

**Classify New Segment:**
```bash
python main.py classify --model assets/models/practical_triage_brain.json --features motion_data.json
```
**Test the trained Model**
```bash
# Test classifying a new segment
python main.py classify --model assets/models/practical_triage_brain.json --features outputs/features/feature_vectors_labeled.jsonl
```

## Key Results

From your 52 annotated segments:

- **12 Near Miss events** detected (23% of data)
- **8 Overshoot behaviors** identified  
- **Clear motion signatures** for each behavior type
- **>95% coverage** with rule-based classification
- **Anomaly detection** flags unknown behaviors

### Behavioral Signatures Discovered:

| Behavior | Samples | Avg Duration | Jerk Intensity | Key Pattern |
|----------|---------|--------------|----------------|-------------|
| Near Miss | 12 | 9.3s | 45.9 m/s³ | High jerk + hard braking |
| Overshoot | 8 | 5.5s | 48.3 m/s³ | Brief + intense braking |
| Pedestrian | 6 | 11.1s | 47.2 m/s³ | Long duration events |
| Hesitation | 4 | 10.6s | 42.5 m/s³ | Lower jerk + indecision |

## 🔧 Configuration

Edit `configs/config.json` to customize:

- **Data paths** - Pose data directory, model locations
- **Feature extraction** - Smoothing parameters, clipping thresholds  
- **Classification** - Behavior types, risk levels
- **Model training** - Sample requirements, anomaly detection

## Motion Features

The system extracts 20 motion features from 6DoF pose data:

**Velocity Features:** mean, std, min, max, range
**Acceleration Features:** mean, std, min, max, range  
**Jerk Features:** mean, std, RMS (motion smoothness indicator)
**Behavioral Features:** deceleration events, acceleration reversals, zero crossings
**Temporal Features:** duration, distance traveled, rates per second

## Classification Rules

**High-Risk Behaviors:**
- **Near Miss:** `jerk > 39.0` + `hard_braking > 14.8`
- **Overshoot:** `jerk > 46.6` + `hard_braking > 16.0`

**Medium-Risk Behaviors:**  
- **Hesitation:** `duration > 8.5s` + `jerk > 32.6`
- **Pedestrian:** `duration > 8.9s` + `jerk > 43.5`

**Anomaly Detection:**
- Statistical outliers beyond 2σ in multiple features
- Flags novel behaviors not seen in training

## Development

**Add New Behavior Type:**
1. Add to `configs/config.json` behavior_types
2. Update classification rules in `src/triage_brain/practical_triage_brain.py`
3. Retrain model with new annotated data

**Extend Features:**
1. Add feature extraction in `src/feature_engineering/extract_features.py`
2. Update feature list in config
3. Retrain classification rules

## Requirements

- Python 3.8+
- pandas, numpy, scipy
- Pose data in feather format
- Annotated clips in JSONL format

## Impact

This system enables:
- **Automated triage** of thousands of AV logs
- **Risk prioritization** for manual review
- **Behavioral pattern discovery** 
- **Continuous safety monitoring**
- **Scalable validation** workflows

---

**Built for AV safety validation at scale** 