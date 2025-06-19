#!/usr/bin/env python3
"""
Triage Brain Video Analysis GUI
Real-time video analysis with ensemble model predictions and timeline annotations
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
import cv2
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import deque
from multiprocessing import Pool, cpu_count
from functools import partial
import pickle
import warnings
warnings.filterwarnings('ignore')

# Try to import scipy for filtering
try:
    from scipy.signal import savgol_filter
    SCIPY_AVAILABLE = True
except ImportError:
    print("⚠️ scipy not available - filtering disabled")
    SCIPY_AVAILABLE = False

# Add the triage_brain modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'triage_brain'))

try:
    from ensemble_triage_brain import EnsembleTriageBrain
except ImportError:
    print("⚠️ Ensemble model not found - running in demo mode")
    EnsembleTriageBrain = None

class VideoPlayer:
    """Video player component with timeline controls"""
    
    def __init__(self, parent, width=640, height=480):
        self.parent = parent
        self.width = width
        self.height = height
        
        # Video state
        self.video_capture = None
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 10  # Argoverse typical FPS
        self.is_playing = False
        self.frame_data = []
        
        # UI Components
        self.setup_ui()
        
    def setup_ui(self):
        """Setup video player UI"""
        # Video frame
        self.video_frame = tk.Frame(self.parent)
        self.video_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Canvas for video display
        self.canvas = tk.Canvas(self.video_frame, width=self.width, height=self.height, bg='black')
        self.canvas.pack(pady=5)
        
        # Controls frame
        controls_frame = tk.Frame(self.video_frame)
        controls_frame.pack(fill=tk.X, pady=5)
        
        # Playback controls
        self.play_button = tk.Button(controls_frame, text="▶", command=self.toggle_playback, width=3)
        self.play_button.pack(side=tk.LEFT, padx=2)
        
        self.stop_button = tk.Button(controls_frame, text="⏹", command=self.stop_video, width=3)
        self.stop_button.pack(side=tk.LEFT, padx=2)
        
        # Timeline
        self.timeline_var = tk.DoubleVar()
        self.timeline = tk.Scale(
            controls_frame, 
            from_=0, to=100, 
            orient=tk.HORIZONTAL, 
            variable=self.timeline_var,
            command=self.on_timeline_change,
            length=400
        )
        self.timeline.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        
        # Frame info
        self.frame_label = tk.Label(controls_frame, text="Frame: 0/0")
        self.frame_label.pack(side=tk.RIGHT, padx=5)
        
    def load_video_from_scene(self, scene_path: str, camera_view: str = "ring_front_center"):
        """Load video frames from Argoverse scene directory"""
        try:
            camera_path = os.path.join(scene_path, "sensors", "cameras", camera_view)
            
            if not os.path.exists(camera_path):
                raise FileNotFoundError(f"Camera path not found: {camera_path}")
            
            # Get all image files
            image_files = sorted([f for f in os.listdir(camera_path) 
                                if f.endswith(('.jpg', '.jpeg', '.png'))])
            
            if not image_files:
                raise FileNotFoundError(f"No image files found in {camera_path}")
            
            self.frame_data = [os.path.join(camera_path, f) for f in image_files]
            self.total_frames = len(self.frame_data)
            self.current_frame = 0
            
            # Update timeline
            self.timeline.config(to=self.total_frames-1)
            
            # Load first frame
            self.display_frame(0)
            
            print(f"✅ Loaded {self.total_frames} frames from {camera_view}")
            return True
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load video: {e}")
            return False
    
    def display_frame(self, frame_idx: int):
        """Display a specific frame"""
        if not self.frame_data or frame_idx >= len(self.frame_data):
            return
        
        try:
            # Load image
            image = cv2.imread(self.frame_data[frame_idx])
            if image is None:
                return
                
            # Resize to fit canvas
            image = cv2.resize(image, (self.width, self.height))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL and then to PhotoImage
            pil_image = Image.fromarray(image_rgb)
            self.photo = ImageTk.PhotoImage(pil_image)
            
            # Update canvas
            self.canvas.delete("all")
            self.canvas.create_image(
                self.width//2, self.height//2, 
                image=self.photo, anchor=tk.CENTER
            )
            
            # Update frame info
            self.current_frame = frame_idx
            self.frame_label.config(text=f"Frame: {frame_idx+1}/{self.total_frames}")
            self.timeline_var.set(frame_idx)
            
            # Notify motion graphs of position change
            if hasattr(self.parent, 'parent') and hasattr(self.parent.parent, 'motion_graphs'):
                self.parent.parent.motion_graphs.highlight_timeline_position(frame_idx, self.total_frames)
            
        except Exception as e:
            print(f"Error displaying frame {frame_idx}: {e}")
    
    def toggle_playback(self):
        """Toggle play/pause"""
        self.is_playing = not self.is_playing
        self.play_button.config(text="⏸" if self.is_playing else "▶")
        
        if self.is_playing:
            self.play_video()
    
    def play_video(self):
        """Play video frames"""
        if not self.is_playing or not self.frame_data:
            return
            
        if self.current_frame < self.total_frames - 1:
            self.display_frame(self.current_frame + 1)
            # Schedule next frame (100ms = 10 FPS)
            self.parent.after(100, self.play_video)
        else:
            self.is_playing = False
            self.play_button.config(text="▶")
    
    def stop_video(self):
        """Stop video and reset to beginning"""
        self.is_playing = False
        self.play_button.config(text="▶")
        self.display_frame(0)
    
    def on_timeline_change(self, value):
        """Handle timeline scrubbing"""
        if not self.is_playing:  # Only allow scrubbing when paused
            frame_idx = int(float(value))
            self.display_frame(frame_idx)

class MotionGraphs:
    """Real-time motion analysis graphs"""
    
    def __init__(self, parent):
        self.parent = parent
        self.motion_data = None
        self.setup_ui()
        
    def setup_ui(self):
        """Setup motion graphs UI"""
        self.graphs_frame = tk.Frame(self.parent)
        self.graphs_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(8, 6), dpi=80, facecolor='white')
        self.fig.suptitle('Motion Analysis', fontsize=14, fontweight='bold')
        
        # Create subplots
        self.ax1 = self.fig.add_subplot(3, 1, 1)
        self.ax2 = self.fig.add_subplot(3, 1, 2)
        self.ax3 = self.fig.add_subplot(3, 1, 3)
        
        # Setup axes
        self.ax1.set_title('Velocity', fontsize=10)
        self.ax1.set_ylabel('m/s')
        self.ax1.grid(True, alpha=0.3)
        
        self.ax2.set_title('Acceleration', fontsize=10)
        self.ax2.set_ylabel('m/s²')
        self.ax2.grid(True, alpha=0.3)
        
        self.ax3.set_title('Jerk', fontsize=10)
        self.ax3.set_ylabel('m/s³')
        self.ax3.set_xlabel('Time (s)')
        self.ax3.grid(True, alpha=0.3)
        
        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, self.graphs_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        plt.tight_layout()
        
    def load_motion_data(self, pose_path: str):
        """Load motion data from pose file"""
        try:
            # Load pose data (feather format)
            pose_df = pd.read_feather(pose_path)
            
            # Calculate motion metrics
            self.motion_data = self.calculate_motion_metrics(pose_df)
            
            # Update graphs
            self.update_graphs()
            
            print(f"✅ Loaded motion data: {len(self.motion_data)} samples")
            return True
            
        except Exception as e:
            print(f"❌ Error loading motion data: {e}")
            return False
    
    def calculate_motion_metrics(self, pose_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate velocity, acceleration, and jerk from pose data with filtering"""
        
        # Extract position data (assuming standard Argoverse format)
        if 'tx_m' in pose_df.columns and 'ty_m' in pose_df.columns:
            x = pose_df['tx_m'].values
            y = pose_df['ty_m'].values
        else:
            # Fallback: use any position columns found
            pos_cols = [col for col in pose_df.columns if any(p in col.lower() for p in ['tx', 'ty', 'x', 'y', 'pos'])]
            if len(pos_cols) >= 2:
                x = pose_df[pos_cols[0]].values
                y = pose_df[pos_cols[1]].values
            else:
                # Generate synthetic data for demo
                t = np.linspace(0, 32, len(pose_df))
                x = np.cumsum(np.random.normal(0, 0.1, len(t))) + 5 * t
                y = np.cumsum(np.random.normal(0, 0.1, len(t)))
        
        # Calculate time vector
        dt = 0.1  # 10 Hz typical for Argoverse
        time = np.arange(len(x)) * dt
        
        # Calculate velocity
        dx = np.gradient(x, dt)
        dy = np.gradient(y, dt)
        velocity = np.sqrt(dx**2 + dy**2)
        
        # Calculate acceleration
        acceleration = np.gradient(velocity, dt)
        
        # Calculate jerk
        jerk = np.gradient(acceleration, dt)
        
        # Apply Savitzky-Golay filtering to reduce noise
        if SCIPY_AVAILABLE:
            try:
                # Filter parameters
                window_length = min(11, len(velocity) // 4)  # Adaptive window
                if window_length % 2 == 0:  # Must be odd
                    window_length += 1
                window_length = max(5, window_length)  # Minimum window
                
                polyorder = min(3, window_length - 1)
                
                # Apply filtering
                velocity_filtered = savgol_filter(velocity, window_length, polyorder)
                acceleration_filtered = savgol_filter(acceleration, window_length, polyorder)
                jerk_filtered = savgol_filter(jerk, window_length, polyorder)
                
                print(f"✅ Applied Savitzky-Golay filter (window: {window_length}, order: {polyorder})")
                
            except Exception as e:
                print(f"⚠️ Filtering failed, using raw data: {e}")
                velocity_filtered = velocity
                acceleration_filtered = acceleration
                jerk_filtered = jerk
        else:
            # Simple moving average as fallback
            window = 5
            velocity_filtered = np.convolve(velocity, np.ones(window)/window, mode='same')
            acceleration_filtered = np.convolve(acceleration, np.ones(window)/window, mode='same')
            jerk_filtered = np.convolve(jerk, np.ones(window)/window, mode='same')
            print(f"✅ Applied simple moving average filter (window: {window})")
        
        # Create motion dataframe
        motion_df = pd.DataFrame({
            'time': time,
            'velocity': velocity_filtered,
            'acceleration': acceleration_filtered,
            'jerk': jerk_filtered,
            'x': x,
            'y': y
        })
        
        return motion_df
    
    def update_graphs(self):
        """Update motion graphs"""
        if self.motion_data is None:
            return
        
        # Clear axes
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        
        time = self.motion_data['time']
        
        # Plot velocity
        self.ax1.plot(time, self.motion_data['velocity'], 'b-', linewidth=1.5)
        self.ax1.set_title('Velocity', fontsize=10)
        self.ax1.set_ylabel('m/s')
        self.ax1.grid(True, alpha=0.3)
        
        # Plot acceleration
        self.ax2.plot(time, self.motion_data['acceleration'], 'g-', linewidth=1.5)
        self.ax2.set_title('Acceleration', fontsize=10)
        self.ax2.set_ylabel('m/s²')
        self.ax2.grid(True, alpha=0.3)
        
        # Plot jerk
        self.ax3.plot(time, self.motion_data['jerk'], 'r-', linewidth=1.5)
        self.ax3.set_title('Jerk', fontsize=10)
        self.ax3.set_ylabel('m/s³')
        self.ax3.set_xlabel('Time (s)')
        self.ax3.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def add_cluster_overlays(self, top_clusters):
        """Add bounding box overlays for top risk clusters"""
        
        if not top_clusters or self.motion_data is None:
            return
        
        print(f"🎨 Adding {len(top_clusters)} cluster overlays to graphs...")
        
        # Clear previous overlays
        self.clear_overlays()
        
        # Color mapping for different behaviors
        behavior_colors = {
            'overshoot': ('red', 0.3),
            'nearmiss': ('orange', 0.3), 
            'oversteering': ('purple', 0.3),
            'hesitation': ('yellow', 0.3),
            'pedestrian': ('pink', 0.3),
            'bicycle': ('cyan', 0.3),
            'vehicle_interaction': ('brown', 0.3),
            'traffic_control': ('magenta', 0.3),
            'road_conditions': ('gray', 0.3),
            'other': ('lightgray', 0.2)
        }
        
        max_time = self.motion_data['time'].max()
        
        for i, cluster in enumerate(top_clusters):
            # Convert frame indices to time
            start_time = (cluster['start_frame'] / len(self.motion_data)) * max_time
            end_time = (cluster['end_frame'] / len(self.motion_data)) * max_time
            
            behavior = cluster['classification']
            color, alpha = behavior_colors.get(behavior, ('red', 0.3))
            
            # Add overlay to all three subplots
            for ax in [self.ax1, self.ax2, self.ax3]:
                y_min, y_max = ax.get_ylim()
                
                # Create rectangle
                rect = Rectangle(
                    (start_time, y_min),
                    end_time - start_time,
                    y_max - y_min,
                    facecolor=color,
                    alpha=alpha,
                    edgecolor=color,
                    linewidth=2,
                    label=f"Cluster {i+1}: {behavior}"
                )
                
                ax.add_patch(rect)
                
                # Add text label (only on top graph to avoid clutter)
                if ax == self.ax1:
                    mid_time = (start_time + end_time) / 2
                    label_text = f"{behavior}\n{cluster['final_score']:.2f}"
                    
                    ax.text(
                        mid_time, y_max * 0.9,
                        label_text,
                        ha='center', va='top',
                        fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.7),
                        color='white' if color in ['red', 'purple', 'brown'] else 'black'
                    )
        
        # Add legend (only to bottom graph)
        if top_clusters:
            legend_elements = []
            for i, cluster in enumerate(top_clusters):
                behavior = cluster['classification']
                color, _ = behavior_colors.get(behavior, ('red', 0.3))
                legend_elements.append(
                    Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.5, 
                             label=f"{behavior} ({cluster['final_score']:.2f})")
                )
            
            self.ax3.legend(handles=legend_elements, loc='upper right', fontsize=8)
        
        # Refresh canvas
        self.canvas.draw()
        
        print(f"✅ Cluster overlays added successfully")
    
    def highlight_timeline_position(self, frame_idx: int, total_frames: int):
        """Highlight current position on timeline"""
        if self.motion_data is None:
            return
        
        # Calculate time position
        time_pos = (frame_idx / total_frames) * self.motion_data['time'].max()
        
        # Add vertical line to all plots
        for ax in [self.ax1, self.ax2, self.ax3]:
            # Remove previous timeline markers
            for line in ax.lines:
                if hasattr(line, '_timeline_marker'):
                    line.remove()
            
            # Add new timeline marker
            line = ax.axvline(x=time_pos, color='blue', linestyle='--', alpha=0.8, linewidth=2)
            line._timeline_marker = True
        
        self.canvas.draw()
    
    def clear_overlays(self):
        """Clear all overlay patches from graphs"""
        
        for ax in [self.ax1, self.ax2, self.ax3]:
            # Remove rectangles and text annotations
            for patch in ax.patches[:]:
                patch.remove()
            
            for text in ax.texts[:]:
                if hasattr(text, '_cluster_label'):
                    text.remove()
            
            # Clear legend
            if ax.legend_:
                ax.legend_.remove()

class EnsemblePanel:
    """Real-time ensemble prediction panel"""
    
    def __init__(self, parent):
        self.parent = parent
        self.ensemble_model = None
        self.setup_ui()
        self.load_ensemble_model()
        
    def setup_ui(self):
        """Setup ensemble prediction UI"""
        self.panel_frame = tk.LabelFrame(self.parent, text="Ensemble Analysis", font=('Arial', 12, 'bold'))
        self.panel_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Prediction display
        pred_frame = tk.Frame(self.panel_frame)
        pred_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Behavior prediction
        tk.Label(pred_frame, text="Behavior:", font=('Arial', 10, 'bold')).grid(row=0, column=0, sticky='w')
        self.behavior_label = tk.Label(pred_frame, text="Not analyzed", font=('Arial', 10), fg='blue')
        self.behavior_label.grid(row=0, column=1, sticky='w', padx=10)
        
        # Confidence
        tk.Label(pred_frame, text="Confidence:", font=('Arial', 10, 'bold')).grid(row=1, column=0, sticky='w')
        self.confidence_label = tk.Label(pred_frame, text="0.000", font=('Arial', 10))
        self.confidence_label.grid(row=1, column=1, sticky='w', padx=10)
        
        # Risk level
        tk.Label(pred_frame, text="Risk Level:", font=('Arial', 10, 'bold')).grid(row=2, column=0, sticky='w')
        self.risk_label = tk.Label(pred_frame, text="UNKNOWN", font=('Arial', 10, 'bold'), fg='gray')
        self.risk_label.grid(row=2, column=1, sticky='w', padx=10)
        
        # Strategy
        tk.Label(pred_frame, text="Strategy:", font=('Arial', 9)).grid(row=3, column=0, sticky='w')
        self.strategy_label = tk.Label(pred_frame, text="No strategy", font=('Arial', 9), fg='gray')
        self.strategy_label.grid(row=3, column=1, sticky='w', padx=10)
        
        # Individual model predictions
        individual_frame = tk.LabelFrame(self.panel_frame, text="Individual Models")
        individual_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.individual_text = tk.Text(individual_frame, height=4, width=50, font=('Courier', 8))
        self.individual_text.pack(padx=5, pady=5)
        
    def load_ensemble_model(self):
        """Load the ensemble model"""
        try:
            if EnsembleTriageBrain is None:
                self.individual_text.insert(tk.END, "⚠️ Running in demo mode - models not loaded\n")
                return
                
            self.ensemble_model = EnsembleTriageBrain()
            
            # Try to load models with correct paths
            model_paths = {
                'advanced': "assets/models/advanced_triage_brain.pkl",
                'simple': "assets/models/simple_ml_triage.pkl",
                'rule': "practical_triage_brain.json"
            }
            
            # Check if model files exist
            missing_models = []
            for name, path in model_paths.items():
                if not os.path.exists(path):
                    missing_models.append(f"{name}: {path}")
            
            if missing_models:
                self.individual_text.insert(tk.END, "⚠️ Some models missing:\n")
                for model in missing_models:
                    self.individual_text.insert(tk.END, f"  • {model}\n")
                # Continue with demo mode but set ensemble_model to None for proper fallback
                self.ensemble_model = None
            else:
                self.ensemble_model.load_models(
                    model_paths['advanced'], 
                    model_paths['simple'], 
                    model_paths['rule']
                )
                self.individual_text.insert(tk.END, "✅ All ensemble models loaded successfully!\n")
                
        except Exception as e:
            self.individual_text.insert(tk.END, f"❌ Error loading models: {e}\n")
            self.ensemble_model = None
    
    def analyze_current_segment(self):
        """Analyze the current video segment"""
        # This is a placeholder - in real implementation, 
        # you would extract features from the current video segment
        
        # Simulate feature extraction
        features_dict = {
            'jerk_rms': np.random.uniform(30, 60),
            'velocity_mean': np.random.uniform(-5, 15),
            'max_deceleration': np.random.uniform(-25, -5),
            'duration_s': np.random.uniform(2, 10),
            'motion_smoothness': np.random.uniform(0.01, 0.1),
            'acceleration_reversals': np.random.randint(10, 100),
            'velocity_std': np.random.uniform(1, 5),
            'deceleration_events': np.random.randint(5, 25),
            'jerk_per_second': np.random.uniform(50, 200),
            'accel_changes_per_second': np.random.uniform(50, 150)
        }
        
        features_array = np.array([
            features_dict['jerk_rms'], features_dict['velocity_mean'], 
            features_dict['velocity_std'], -50, 50, -0.5, 12.0, -20, 20,
            -6.0, 37.0, 37.1, features_dict['max_deceleration'], 
            features_dict['deceleration_events'], 2, features_dict['acceleration_reversals'], 
            features_dict['motion_smoothness'], features_dict['jerk_per_second'], 
            features_dict['accel_changes_per_second'], features_dict['duration_s'], -1.35
        ])
        
        try:
            if self.ensemble_model is None:
                # Demo mode
                result = {
                    'ensemble_analysis': {
                        'ensemble_prediction': 'demo_behavior',
                        'confidence': np.random.uniform(0.2, 0.8),
                        'strategy': 'demo_mode',
                        'individual_predictions': {
                            'advanced_ml': {'predicted_behavior': 'hesitation', 'confidence': 0.187},
                            'simple_ml': {'predicted_behavior': 'other', 'confidence': 0.271},
                            'rule_based': {'predicted_behavior': 'overshoot', 'confidence': 0.550}
                        }
                    },
                    'risk_assessment': {'risk_level': 'MEDIUM'}
                }
            else:
                # Real analysis
                result = self.ensemble_model.analyze_segment(features_array, features_dict)
            
            self.update_display(result)
            
        except Exception as e:
            self.individual_text.delete(1.0, tk.END)
            self.individual_text.insert(tk.END, f"❌ Analysis failed: {e}\n")
    
    def update_display(self, result: Dict):
        """Update the ensemble prediction display"""
        
        ensemble = result['ensemble_analysis']
        
        # Update main prediction
        self.behavior_label.config(text=ensemble['ensemble_prediction'])
        self.confidence_label.config(text=f"{ensemble['confidence']:.3f}")
        
        # Update risk level with color coding
        risk_level = result['risk_assessment']['risk_level']
        self.risk_label.config(text=risk_level)
        
        risk_colors = {
            'LOW': 'green',
            'MEDIUM': 'orange', 
            'HIGH': 'red',
            'CRITICAL': 'darkred'
        }
        self.risk_label.config(fg=risk_colors.get(risk_level, 'gray'))
        
        # Update strategy
        strategy = ensemble.get('strategy', 'unknown')
        self.strategy_label.config(text=strategy)
        
        # Update individual predictions
        self.individual_text.delete(1.0, tk.END)
        self.individual_text.insert(tk.END, "Individual Model Predictions:\n")
        
        for model, pred in ensemble['individual_predictions'].items():
            if 'error' not in pred:
                behavior = pred['predicted_behavior']
                conf = pred['confidence']
                self.individual_text.insert(tk.END, f"  {model:12}: {behavior:12} ({conf:.3f})\n")
            else:
                self.individual_text.insert(tk.END, f"  {model:12}: ERROR - {pred['error']}\n")

class FrameAnalyzer:
    """Frame-level analysis engine with multiprocessing support"""
    
    def __init__(self, ensemble_model):
        self.ensemble_model = ensemble_model
        self.risk_multipliers = {
            'LOW': 0.25,
            'MEDIUM': 0.5, 
            'HIGH': 0.75,
            'CRITICAL': 1.0
        }
    
    def extract_frame_features(self, motion_data, frame_idx):
        """Extract motion features for a single frame with surrounding context"""
        
        # Use a small window around the frame for feature calculation
        window_size = 10  # frames before and after
        start_idx = max(0, frame_idx - window_size)
        end_idx = min(len(motion_data), frame_idx + window_size + 1)
        
        if end_idx - start_idx < 5:  # Too small window
            return None
            
        window_data = motion_data.iloc[start_idx:end_idx]
        
        # Calculate motion features for this window
        velocity = window_data['velocity']
        acceleration = window_data['acceleration']
        jerk = window_data['jerk']
        
        features = {
            'velocity_mean': float(velocity.mean()),
            'velocity_std': float(velocity.std()),
            'velocity_min': float(velocity.min()),
            'velocity_max': float(velocity.max()),
            'acceleration_mean': float(acceleration.mean()),
            'acceleration_std': float(acceleration.std()),
            'acceleration_min': float(acceleration.min()),
            'acceleration_max': float(acceleration.max()),
            'jerk_rms': float(np.sqrt(np.mean(jerk**2))),
            'jerk_mean': float(jerk.mean()),
            'jerk_std': float(jerk.std()),
            'max_deceleration': float(acceleration.min()),
            'duration_s': float(window_data['time'].iloc[-1] - window_data['time'].iloc[0]),
            'motion_smoothness': float(1.0 / (1.0 + jerk.std() + 0.001)),
            'acceleration_reversals': self._count_sign_changes(acceleration),
            'deceleration_events': int((acceleration < -2.0).sum()),
            'jerk_per_second': 0.0,  # Will calculate after duration
            'accel_changes_per_second': 0.0  # Will calculate after duration
        }
        
        # Calculate per-second metrics
        duration = max(features['duration_s'], 0.1)
        features['jerk_per_second'] = features['jerk_rms'] / duration
        features['accel_changes_per_second'] = features['acceleration_reversals'] / duration
        
        return features
    
    def _count_sign_changes(self, series):
        """Count sign changes in a series"""
        if len(series) < 2:
            return 0
        signs = np.sign(series)
        return int((np.diff(signs) != 0).sum())

    def analyze_single_frame(self, motion_data, frame_idx):
        """Analyze a single frame using the ensemble model"""
        
        try:
            # Extract features for this frame
            features_dict = self.extract_frame_features(motion_data, frame_idx)
            
            if features_dict is None:
                return None
            
            # Create features array for advanced model
            features_array = np.array([
                features_dict['jerk_rms'],
                features_dict['velocity_mean'],
                features_dict['velocity_std'],
                features_dict['velocity_min'],
                features_dict['velocity_max'],
                features_dict['acceleration_mean'],
                features_dict['acceleration_std'],
                features_dict['acceleration_min'],
                features_dict['acceleration_max'],
                features_dict['jerk_mean'],
                features_dict['jerk_std'],
                features_dict['max_deceleration'],
                features_dict['deceleration_events'],
                2,  # placeholder
                features_dict['acceleration_reversals'],
                features_dict['motion_smoothness'],
                features_dict['jerk_per_second'],
                features_dict['accel_changes_per_second'],
                features_dict['duration_s'],
                -1.35  # placeholder
            ])
            
            # Get ensemble prediction
            if self.ensemble_model and hasattr(self.ensemble_model, 'predict_ensemble'):
                result = self.ensemble_model.predict_ensemble(features_array, features_dict)
                
                # Calculate risk score
                risk_level = result.get('risk_assessment', {}).get('risk_level', 'LOW')
                confidence = result['ensemble_analysis']['confidence']
                risk_score = confidence * self.risk_multipliers.get(risk_level, 0.25)
                
                return {
                    'frame_idx': frame_idx,
                    'classification': result['ensemble_analysis']['ensemble_prediction'],
                    'confidence': confidence,
                    'risk_level': risk_level,
                    'risk_score': risk_score,
                    'features': features_dict,
                    'strategy': result['ensemble_analysis'].get('strategy', 'unknown')
                }
            else:
                # Fallback demo prediction
                return {
                    'frame_idx': frame_idx,
                    'classification': np.random.choice(['hesitation', 'overshoot', 'nearmiss', 'other']),
                    'confidence': np.random.uniform(0.2, 0.8),
                    'risk_level': np.random.choice(['LOW', 'MEDIUM', 'HIGH']),
                    'risk_score': np.random.uniform(0.1, 0.8),
                    'features': features_dict,
                    'strategy': 'demo_mode'
                }
                
        except Exception as e:
            print(f"Error analyzing frame {frame_idx}: {e}")
            return None
    
    def analyze_full_scene_mp(self, motion_data, max_workers=None):
        """Analyze all frames using multiprocessing"""
        
        print(f"🔄 Starting multiprocessing analysis of {len(motion_data)} frames...")
        
        # Use half of available CPUs by default
        if max_workers is None:
            max_workers = max(1, os.cpu_count() // 2)
        
        # Prepare frame indices (assuming 50% have valid pose data)
        valid_frame_indices = list(range(0, len(motion_data), 2))  # Every other frame
        
        start_time = time.time()
        
        try:
            # Sequential processing for now (multiprocessing has pickle issues with ensemble model)
            results = []
            for i, frame_idx in enumerate(valid_frame_indices):
                if i % 50 == 0:  # Progress update
                    progress = (i / len(valid_frame_indices)) * 100
                    print(f"  Progress: {progress:.1f}% ({i}/{len(valid_frame_indices)} frames)")
                
                result = self.analyze_single_frame(motion_data, frame_idx)
                if result is not None:
                    results.append(result)
            
            # TODO: Implement proper multiprocessing once ensemble model serialization is fixed
            # with Pool(processes=max_workers) as pool:
            #     analyze_func = partial(self.analyze_single_frame, motion_data)
            #     results = pool.map(analyze_func, valid_frame_indices)
            
            # Filter out None results
            valid_results = [r for r in results if r is not None]
            
            elapsed_time = time.time() - start_time
            print(f"✅ Analysis complete: {len(valid_results)} frames analyzed in {elapsed_time:.1f}s")
            
            return valid_results
            
        except Exception as e:
            print(f"❌ Multiprocessing analysis failed: {e}")
            return []
    
    def create_clusters(self, frame_predictions):
        """Create clusters from frame predictions"""
        
        if not frame_predictions:
            return []
        
        print(f"🔗 Creating clusters from {len(frame_predictions)} frame predictions...")
        
        # Sort by frame index
        sorted_predictions = sorted(frame_predictions, key=lambda x: x['frame_idx'])
        
        clusters = []
        current_cluster = None
        
        for prediction in sorted_predictions:
            classification = prediction['classification']
            frame_idx = prediction['frame_idx']
            
            # Start new cluster or continue existing one
            if (current_cluster is None or 
                current_cluster['classification'] != classification or
                frame_idx - current_cluster['end_frame'] > 5):  # Gap threshold: 5 frames
                
                # Save previous cluster
                if current_cluster is not None:
                    clusters.append(self._finalize_cluster(current_cluster))
                
                # Start new cluster
                current_cluster = {
                    'classification': classification,
                    'start_frame': frame_idx,
                    'end_frame': frame_idx,
                    'frames': [frame_idx],
                    'predictions': [prediction]
                }
            else:
                # Continue current cluster
                current_cluster['end_frame'] = frame_idx
                current_cluster['frames'].append(frame_idx)
                current_cluster['predictions'].append(prediction)
        
        # Don't forget the last cluster
        if current_cluster is not None:
            clusters.append(self._finalize_cluster(current_cluster))
        
        # Filter clusters by duration (minimum 2 seconds = 20 frames at 10fps)
        min_duration_frames = 20
        valid_clusters = []
        
        for cluster in clusters:
            duration_frames = cluster['end_frame'] - cluster['start_frame'] + 1
            if duration_frames >= min_duration_frames:
                valid_clusters.append(cluster)
            else:
                print(f"  Dropped short cluster: {cluster['classification']} "
                      f"({duration_frames} frames < {min_duration_frames})")
        
        print(f"✅ Created {len(valid_clusters)} valid clusters (dropped {len(clusters) - len(valid_clusters)} short ones)")
        
        return valid_clusters
    
    def _finalize_cluster(self, cluster):
        """Calculate aggregate statistics for a cluster"""
        
        predictions = cluster['predictions']
        
        # Calculate averages
        avg_confidence = np.mean([p['confidence'] for p in predictions])
        avg_risk_score = np.mean([p['risk_score'] for p in predictions])
        
        # Get most common risk level
        risk_levels = [p['risk_level'] for p in predictions]
        most_common_risk = max(set(risk_levels), key=risk_levels.count)
        
        # Calculate duration
        duration_frames = cluster['end_frame'] - cluster['start_frame'] + 1
        duration_seconds = duration_frames * 0.1  # 10 FPS
        
        cluster.update({
            'avg_confidence': avg_confidence,
            'avg_risk_score': avg_risk_score,
            'dominant_risk_level': most_common_risk,
            'duration_frames': duration_frames,
            'duration_seconds': duration_seconds,
            'final_score': avg_confidence * self.risk_multipliers.get(most_common_risk, 0.25)
        })
        
        return cluster
    
    def select_top_clusters(self, clusters, k=2):
        """Select top K clusters based on final score"""
        
        if not clusters:
            return []
        
        # Sort by final score (descending)
        sorted_clusters = sorted(clusters, key=lambda x: x['final_score'], reverse=True)
        
        # Select top K
        top_clusters = sorted_clusters[:k]
        
        print(f"🎯 Selected top {len(top_clusters)} clusters:")
        for i, cluster in enumerate(top_clusters):
            print(f"  {i+1}. {cluster['classification']} "
                  f"(frames {cluster['start_frame']}-{cluster['end_frame']}, "
                  f"score: {cluster['final_score']:.3f})")
        
        return top_clusters

class TriageBrainGUI:
    """Main GUI application"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Triage Brain - Video Analysis System")
        self.root.geometry("1400x900")
        
        # Data
        self.scenes_df = None
        self.current_scene = None
        self.frame_analyzer = None
        
        # Components
        self.video_player = None
        self.motion_graphs = None
        self.ensemble_panel = None
        
        self.setup_ui()
        self.load_scenes()
        
    def setup_ui(self):
        """Setup the main GUI layout"""
        
        # Top toolbar
        toolbar = tk.Frame(self.root, relief=tk.RAISED, bd=1)
        toolbar.pack(fill=tk.X, padx=5, pady=2)
        
        # Scene selector
        tk.Label(toolbar, text="Scene:", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=5)
        
        self.scene_var = tk.StringVar()
        self.scene_combo = ttk.Combobox(
            toolbar, 
            textvariable=self.scene_var, 
            state="readonly",
            width=50
        )
        self.scene_combo.pack(side=tk.LEFT, padx=5)
        self.scene_combo.bind('<<ComboboxSelected>>', self.on_scene_selected)
        
        # Load button
        self.load_button = tk.Button(
            toolbar, 
            text="Load Scene", 
            command=self.load_current_scene,
            bg='lightgreen'
        )
        self.load_button.pack(side=tk.LEFT, padx=10)
        
        # Export button
        self.export_button = tk.Button(
            toolbar, 
            text="Export Results", 
            command=self.export_results,
            bg='lightyellow'
        )
        self.export_button.pack(side=tk.LEFT, padx=5)
        
        # Status
        self.status_label = tk.Label(toolbar, text="Ready", fg='blue')
        self.status_label.pack(side=tk.RIGHT, padx=10)
        
        # Main content area
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel - Video and ensemble
        left_panel = tk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Video player
        video_frame = tk.LabelFrame(left_panel, text="Video Player", font=('Arial', 12, 'bold'))
        video_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        self.video_player = VideoPlayer(video_frame, width=640, height=360)
        
        # Ensemble panel
        self.ensemble_panel = EnsemblePanel(left_panel)
        
        # Right panel - Motion graphs
        right_panel = tk.Frame(main_frame, width=500)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        right_panel.pack_propagate(False)
        
        graphs_frame = tk.LabelFrame(right_panel, text="Motion Analysis", font=('Arial', 12, 'bold'))
        graphs_frame.pack(fill=tk.BOTH, expand=True)
        
        self.motion_graphs = MotionGraphs(graphs_frame)
        
    def load_scenes(self):
        """Load scene data from CSV"""
        try:
            # Look for scene_summary.csv in data_exploration directory
            csv_paths = [
                "tools/data_exploration/scene_summary.csv",
                "data_exploration/scene_summary.csv", 
                "scene_summary.csv"
            ]
            
            csv_path = None
            for path in csv_paths:
                if os.path.exists(path):
                    csv_path = path
                    break
            
            if csv_path is None:
                raise FileNotFoundError("scene_summary.csv not found")
                
            self.scenes_df = pd.read_csv(csv_path)
            
            # Populate scene selector
            scene_options = []
            for _, row in self.scenes_df.iterrows():
                scene_id = row['scene_id'][:8]  # Short ID for display
                num_frames = row['num_frames_total']
                scene_options.append(f"{scene_id} ({num_frames} frames)")
            
            self.scene_combo['values'] = scene_options
            
            self.status_label.config(text=f"Loaded {len(self.scenes_df)} scenes", fg='green')
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load scenes: {e}")
            self.status_label.config(text="Failed to load scenes", fg='red')
    
    def on_scene_selected(self, event=None):
        """Handle scene selection"""
        if self.scene_combo.get():
            self.load_button.config(state='normal')
    
    def load_current_scene(self):
        """Load the currently selected scene with auto-analysis"""
        if not self.scene_combo.get() or self.scenes_df is None:
            return
        
        try:
            # Get selected scene index
            scene_idx = self.scene_combo.current()
            self.current_scene = self.scenes_df.iloc[scene_idx]
            
            scene_path = self.current_scene['path']
            pose_path = self.current_scene['pose_path']
            scene_id = self.current_scene['scene_id']
            
            self.status_label.config(text="Loading scene...", fg='blue')
            
            # Initialize frame analyzer if not done
            if self.frame_analyzer is None:
                self.frame_analyzer = FrameAnalyzer(self.ensemble_panel.ensemble_model)
            
            # Check for cached analysis
            cache_dir = "cache"
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, f"analysis_{scene_id}.pkl")
            
            # Load video
            if self.video_player.load_video_from_scene(scene_path):
                # Load motion data
                if self.motion_graphs.load_motion_data(pose_path):
                    
                    # Check cache
                    if os.path.exists(cache_file):
                        print(f"✅ Loading cached analysis from {cache_file}")
                        try:
                            with open(cache_file, 'rb') as f:
                                cached_data = pickle.load(f)
                            
                            top_clusters = cached_data.get('top_clusters', [])
                            self.status_label.config(text="Scene loaded (cached analysis)", fg='green')
                            
                        except Exception as e:
                            print(f"❌ Failed to load cache: {e}")
                            top_clusters = self._run_fresh_analysis(cache_file)
                    else:
                        # Run fresh analysis
                        top_clusters = self._run_fresh_analysis(cache_file)
                    
                    # Update graphs with clusters
                    self.motion_graphs.add_cluster_overlays(top_clusters)
                    
                else:
                    self.status_label.config(text="Video loaded, motion data failed", fg='orange')
            else:
                self.status_label.config(text="Failed to load scene", fg='red')
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load scene: {e}")
            self.status_label.config(text="Scene load error", fg='red')
    
    def _run_fresh_analysis(self, cache_file):
        """Run fresh analysis and cache results"""
        
        self.status_label.config(text="Running ensemble analysis...", fg='blue')
        
        # Run full scene analysis
        print(f"🔄 Starting fresh analysis...")
        frame_predictions = self.frame_analyzer.analyze_full_scene_mp(
            self.motion_graphs.motion_data
        )
        
        print(f"📊 Got {len(frame_predictions)} frame predictions")
        
        # Create clusters
        clusters = self.frame_analyzer.create_clusters(frame_predictions)
        print(f"🔗 Created {len(clusters)} total clusters")
        
        # Select top clusters
        top_clusters = self.frame_analyzer.select_top_clusters(clusters, k=2)
        print(f"🎯 Selected {len(top_clusters)} top clusters")
        
        # Debug: Print cluster details
        for i, cluster in enumerate(top_clusters):
            print(f"  Cluster {i+1}: {cluster['classification']} "
                  f"(frames {cluster['start_frame']}-{cluster['end_frame']}, "
                  f"score: {cluster['final_score']:.3f})")
        
        # Cache results
        try:
            cache_data = {
                'scene_id': self.current_scene['scene_id'],
                'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'frame_predictions': frame_predictions,
                'all_clusters': clusters,
                'top_clusters': top_clusters
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            print(f"💾 Analysis cached to {cache_file}")
            
        except Exception as e:
            print(f"⚠️ Failed to cache analysis: {e}")
        
        self.status_label.config(text="Analysis complete - scene loaded", fg='green')
        return top_clusters
    
    def export_results(self):
        """Export comprehensive analysis results"""
        if self.current_scene is None:
            messagebox.showwarning("Warning", "No scene loaded")
            return
        
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if filename:
                # Get cached analysis data
                scene_id = self.current_scene['scene_id']
                cache_file = os.path.join("cache", f"analysis_{scene_id}.pkl")
                
                analysis_data = {}
                if os.path.exists(cache_file):
                    try:
                        with open(cache_file, 'rb') as f:
                            analysis_data = pickle.load(f)
                    except Exception as e:
                        print(f"Warning: Could not load analysis cache: {e}")
                
                # Prepare comprehensive results
                results = {
                    'export_metadata': {
                        'export_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'scene_id': self.current_scene['scene_id'],
                        'scene_path': self.current_scene['path'],
                        'pose_path': self.current_scene['pose_path'],
                        'total_frames': int(self.current_scene['num_frames_total']),
                        'analysis_timestamp': analysis_data.get('analysis_timestamp', 'unknown')
                    },
                    'scene_summary': {
                        'duration_seconds': float(self.current_scene['num_frames_total']) * 0.1,
                        'camera_views': eval(self.current_scene['camera_views']) if isinstance(self.current_scene['camera_views'], str) else self.current_scene['camera_views'],
                        'has_pose_data': bool(self.current_scene['has_pose']),
                        'motion_summary': self.get_motion_summary()
                    },
                    'cluster_analysis': {
                        'top_clusters': analysis_data.get('top_clusters', []),
                        'all_clusters': analysis_data.get('all_clusters', []),
                        'total_analyzed_frames': len(analysis_data.get('frame_predictions', [])),
                        'cluster_summary': self._generate_cluster_summary(analysis_data.get('top_clusters', []))
                    },
                    'frame_level_predictions': {
                        'predictions': analysis_data.get('frame_predictions', []),
                        'risk_distribution': self._calculate_risk_distribution(analysis_data.get('frame_predictions', []))
                    },
                    'export_segments': self._prepare_export_segments(analysis_data.get('top_clusters', []))
                }
                
                with open(filename, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                
                messagebox.showinfo("Success", f"Comprehensive results exported to {filename}")
                
                # Optionally export video segments
                if analysis_data.get('top_clusters'):
                    export_videos = messagebox.askyesno(
                        "Export Video Segments", 
                        "Would you also like to export video segments for the top risk clusters?"
                    )
                    if export_videos:
                        self._export_video_segments(analysis_data['top_clusters'], filename)
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export results: {e}")
    
    def _generate_cluster_summary(self, top_clusters):
        """Generate summary statistics for clusters"""
        if not top_clusters:
            return {}
        
        behaviors = [c['classification'] for c in top_clusters]
        risk_levels = [c['dominant_risk_level'] for c in top_clusters]
        durations = [c['duration_seconds'] for c in top_clusters]
        
        return {
            'total_clusters': len(top_clusters),
            'behavior_distribution': {b: behaviors.count(b) for b in set(behaviors)},
            'risk_distribution': {r: risk_levels.count(r) for r in set(risk_levels)},
            'total_risky_duration': sum(durations),
            'avg_cluster_duration': np.mean(durations) if durations else 0,
            'max_risk_score': max([c['final_score'] for c in top_clusters]) if top_clusters else 0
        }
    
    def _calculate_risk_distribution(self, frame_predictions):
        """Calculate distribution of risk levels across all frames"""
        if not frame_predictions:
            return {}
        
        risk_levels = [p['risk_level'] for p in frame_predictions]
        total_frames = len(frame_predictions)
        
        return {
            'total_frames_analyzed': total_frames,
            'risk_distribution': {
                level: {'count': risk_levels.count(level), 
                       'percentage': (risk_levels.count(level) / total_frames) * 100}
                for level in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
            },
            'avg_confidence': np.mean([p['confidence'] for p in frame_predictions]),
            'avg_risk_score': np.mean([p['risk_score'] for p in frame_predictions])
        }
    
    def _prepare_export_segments(self, top_clusters):
        """Prepare segment information for video export"""
        segments = []
        
        for i, cluster in enumerate(top_clusters):
            segment = {
                'segment_id': f"segment_{i+1}",
                'classification': cluster['classification'],
                'start_frame': cluster['start_frame'],
                'end_frame': cluster['end_frame'],
                'start_time_seconds': cluster['start_frame'] * 0.1,
                'end_time_seconds': cluster['end_frame'] * 0.1,
                'duration_seconds': cluster['duration_seconds'],
                'risk_level': cluster['dominant_risk_level'],
                'confidence': cluster['avg_confidence'],
                'risk_score': cluster['avg_risk_score'],
                'final_score': cluster['final_score'],
                'frame_count': cluster['duration_frames']
            }
            segments.append(segment)
        
        return segments
    
    def _export_video_segments(self, top_clusters, base_filename):
        """Export video segments for top clusters"""
        try:
            base_path = os.path.splitext(base_filename)[0]
            export_dir = f"{base_path}_video_segments"
            os.makedirs(export_dir, exist_ok=True)
            
            for i, cluster in enumerate(top_clusters):
                segment_filename = os.path.join(
                    export_dir, 
                    f"segment_{i+1}_{cluster['classification']}_frames_{cluster['start_frame']}-{cluster['end_frame']}.mp4"
                )
                
                self._create_video_segment(cluster, segment_filename)
            
            messagebox.showinfo("Success", f"Video segments exported to {export_dir}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export video segments: {e}")
    
    def _create_video_segment(self, cluster, output_filename):
        """Create a video file for a specific cluster segment"""
        
        if not self.video_player.frame_data:
            return
        
        # Get frame range
        start_frame = max(0, cluster['start_frame'])
        end_frame = min(len(self.video_player.frame_data) - 1, cluster['end_frame'])
        
        # Read first frame to get dimensions
        first_image = cv2.imread(self.video_player.frame_data[start_frame])
        if first_image is None:
            return
        
        height, width, _ = first_image.shape
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_filename, fourcc, 10.0, (width, height))
        
        try:
            # Write frames
            for frame_idx in range(start_frame, end_frame + 1):
                if frame_idx < len(self.video_player.frame_data):
                    image = cv2.imread(self.video_player.frame_data[frame_idx])
                    if image is not None:
                        video_writer.write(image)
            
            print(f"✅ Created video segment: {output_filename}")
            
        finally:
            video_writer.release()
    
    def get_motion_summary(self) -> Dict:
        """Get summary statistics of motion data"""
        if self.motion_graphs.motion_data is None:
            return {}
        
        data = self.motion_graphs.motion_data
        
        return {
            'duration_seconds': float(data['time'].max()),
            'max_velocity': float(data['velocity'].max()),
            'max_acceleration': float(data['acceleration'].max()),
            'max_jerk': float(data['jerk'].max()),
            'avg_velocity': float(data['velocity'].mean()),
            'velocity_std': float(data['velocity'].std())
        }
    
    def run(self):
        """Start the GUI application"""
        print("🚀 Starting Triage Brain GUI...")
        print("=" * 50)
        print("Features:")
        print("• Video playback with timeline control")
        print("• Real-time motion analysis graphs")
        print("• Ensemble model predictions")
        print("• Scene-based analysis workflow")
        print("• Export capabilities")
        print("=" * 50)
        
        self.root.mainloop()

class RiskySegmentDetector:
    """Detect and annotate risky segments in real-time"""
    
    def __init__(self, ensemble_model, motion_graphs, video_player):
        self.ensemble_model = ensemble_model
        self.motion_graphs = motion_graphs
        self.video_player = video_player
        self.risky_segments = []
        self.segment_size = 30  # frames per segment
        
    def analyze_segment(self, start_frame: int, end_frame: int) -> Dict:
        """Analyze a video segment for risk"""
        
        if self.motion_graphs.motion_data is None:
            return None
        
        # Extract motion data for this segment
        data = self.motion_graphs.motion_data
        total_frames = len(data)
        
        # Map frame indices to motion data indices
        start_idx = int((start_frame / self.video_player.total_frames) * total_frames)
        end_idx = int((end_frame / self.video_player.total_frames) * total_frames)
        
        if start_idx >= end_idx or end_idx > total_frames:
            return None
        
        segment_data = data.iloc[start_idx:end_idx]
        
        # Calculate features for this segment
        features = self.extract_segment_features(segment_data)
        
        # Get ensemble prediction
        if self.ensemble_model and self.ensemble_model.ensemble_model:
            try:
                result = self.ensemble_model.analyze_segment(
                    features['features_array'], 
                    features['features_dict']
                )
                return result
            except Exception as e:
                print(f"Error in ensemble analysis: {e}")
                return None
        
        return None
    
    def extract_segment_features(self, segment_data: pd.DataFrame) -> Dict:
        """Extract features from a motion data segment"""
        
        if len(segment_data) < 5:  # Too small segment
            return None
        
        # Calculate segment statistics
        velocity = segment_data['velocity']
        acceleration = segment_data['acceleration'] 
        jerk = segment_data['jerk']
        
        features_dict = {
            'velocity_mean': float(velocity.mean()),
            'velocity_std': float(velocity.std()),
            'velocity_min': float(velocity.min()),
            'velocity_max': float(velocity.max()),
            'acceleration_mean': float(acceleration.mean()),
            'acceleration_std': float(acceleration.std()),
            'acceleration_min': float(acceleration.min()),
            'acceleration_max': float(acceleration.max()),
            'jerk_rms': float(np.sqrt(np.mean(jerk**2))),
            'jerk_mean': float(jerk.mean()),
            'jerk_std': float(jerk.std()),
            'max_deceleration': float(acceleration.min()),
            'duration_s': float(segment_data['time'].iloc[-1] - segment_data['time'].iloc[0]),
            'motion_smoothness': float(1.0 / (1.0 + jerk.std())),
            'acceleration_reversals': self.count_sign_changes(acceleration),
            'deceleration_events': int((acceleration < -2.0).sum()),
            'jerk_per_second': float(np.sqrt(np.mean(jerk**2)) / max(features_dict.get('duration_s', 1), 0.1)),
            'accel_changes_per_second': float(self.count_sign_changes(acceleration) / max(features_dict.get('duration_s', 1), 0.1))
        }
        
        # Create features array (matching ensemble model expectations)
        features_array = np.array([
            features_dict['jerk_rms'],
            features_dict['velocity_mean'],
            features_dict['velocity_std'],
            features_dict['velocity_min'],
            features_dict['velocity_max'],
            features_dict['acceleration_mean'],
            features_dict['acceleration_std'],
            features_dict['acceleration_min'],
            features_dict['acceleration_max'],
            features_dict['jerk_mean'],
            features_dict['jerk_std'],
            features_dict['max_deceleration'],
            features_dict['deceleration_events'],
            2,  # placeholder
            features_dict['acceleration_reversals'],
            features_dict['motion_smoothness'],
            features_dict['jerk_per_second'],
            features_dict['accel_changes_per_second'],
            features_dict['duration_s'],
            -1.35  # placeholder
        ])
        
        return {
            'features_dict': features_dict,
            'features_array': features_array
        }
    
    def count_sign_changes(self, series: pd.Series) -> int:
        """Count sign changes in a series"""
        signs = np.sign(series)
        return int((np.diff(signs) != 0).sum())

class TimelineAnnotator:
    """Timeline annotation widget for marking risky segments"""
    
    def __init__(self, parent, video_player):
        self.parent = parent
        self.video_player = video_player
        self.annotations = []
        self.setup_ui()
        
    def setup_ui(self):
        """Setup timeline annotation UI"""
        
        self.timeline_frame = tk.LabelFrame(self.parent, text="Timeline Annotations")
        self.timeline_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Annotation canvas
        self.canvas = tk.Canvas(self.timeline_frame, height=60, bg='lightgray')
        self.canvas.pack(fill=tk.X, padx=5, pady=5)
        
        # Controls
        controls_frame = tk.Frame(self.timeline_frame)
        controls_frame.pack(fill=tk.X, padx=5, pady=2)
        
        self.mark_start_button = tk.Button(
            controls_frame, 
            text="Mark Start", 
            command=self.mark_segment_start,
            bg='lightgreen'
        )
        self.mark_start_button.pack(side=tk.LEFT, padx=2)
        
        self.mark_end_button = tk.Button(
            controls_frame, 
            text="Mark End", 
            command=self.mark_segment_end,
            bg='lightcoral'
        )
        self.mark_end_button.pack(side=tk.LEFT, padx=2)
        
        # Risk level selector
        tk.Label(controls_frame, text="Risk:").pack(side=tk.LEFT, padx=10)
        self.risk_var = tk.StringVar(value="HIGH")
        risk_combo = ttk.Combobox(
            controls_frame, 
            textvariable=self.risk_var,
            values=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
            width=10,
            state="readonly"
        )
        risk_combo.pack(side=tk.LEFT, padx=2)
        
        # Clear button
        self.clear_button = tk.Button(
            controls_frame, 
            text="Clear All", 
            command=self.clear_annotations,
            bg='lightyellow'
        )
        self.clear_button.pack(side=tk.RIGHT, padx=2)
        
        # Current selection
        self.current_start = None
        self.update_timeline()
        
    def mark_segment_start(self):
        """Mark the start of a risky segment"""
        if self.video_player.video_capture is None:
            return
            
        self.current_start = self.video_player.current_frame
        self.update_timeline()
        
    def mark_segment_end(self):
        """Mark the end of a risky segment"""
        if self.current_start is None:
            messagebox.showwarning("Warning", "Please mark segment start first")
            return
            
        end_frame = self.video_player.current_frame
        
        if end_frame <= self.current_start:
            messagebox.showwarning("Warning", "End frame must be after start frame")
            return
        
        # Add annotation
        annotation = {
            'start_frame': self.current_start,
            'end_frame': end_frame,
            'risk_level': self.risk_var.get(),
            'timestamp': time.strftime('%H:%M:%S')
        }
        
        self.annotations.append(annotation)
        self.current_start = None
        self.update_timeline()
        
        print(f"✅ Marked risky segment: frames {annotation['start_frame']}-{annotation['end_frame']} ({annotation['risk_level']})")
    
    def clear_annotations(self):
        """Clear all annotations"""
        self.annotations = []
        self.current_start = None
        self.update_timeline()
        
    def update_timeline(self):
        """Update the timeline visualization"""
        self.canvas.delete("all")
        
        if self.video_player.total_frames == 0:
            return
        
        canvas_width = self.canvas.winfo_width()
        if canvas_width <= 1:
            canvas_width = 800  # Default width
            
        canvas_height = 60
        
        # Draw timeline background
        self.canvas.create_rectangle(10, 20, canvas_width-10, 40, fill='white', outline='black')
        
        # Draw annotations
        for annotation in self.annotations:
            start_x = 10 + (annotation['start_frame'] / self.video_player.total_frames) * (canvas_width - 20)
            end_x = 10 + (annotation['end_frame'] / self.video_player.total_frames) * (canvas_width - 20)
            
            colors = {
                'LOW': 'yellow',
                'MEDIUM': 'orange', 
                'HIGH': 'red',
                'CRITICAL': 'darkred'
            }
            
            color = colors.get(annotation['risk_level'], 'gray')
            
            self.canvas.create_rectangle(start_x, 20, end_x, 40, fill=color, alpha=0.7)
            
        # Draw current selection
        if self.current_start is not None:
            start_x = 10 + (self.current_start / self.video_player.total_frames) * (canvas_width - 20)
            current_x = 10 + (self.video_player.current_frame / self.video_player.total_frames) * (canvas_width - 20)
            
            self.canvas.create_rectangle(start_x, 20, current_x, 40, fill='lightblue', alpha=0.5)
        
        # Draw current position
        if self.video_player.total_frames > 0:
            pos_x = 10 + (self.video_player.current_frame / self.video_player.total_frames) * (canvas_width - 20)
            self.canvas.create_line(pos_x, 10, pos_x, 50, fill='blue', width=2)

def main():
    """Main entry point"""
    
    print("🧠 TRIAGE BRAIN - VIDEO ANALYSIS GUI")
    print("=" * 50)
    print("A comprehensive video analysis tool with:")
    print("• Real-time video playback and timeline control")
    print("• Motion analysis with velocity, acceleration, and jerk graphs")
    print("• Ensemble ML model predictions")
    print("• Manual annotation system for risky segments")
    print("• Integration with Argoverse dataset")
    print("=" * 50)
    
    try:
        app = TriageBrainGUI()
        app.run()
    except Exception as e:
        print(f"❌ Failed to start GUI: {e}")
        messagebox.showerror("Error", f"Failed to start application: {e}")

if __name__ == "__main__":
    main()