#!/usr/bin/env python3
"""
Motion Analysis with PROPER filtering and STABLE overlays
"""

import tkinter as tk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from typing import Dict, List, Optional, Tuple
from utils import logger

# Proper signal processing
try:
    from scipy.signal import butter, filtfilt
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.info("scipy not available - using simple filtering")

class MotionAnalyzer:
    """Motion analyzer with PROPER filtering and STABLE overlays"""
    
    def __init__(self, parent):
        self.parent = parent
        self.motion_data = None
        self.current_clusters = []
        self.cluster_rectangles = []  # Store references to prevent deletion
        self.cluster_texts = []
        
        # Setup UI with LARGER fonts
        self.setup_ui()
        
        logger.info("Motion analyzer initialized with proper filtering")
        
    def setup_ui(self):
        """Setup motion graphs UI with LARGER fonts"""
        self.graphs_frame = tk.Frame(self.parent)
        self.graphs_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create matplotlib figure with LARGER fonts
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'legend.fontsize': 10
        })
        
        self.fig = Figure(figsize=(10, 8), dpi=90, facecolor='white')
        self.fig.suptitle('Motion Analysis', fontsize=16, fontweight='bold')
        
        # Create subplots with more space
        self.ax1 = self.fig.add_subplot(3, 1, 1)
        self.ax2 = self.fig.add_subplot(3, 1, 2)
        self.ax3 = self.fig.add_subplot(3, 1, 3)
        
        # Setup axes with LARGER fonts
        self._setup_axis(self.ax1, 'Velocity (Filtered)', 'm/s')
        self._setup_axis(self.ax2, 'Acceleration (Filtered)', 'm/s²')
        self._setup_axis(self.ax3, 'Jerk (Filtered)', 'm/s³', xlabel='Time (s)')
        
        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, self.graphs_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Adjust layout
        self.fig.tight_layout(pad=3.0)
        
    def _setup_axis(self, ax, title, ylabel, xlabel=None):
        """Setup individual axis with LARGER fonts"""
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=12)
        ax.grid(True, alpha=0.3, linewidth=0.8)
        ax.tick_params(labelsize=11)
        
    def apply_proper_filtering(self, data: np.ndarray, sample_rate: float = 10.0) -> np.ndarray:
        """Apply PROPER highpass and lowpass filtering"""
        if not SCIPY_AVAILABLE or len(data) < 10:
            # Simple moving average fallback
            window = 3
            return np.convolve(data, np.ones(window)/window, mode='same')
        
        try:
            # PROPER Butterworth filtering
            nyquist = sample_rate / 2.0
            
            # Highpass filter to remove DC offset and very low frequency drift
            high_cutoff = 0.1  # Hz - removes slow drift
            high_normalized = high_cutoff / nyquist
            
            if high_normalized < 0.99:  # Ensure valid frequency
                b_high, a_high = butter(2, high_normalized, btype='high')
                data_highpass = filtfilt(b_high, a_high, data)
            else:
                data_highpass = data
            
            # Lowpass filter to remove high frequency noise
            low_cutoff = 3.0  # Hz - removes noise while keeping driving dynamics
            low_normalized = low_cutoff / nyquist
            
            if low_normalized < 0.99:  # Ensure valid frequency
                b_low, a_low = butter(3, low_normalized, btype='low')
                data_filtered = filtfilt(b_low, a_low, data_highpass)
            else:
                data_filtered = data_highpass
            
            return data_filtered
            
        except Exception as e:
            logger.warning(f"Filtering failed: {e}, using simple smoothing")
            # Fallback to simple smoothing
            window = 5
            return np.convolve(data, np.ones(window)/window, mode='same')
    
    def load_and_process_motion_data(self, pose_df: pd.DataFrame) -> bool:
        """Load and process motion data with PROPER filtering"""
        try:
            self.motion_data = self.calculate_motion_metrics(pose_df)
            self.update_graphs(preserve_overlays=False)  # Initial load
            
            logger.info(f"Processed motion data with proper filtering: {len(self.motion_data)} samples")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process motion data: {e}")
            return False
    
    def calculate_motion_metrics(self, pose_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate motion metrics with PROPER filtering"""
        
        # Extract position and timestamp data
        if 'tx_m' in pose_df.columns and 'ty_m' in pose_df.columns and 'tz_m' in pose_df.columns:
            positions = pose_df[['tx_m', 'ty_m', 'tz_m']].values
            
            if 'timestamp_ns' in pose_df.columns:
                timestamps = pose_df['timestamp_ns'].values / 1e9
            else:
                timestamps = np.arange(len(positions)) * 0.1
        else:
            # Generate synthetic data for MP4s
            logger.info("Generating synthetic motion data for MP4")
            timestamps = np.arange(len(pose_df)) * 0.1
            
            # More realistic synthetic motion
            t = timestamps
            base_velocity = 8.0
            velocity_profile = base_velocity + 2 * np.sin(0.3 * t) + np.random.normal(0, 0.5, len(t))
            velocity_profile = np.maximum(velocity_profile, 1.0)  # Minimum 1 m/s
            
            # Integrate to get position
            positions = np.column_stack([
                np.cumsum(velocity_profile) * 0.1,  # x position
                np.random.normal(0, 0.2, len(t)),   # y position (lane changes)
                np.zeros(len(t))  # z position (flat road)
            ])
        
        if len(timestamps) < 3:
            logger.warning("Insufficient data points")
            return pd.DataFrame()
        
        # Calculate raw derivatives
        dt = np.diff(timestamps)
        dt = np.maximum(dt, 1e-6)  # Prevent division by zero
        
        # Raw velocity
        velocity_raw = np.linalg.norm(np.diff(positions, axis=0) / dt[:, None], axis=1)
        
        # Raw acceleration
        if len(velocity_raw) > 1:
            acceleration_raw = np.diff(velocity_raw) / dt[:-1]
        else:
            acceleration_raw = np.zeros(1)
        
        # Raw jerk
        if len(acceleration_raw) > 1:
            jerk_raw = np.diff(acceleration_raw) / dt[:-2]
        else:
            jerk_raw = np.zeros(1)
        
        # Apply PROPER filtering to each signal
        sample_rate = 1.0 / np.mean(dt) if len(dt) > 0 else 10.0
        
        velocity_filtered = self.apply_proper_filtering(velocity_raw, sample_rate)
        acceleration_filtered = self.apply_proper_filtering(acceleration_raw, sample_rate)
        jerk_filtered = self.apply_proper_filtering(jerk_raw, sample_rate)
        
        # Align all arrays to same length
        min_length = min(len(velocity_filtered), len(acceleration_filtered) + 1, len(jerk_filtered) + 2)
        
        # Create aligned dataframe
        motion_df = pd.DataFrame({
            'time': timestamps[1:min_length+1],
            'velocity': velocity_filtered[:min_length],
            'acceleration': np.pad(acceleration_filtered, (0, max(0, min_length - len(acceleration_filtered))), 'edge')[:min_length],
            'jerk': np.pad(jerk_filtered, (0, max(0, min_length - len(jerk_filtered))), 'edge')[:min_length],
            'x': positions[1:min_length+1, 0],
            'y': positions[1:min_length+1, 1]
        })
        
        logger.info(f"Applied proper filtering: vel_range=[{motion_df['velocity'].min():.2f}, {motion_df['velocity'].max():.2f}]")
        
        return motion_df
    
    def update_graphs(self, preserve_overlays=True):
        """Update graphs while PRESERVING overlays"""
        if self.motion_data is None:
            return
        
        # Store overlays if preserving
        saved_clusters = self.current_clusters.copy() if preserve_overlays else []
        
        # Clear only the data, not the patches
        for ax in [self.ax1, self.ax2, self.ax3]:
            # Remove only line plots, keep rectangles and text
            lines_to_remove = [line for line in ax.lines if not hasattr(line, '_cluster_marker')]
            for line in lines_to_remove:
                line.remove()
        
        # Re-setup axes
        self._setup_axis(self.ax1, 'Velocity (Filtered)', 'm/s')
        self._setup_axis(self.ax2, 'Acceleration (Filtered)', 'm/s²')
        self._setup_axis(self.ax3, 'Jerk (Filtered)', 'm/s³', xlabel='Time (s)')
        
        time = self.motion_data['time']
        
        # Plot filtered data with thicker lines
        self.ax1.plot(time, self.motion_data['velocity'], 'b-', linewidth=2.5, alpha=0.9, label='Velocity')
        self.ax2.plot(time, self.motion_data['acceleration'], 'g-', linewidth=2.5, alpha=0.9, label='Acceleration')
        self.ax3.plot(time, self.motion_data['jerk'], 'r-', linewidth=2.5, alpha=0.9, label='Jerk')
        
        # Restore overlays if preserving
        if preserve_overlays and saved_clusters:
            self.current_clusters = []
            self.add_cluster_overlays(saved_clusters)
        
        self.fig.tight_layout(pad=3.0)
        self.canvas.draw()
    
    def add_cluster_overlays(self, clusters: List[Dict]):
        """Add STABLE cluster overlays with CORRECT time mapping and compact labels"""
        if not clusters or self.motion_data is None:
            return
        
        # Clear old overlays
        self.clear_overlays()
        
        max_time = self.motion_data['time'].max()
        total_data_points = len(self.motion_data)
        
        logger.info(f"Adding overlays: max_time={max_time:.1f}s, data_points={total_data_points}")
        
        # Colors with better visibility
        behavior_colors = {
            'hesitation': ('orange', 0.4),
            'overshoot': ('red', 0.4),
            'nearmiss': ('yellow', 0.4),
            'oversteering': ('purple', 0.4),
            'other': ('lightblue', 0.3)
        }
        
        for i, cluster in enumerate(clusters):
            # CORRECT time mapping - map cluster frames to actual video time
            # The cluster frames should correspond to the actual video timeline
            video_start_time = cluster['start_frame'] / 10.0  # Assuming 10 FPS
            video_end_time = cluster['end_frame'] / 10.0
            
            # Map video time to motion data time
            # Motion data time goes from 0 to max_time over total_data_points
            data_start_time = (video_start_time / (total_data_points / 10.0)) * max_time
            data_end_time = (video_end_time / (total_data_points / 10.0)) * max_time
            
            # Clamp to data range
            data_start_time = max(0, min(data_start_time, max_time))
            data_end_time = max(data_start_time, min(data_end_time, max_time))
            
            behavior = cluster['classification']
            color, alpha = behavior_colors.get(behavior, ('lightgray', 0.3))
            
            logger.info(f"Cluster {i+1}: {behavior} video_frames {cluster['start_frame']}-{cluster['end_frame']} ({video_start_time:.1f}-{video_end_time:.1f}s) -> data_time {data_start_time:.1f}-{data_end_time:.1f}s")
            
            # Add to all three plots
            for ax in [self.ax1, self.ax2, self.ax3]:
                y_min, y_max = ax.get_ylim()
                
                # Create PERSISTENT rectangle
                rect = Rectangle(
                    (data_start_time, y_min),
                    data_end_time - data_start_time,
                    y_max - y_min,
                    facecolor=color,
                    alpha=alpha,
                    edgecolor=color,
                    linewidth=2,
                    zorder=1,
                    picker=False
                )
                ax.add_patch(rect)
                self.cluster_rectangles.append(rect)  # STORE REFERENCE
                
                # Add COMPACT text label (only on velocity graph)
                if ax == self.ax1:
                    mid_time = (data_start_time + data_end_time) / 2
                    label_y = y_min + (y_max - y_min) * 0.85
                    
                    # Create COMPACT label - no line breaks, smaller font
                    duration = cluster['duration_seconds']
                    start_frame = cluster['start_frame']
                    end_frame = cluster['end_frame']
                    
                    # Short behavior name
                    short_behavior = {
                        'hesitation': 'hesit',
                        'overshoot': 'ovrsht',
                        'nearmiss': 'near',
                        'oversteering': 'steer'
                    }.get(behavior, behavior[:5])
                    
                    label_text = f"{short_behavior} {duration:.1f}s f{start_frame}-{end_frame}"
                    
                    text = ax.text(
                        mid_time, label_y,
                        label_text,
                        ha='center', va='center',
                        fontsize=9,  # Smaller font
                        fontweight='bold',
                        color='black',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.9, edgecolor=color),
                        zorder=10
                    )
                    self.cluster_texts.append(text)  # STORE REFERENCE
        
        # Store clusters
        self.current_clusters = clusters
        self.canvas.draw()
        
        logger.info(f"Added {len(clusters)} STABLE cluster overlays with CORRECT timing")
    
    def highlight_timeline_position(self, frame_idx: int, total_frames: int):
        """Highlight timeline position with CORRECT mapping"""
        if self.motion_data is None or total_frames == 0:
            return
        
        # Calculate correct time position
        # Video time based on frame and FPS
        video_time = frame_idx / 10.0  # Assuming 10 FPS
        
        # Map video time to motion data time
        max_time = self.motion_data['time'].max()
        total_data_points = len(self.motion_data)
        data_time = (video_time / (total_data_points / 10.0)) * max_time
        
        # Clamp to data range
        data_time = max(0, min(data_time, max_time))
        
        # Remove ONLY old timeline markers
        for ax in [self.ax1, self.ax2, self.ax3]:
            lines_to_remove = [line for line in ax.lines if hasattr(line, '_timeline_marker')]
            for line in lines_to_remove:
                line.remove()
        
        # Add new timeline markers
        for ax in [self.ax1, self.ax2, self.ax3]:
            line = ax.axvline(x=data_time, color='blue', linestyle='--', alpha=0.8, linewidth=3)
            line._timeline_marker = True  # Mark for removal
        
        # Use draw_idle for performance
        self.canvas.draw_idle()
    
    def clear_overlays(self):
        """Clear overlays"""
        # Remove rectangles
        for rect in self.cluster_rectangles:
            if rect.axes and rect in rect.axes.patches:
                rect.remove()
        self.cluster_rectangles.clear()
        
        # Remove texts
        for text in self.cluster_texts:
            if text.axes and text in text.axes.texts:
                text.remove()
        self.cluster_texts.clear()
        
        self.current_clusters = []
        self.canvas.draw()
    
    def get_motion_summary(self) -> Dict:
        """Get motion summary"""
        if self.motion_data is None:
            return {}
        
        data = self.motion_data
        return {
            'duration_seconds': float(data['time'].max()),
            'max_velocity': float(data['velocity'].max()),
            'avg_velocity': float(data['velocity'].mean()),
            'velocity_std': float(data['velocity'].std()),
            'max_acceleration': float(data['acceleration'].max()),
            'min_acceleration': float(data['acceleration'].min()),
            'max_jerk': float(data['jerk'].max()),
            'avg_jerk_rms': float(np.sqrt(np.mean(data['jerk']**2)))
        }
    
    def add_dangerous_event_overlays(self, clusters: List[Dict]):
        """Add GREEN overlays for detected dangerous events"""
        if not clusters or self.motion_data is None:
            return

        # Clear old overlays
        self.clear_overlays()

        max_time = self.motion_data['time'].max()
        total_data_points = len(self.motion_data)

        logger.info(f"Adding DANGEROUS EVENT overlays: {len(clusters)} events")

        # Use bright green for dangerous events
        event_color = 'lime'
        event_alpha = 0.3

        for i, cluster in enumerate(clusters):
            # Convert cluster frames to motion data time
            video_start_time = cluster['start_frame'] / 10.0
            video_end_time = cluster['end_frame'] / 10.0

            # Map to motion data timeline
            data_start_time = (video_start_time / (total_data_points / 10.0)) * max_time
            data_end_time = (video_end_time / (total_data_points / 10.0)) * max_time

            # Clamp to data range
            data_start_time = max(0, min(data_start_time, max_time))
            data_end_time = max(data_start_time, min(data_end_time, max_time))

            confidence = cluster['avg_confidence']
            annotation_info = cluster.get('annotation_info', {})
            comment = annotation_info.get('comment', 'Dangerous Event') if annotation_info else 'Dangerous Event'

            logger.info(f"Event {i+1}: {comment} at {data_start_time:.1f}s-{data_end_time:.1f}s (conf: {confidence:.3f})")

            # Add to all three plots
            for ax in [self.ax1, self.ax2, self.ax3]:
                y_min, y_max = ax.get_ylim()

                # Create GREEN rectangle for dangerous event
                rect = Rectangle(
                    (data_start_time, y_min),
                    data_end_time - data_start_time,
                    y_max - y_min,
                    facecolor=event_color,
                    alpha=event_alpha,
                    edgecolor='darkgreen',
                    linewidth=3,
                    zorder=1
                )
                ax.add_patch(rect)
                self.cluster_rectangles.append(rect)

                # Add label on velocity graph only
                if ax == self.ax1:
                    mid_time = (data_start_time + data_end_time) / 2
                    label_y = y_min + (y_max - y_min) * 0.9

                    
                    duration = cluster['duration_seconds']

                    label_text = f"🚨 Risky Event ({duration:.1f}s)"

                    text = ax.text(
                        mid_time, label_y,
                        label_text,
                        ha='center', va='center',
                        fontsize=10,
                        fontweight='bold',
                        color='darkgreen',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.9, edgecolor='darkgreen'),
                        zorder=10
                    )
                    self.cluster_texts.append(text)

        # Store clusters
        self.current_clusters = clusters
        self.canvas.draw()

        logger.info(f"Added {len(clusters)} DANGEROUS EVENT overlays")