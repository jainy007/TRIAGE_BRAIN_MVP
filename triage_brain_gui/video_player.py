#!/usr/bin/env python3
"""
Video Player Component for Triage Brain GUI
Enhanced to support both frame sequences and direct MP4 loading
"""

import tkinter as tk
import cv2
import numpy as np
import os
import glob
from PIL import Image, ImageTk
from typing import List, Optional, Callable
from utils import logger

class VideoPlayer:
    """Enhanced video player supporting both frame sequences and MP4 files"""
    
    def __init__(self, parent, width=640, height=360):
        self.parent = parent
        self.width = width
        self.height = height
        
        # Video state
        self.frame_data = []  # For frame sequences
        self.video_path = None  # For MP4 files
        self.cap = None  # OpenCV VideoCapture for MP4
        self.current_frame = 0
        self.total_frames = 0
        self.is_playing = False
        self.frame_change_callback = None
        self.fps = 10  # Default FPS
        
        # Seeking state
        self.is_seeking = False  # Track if user is dragging slider
        self.seek_frame = 0      # Target frame for seeking
        
        # Create UI
        self.create_ui()
        
        # Start update loop
        self.update_display()
        
        logger.info("Enhanced video player initialized")
    
    def create_ui(self):
        """Create video player UI with proper seeking"""
        # Video canvas
        self.canvas = tk.Canvas(self.parent, width=self.width, height=self.height, bg='black')
        self.canvas.pack(pady=10)
        
        # Slider with proper event bindings for seeking
        self.slider = tk.Scale(
            self.parent, 
            orient=tk.HORIZONTAL, 
            length=600,
            showvalue=False,
            command=self.on_slider_change
        )
        self.slider.pack(fill=tk.X, padx=20, pady=5)
        
        # Bind slider events for proper seeking
        self.slider.bind("<Button-1>", self.on_slider_press)
        self.slider.bind("<B1-Motion>", self.on_slider_drag)
        self.slider.bind("<ButtonRelease-1>", self.on_slider_release)
        
        # Controls
        controls = tk.Frame(self.parent)
        controls.pack(pady=10)
        
        self.play_btn = tk.Button(controls, text="Play", command=self.toggle_play, font=('Arial', 10, 'bold'))
        self.play_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = tk.Button(controls, text="Stop", command=self.stop, font=('Arial', 10, 'bold'))
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        self.prev_btn = tk.Button(controls, text="<<", command=self.prev_frame, font=('Arial', 10, 'bold'))
        self.prev_btn.pack(side=tk.LEFT, padx=5)
        
        self.next_btn = tk.Button(controls, text=">>", command=self.next_frame, font=('Arial', 10, 'bold'))
        self.next_btn.pack(side=tk.LEFT, padx=5)
        
        # Info
        self.info_label = tk.Label(self.parent, text="No video loaded", font=('Arial', 11))
        self.info_label.pack(pady=5)
    
    def load_video_from_scene(self, frame_paths: List[str]) -> bool:
        """Load video from frame paths (original method)"""
        try:
            if not frame_paths:
                logger.error("No frame paths provided")
                return False
            
            # Clear MP4 mode
            self._clear_mp4_mode()
            
            self.frame_data = frame_paths
            self.total_frames = len(frame_paths)
            self.current_frame = 0
            self.is_playing = False
            
            # Update slider range
            if self.total_frames > 0:
                self.slider.configure(from_=0, to=self.total_frames-1)
            
            # Load first frame
            self.show_frame(0)
            self.play_btn.config(text="Play")
            
            logger.info(f"Loaded {self.total_frames} frames from scene")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load video from scene: {e}")
            return False
    
    def load_video_from_mp4(self, mp4_path: str) -> bool:
        """Load video directly from MP4 file"""
        try:
            if not os.path.exists(mp4_path):
                logger.error(f"MP4 file not found: {mp4_path}")
                return False
            
            # Clear frame mode
            self._clear_frame_mode()
            
            # Load MP4
            self.video_path = mp4_path
            self.cap = cv2.VideoCapture(mp4_path)
            
            if not self.cap.isOpened():
                raise ValueError(f"Cannot open video: {mp4_path}")
            
            # Get video properties
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            # Reset state
            self.current_frame = 0
            self.is_playing = False
            
            # Update UI
            if self.total_frames > 0:
                self.slider.configure(from_=0, to=self.total_frames-1)
            
            # Show first frame
            self.show_frame(0)
            self.play_btn.config(text="Play")
            
            logger.info(f"Loaded MP4: {os.path.basename(mp4_path)} ({self.total_frames} frames, {self.fps:.1f} FPS)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load MP4 {mp4_path}: {e}")
            return False
    
    def _clear_mp4_mode(self):
        """Clear MP4 mode resources"""
        if self.cap:
            self.cap.release()
            self.cap = None
        self.video_path = None
    
    def _clear_frame_mode(self):
        """Clear frame mode resources"""
        self.frame_data = []
    
    def show_frame(self, frame_idx: int):
        """Display a specific frame (works for both modes)"""
        if frame_idx >= self.total_frames or frame_idx < 0:
            return
        
        try:
            image = None
            
            if self.cap:  # MP4 mode
                # Seek to frame
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = self.cap.read()
                
                if ret:
                    image = cv2.resize(frame, (self.width, self.height))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            elif self.frame_data:  # Frame sequence mode
                if frame_idx < len(self.frame_data):
                    img = cv2.imread(self.frame_data[frame_idx])
                    if img is not None:
                        image = cv2.resize(img, (self.width, self.height))
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if image is not None:
                # Convert to PhotoImage
                pil_img = Image.fromarray(image)
                photo = ImageTk.PhotoImage(pil_img)
                
                # Update canvas
                self.canvas.delete("all")
                self.canvas.create_image(self.width//2, self.height//2, image=photo)
                self.canvas.image = photo  # Keep reference
                
                # Update state
                self.current_frame = frame_idx
                self.slider.set(frame_idx)
                self.info_label.config(text=f"Frame {frame_idx+1}/{self.total_frames}")
                
                # Callback
                if self.frame_change_callback:
                    self.frame_change_callback(frame_idx, self.total_frames)
                
        except Exception as e:
            logger.error(f"Error showing frame {frame_idx}: {e}")
    
    def toggle_play(self):
        """Toggle play/pause"""
        self.is_playing = not self.is_playing
        self.play_btn.config(text="Pause" if self.is_playing else "Play")
    
    def stop(self):
        """Stop playback"""
        self.is_playing = False
        self.play_btn.config(text="Play")
        self.show_frame(0)
    
    def prev_frame(self):
        """Previous frame"""
        if self.current_frame > 0:
            self.show_frame(self.current_frame - 1)
    
    def next_frame(self):
        """Next frame"""
        if self.current_frame < self.total_frames - 1:
            self.show_frame(self.current_frame + 1)
    
    def on_slider_press(self, event):
        """Handle slider press - start seeking"""
        self.is_seeking = True
        # Pause playback during seeking
        if self.is_playing:
            self.was_playing = True
            self.is_playing = False
        else:
            self.was_playing = False
    
    def on_slider_drag(self, event):
        """Handle slider drag - update seek frame but don't show yet"""
        if self.is_seeking:
            self.seek_frame = int(self.slider.get())
    
    def on_slider_release(self, event):
        """Handle slider release - perform the seek"""
        if self.is_seeking:
            target_frame = int(self.slider.get())
            self.show_frame(target_frame)
            
            # Resume playback if it was playing before
            if self.was_playing:
                self.is_playing = True
            
            self.is_seeking = False
    
    def on_slider_change(self, value):
        """Handle slider value change (called automatically)"""
        # Only seek if not currently seeking (to avoid conflicts)
        if not self.is_seeking and not self.is_playing:
            frame_idx = int(float(value))
            self.show_frame(frame_idx)
    
    def update_display(self):
        """Update loop for playback"""
        if self.is_playing and self.total_frames > 0:
            if self.current_frame < self.total_frames - 1:
                self.show_frame(self.current_frame + 1)
            else:
                self.is_playing = False
                self.play_btn.config(text="Play")
        
        # Schedule next update
        self.parent.after(100, self.update_display)
    
    def set_frame_change_callback(self, callback: Callable[[int, int], None]):
        """Set callback for frame changes"""
        self.frame_change_callback = callback
    
    def clear_video(self):
        """Clear video"""
        self._clear_mp4_mode()
        self._clear_frame_mode()
        
        self.total_frames = 0
        self.current_frame = 0
        self.is_playing = False
        self.canvas.delete("all")
        self.info_label.config(text="No video loaded")
        self.play_btn.config(text="Play")
    
    def get_video_info(self) -> dict:
        """Get current video information"""
        return {
            'total_frames': self.total_frames,
            'current_frame': self.current_frame,
            'fps': self.fps,
            'is_mp4_mode': self.cap is not None,
            'is_frame_mode': len(self.frame_data) > 0,
            'video_path': self.video_path
        }