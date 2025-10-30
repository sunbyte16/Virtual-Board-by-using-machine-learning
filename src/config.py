"""
Configuration settings for Virtual Board
"""
import os
from dataclasses import dataclass
from typing import Tuple

@dataclass
class DrawingConfig:
    """Drawing-related configuration"""
    drawing_color: Tuple[int, int, int] = (0, 255, 0)  # Green
    eraser_color: Tuple[int, int, int] = (0, 0, 0)     # Black
    brush_thickness: int = 5
    eraser_thickness: int = 20
    canvas_background: Tuple[int, int, int] = (0, 0, 0)  # Black

@dataclass
class HandTrackingConfig:
    """Hand tracking configuration"""
    static_image_mode: bool = False
    max_num_hands: int = 1
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.5

@dataclass
class MLConfig:
    """Machine Learning configuration"""
    model_path: str = "models/digit_model.h5"
    input_shape: Tuple[int, int, int] = (28, 28, 1)
    epochs: int = 5
    batch_size: int = 128
    validation_split: float = 0.2

@dataclass
class CameraConfig:
    """Camera configuration"""
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    fps: int = 30

@dataclass
class UIConfig:
    """UI configuration"""
    window_name_main: str = "Virtual Board - Main"
    window_name_canvas: str = "Virtual Board - Canvas"
    font: int = 0  # cv2.FONT_HERSHEY_SIMPLEX
    font_scale: float = 1.0
    font_thickness: int = 2

class Config:
    """Main configuration class"""
    def __init__(self):
        self.drawing = DrawingConfig()
        self.hand_tracking = HandTrackingConfig()
        self.ml = MLConfig()
        self.camera = CameraConfig()
        self.ui = UIConfig()
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(self.ml.model_path), exist_ok=True)

# Global config instance
config = Config()