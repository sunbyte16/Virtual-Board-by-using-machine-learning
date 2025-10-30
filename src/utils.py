"""
Utility functions for Virtual Board
"""
import cv2
import numpy as np
import os
import json
from datetime import datetime
from typing import Optional, Tuple, List

def create_directory(path: str) -> None:
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)

def save_canvas(canvas: np.ndarray, filename: Optional[str] = None) -> str:
    """Save canvas to file"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"canvas_{timestamp}.png"
    
    # Create saves directory
    save_dir = "saves"
    create_directory(save_dir)
    
    filepath = os.path.join(save_dir, filename)
    cv2.imwrite(filepath, canvas)
    return filepath

def load_canvas(filepath: str) -> Optional[np.ndarray]:
    """Load canvas from file"""
    if os.path.exists(filepath):
        return cv2.imread(filepath)
    return None

def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize image to target size"""
    return cv2.resize(image, target_size)

def preprocess_for_ml(image: np.ndarray, target_size: Tuple[int, int] = (28, 28)) -> np.ndarray:
    """Preprocess image for ML model"""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize
    image = cv2.resize(image, target_size)
    
    # Normalize
    image = image.astype('float32') / 255.0
    
    # Add batch and channel dimensions
    image = image.reshape(1, target_size[0], target_size[1], 1)
    
    return image

def calculate_distance(point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
    """Calculate Euclidean distance between two points"""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def smooth_points(points: List[Tuple[int, int]], window_size: int = 3) -> List[Tuple[int, int]]:
    """Smooth a list of points using moving average"""
    if len(points) < window_size:
        return points
    
    smoothed = []
    for i in range(len(points)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(points), i + window_size // 2 + 1)
        
        avg_x = sum(p[0] for p in points[start_idx:end_idx]) / (end_idx - start_idx)
        avg_y = sum(p[1] for p in points[start_idx:end_idx]) / (end_idx - start_idx)
        
        smoothed.append((int(avg_x), int(avg_y)))
    
    return smoothed

def draw_text_with_background(image: np.ndarray, text: str, position: Tuple[int, int], 
                            font_scale: float = 1.0, color: Tuple[int, int, int] = (255, 255, 255),
                            bg_color: Tuple[int, int, int] = (0, 0, 0), thickness: int = 2) -> None:
    """Draw text with background rectangle"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Draw background rectangle
    cv2.rectangle(image, 
                 (position[0] - 5, position[1] - text_height - 5),
                 (position[0] + text_width + 5, position[1] + baseline + 5),
                 bg_color, -1)
    
    # Draw text
    cv2.putText(image, text, position, font, font_scale, color, thickness)

def create_gradient_background(width: int, height: int, 
                             color1: Tuple[int, int, int] = (0, 0, 0),
                             color2: Tuple[int, int, int] = (50, 50, 50)) -> np.ndarray:
    """Create gradient background"""
    background = np.zeros((height, width, 3), dtype=np.uint8)
    
    for i in range(height):
        ratio = i / height
        r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
        g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
        b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
        background[i, :] = [b, g, r]  # BGR format
    
    return background

def save_session_data(data: dict, filename: str = "session.json") -> None:
    """Save session data to JSON file"""
    save_dir = "saves"
    create_directory(save_dir)
    
    filepath = os.path.join(save_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def load_session_data(filename: str = "session.json") -> Optional[dict]:
    """Load session data from JSON file"""
    filepath = os.path.join("saves", filename)
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None

class FPSCounter:
    """FPS counter utility"""
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.frame_times = []
        self.last_time = None
    
    def update(self) -> float:
        """Update FPS counter and return current FPS"""
        current_time = cv2.getTickCount()
        
        if self.last_time is not None:
            frame_time = (current_time - self.last_time) / cv2.getTickFrequency()
            self.frame_times.append(frame_time)
            
            if len(self.frame_times) > self.window_size:
                self.frame_times.pop(0)
        
        self.last_time = current_time
        
        if self.frame_times:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            return 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        
        return 0