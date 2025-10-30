"""
Advanced gesture detection for Virtual Board
"""
import numpy as np
from typing import List, Tuple, Optional
from enum import Enum
from collections import deque
import mediapipe as mp

class GestureType(Enum):
    """Enumeration of gesture types"""
    NONE = "none"
    DRAW = "draw"
    ERASE = "erase"
    RECOGNIZE = "recognize"
    CLEAR = "clear"
    SAVE = "save"

class GestureDetector:
    """Advanced gesture detection using MediaPipe landmarks"""
    
    def __init__(self, smoothing_window: int = 5):
        self.smoothing_window = smoothing_window
        self.gesture_history = deque(maxlen=smoothing_window)
        
        # Finger landmark indices
        self.FINGER_TIPS = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        self.FINGER_PIPS = [3, 6, 10, 14, 18]  # PIP joints
        self.FINGER_MCPS = [2, 5, 9, 13, 17]   # MCP joints
        
    def is_finger_up(self, landmarks: List, finger_idx: int) -> bool:
        """Check if a specific finger is up"""
        if finger_idx == 0:  # Thumb (special case)
            return landmarks[self.FINGER_TIPS[0]].x > landmarks[self.FINGER_PIPS[0]].x
        else:
            return landmarks[self.FINGER_TIPS[finger_idx]].y < landmarks[self.FINGER_PIPS[finger_idx]].y
    
    def get_fingers_up(self, landmarks: List) -> List[bool]:
        """Get list of which fingers are up"""
        return [self.is_finger_up(landmarks, i) for i in range(5)]
    
    def detect_basic_gesture(self, landmarks: List) -> GestureType:
        """Detect basic gestures based on finger positions"""
        fingers_up = self.get_fingers_up(landmarks)
        
        # Count fingers up
        fingers_count = sum(fingers_up)
        
        # Gesture patterns
        if fingers_up == [False, True, False, False, False]:  # Only index
            return GestureType.DRAW
        elif fingers_up == [False, True, True, False, False]:  # Index + Middle
            return GestureType.ERASE
        elif fingers_up == [False, True, True, True, False]:  # Index + Middle + Ring
            return GestureType.RECOGNIZE
        elif fingers_up == [True, True, True, True, True]:  # All fingers
            return GestureType.CLEAR
        elif fingers_up == [True, False, False, False, True]:  # Thumb + Pinky
            return GestureType.SAVE
        else:
            return GestureType.NONE
    
    def detect_advanced_gesture(self, landmarks: List) -> Tuple[GestureType, dict]:
        """Detect advanced gestures with additional information"""
        basic_gesture = self.detect_basic_gesture(landmarks)
        
        # Additional gesture information
        gesture_info = {
            'confidence': self.calculate_gesture_confidence(landmarks, basic_gesture),
            'hand_center': self.get_hand_center(landmarks),
            'index_tip': self.get_finger_tip_position(landmarks, 1),
            'gesture_stability': self.get_gesture_stability(basic_gesture)
        }
        
        return basic_gesture, gesture_info
    
    def calculate_gesture_confidence(self, landmarks: List, gesture: GestureType) -> float:
        """Calculate confidence score for detected gesture"""
        fingers_up = self.get_fingers_up(landmarks)
        
        # Define expected patterns for each gesture
        expected_patterns = {
            GestureType.DRAW: [False, True, False, False, False],
            GestureType.ERASE: [False, True, True, False, False],
            GestureType.RECOGNIZE: [False, True, True, True, False],
            GestureType.CLEAR: [True, True, True, True, True],
            GestureType.SAVE: [True, False, False, False, True],
            GestureType.NONE: [False, False, False, False, False]
        }
        
        if gesture in expected_patterns:
            expected = expected_patterns[gesture]
            matches = sum(1 for i, (actual, exp) in enumerate(zip(fingers_up, expected)) if actual == exp)
            return matches / len(expected)
        
        return 0.0
    
    def get_hand_center(self, landmarks: List) -> Tuple[float, float]:
        """Get center point of hand"""
        x_coords = [lm.x for lm in landmarks]
        y_coords = [lm.y for lm in landmarks]
        
        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)
        
        return (center_x, center_y)
    
    def get_finger_tip_position(self, landmarks: List, finger_idx: int) -> Tuple[float, float]:
        """Get position of specific finger tip"""
        tip_landmark = landmarks[self.FINGER_TIPS[finger_idx]]
        return (tip_landmark.x, tip_landmark.y)
    
    def get_gesture_stability(self, current_gesture: GestureType) -> float:
        """Calculate gesture stability based on history"""
        self.gesture_history.append(current_gesture)
        
        if len(self.gesture_history) < 2:
            return 0.0
        
        # Count how many recent gestures match the current one
        matches = sum(1 for g in self.gesture_history if g == current_gesture)
        return matches / len(self.gesture_history)
    
    def is_gesture_stable(self, threshold: float = 0.7) -> bool:
        """Check if current gesture is stable"""
        if not self.gesture_history:
            return False
        
        current_gesture = self.gesture_history[-1]
        stability = self.get_gesture_stability(current_gesture)
        return stability >= threshold
    
    def detect_pinch_gesture(self, landmarks: List) -> Tuple[bool, float]:
        """Detect pinch gesture between thumb and index finger"""
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        
        # Calculate distance between thumb and index finger tips
        distance = np.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
        
        # Pinch threshold (adjust based on testing)
        pinch_threshold = 0.05
        is_pinching = distance < pinch_threshold
        
        return is_pinching, distance
    
    def detect_swipe_gesture(self, landmarks: List, previous_landmarks: Optional[List] = None) -> Tuple[bool, str]:
        """Detect swipe gestures"""
        if previous_landmarks is None:
            return False, "none"
        
        # Use index finger tip for swipe detection
        current_tip = landmarks[8]
        previous_tip = previous_landmarks[8]
        
        # Calculate movement
        dx = current_tip.x - previous_tip.x
        dy = current_tip.y - previous_tip.y
        
        # Minimum movement threshold
        movement_threshold = 0.1
        
        if abs(dx) > movement_threshold or abs(dy) > movement_threshold:
            # Determine swipe direction
            if abs(dx) > abs(dy):
                direction = "right" if dx > 0 else "left"
            else:
                direction = "down" if dy > 0 else "up"
            
            return True, direction
        
        return False, "none"
    
    def reset_history(self):
        """Reset gesture history"""
        self.gesture_history.clear()