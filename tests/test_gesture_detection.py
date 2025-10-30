"""
Tests for gesture detection functionality
"""
import unittest
import numpy as np
from unittest.mock import Mock, MagicMock

from src.gesture_detector import GestureDetector, GestureType

class TestGestureDetector(unittest.TestCase):
    
    def setUp(self):
        self.detector = GestureDetector()
    
    def create_mock_landmarks(self, fingers_up):
        """Create mock landmarks for testing"""
        landmarks = []
        
        # Create 21 mock landmarks (MediaPipe hand landmarks)
        for i in range(21):
            landmark = Mock()
            landmark.x = 0.5
            landmark.y = 0.5
            landmarks.append(landmark)
        
        # Set finger positions based on fingers_up
        finger_tips = [4, 8, 12, 16, 20]
        finger_pips = [3, 6, 10, 14, 18]
        
        for i, is_up in enumerate(fingers_up):
            tip_idx = finger_tips[i]
            pip_idx = finger_pips[i]
            
            if i == 0:  # Thumb (special case)
                landmarks[tip_idx].x = 0.6 if is_up else 0.4
                landmarks[pip_idx].x = 0.5
            else:
                landmarks[tip_idx].y = 0.3 if is_up else 0.7
                landmarks[pip_idx].y = 0.5
        
        return landmarks
    
    def test_detect_draw_gesture(self):
        """Test detection of draw gesture (index finger up)"""
        landmarks = self.create_mock_landmarks([False, True, False, False, False])
        gesture = self.detector.detect_basic_gesture(landmarks)
        self.assertEqual(gesture, GestureType.DRAW)
    
    def test_detect_erase_gesture(self):
        """Test detection of erase gesture (index + middle up)"""
        landmarks = self.create_mock_landmarks([False, True, True, False, False])
        gesture = self.detector.detect_basic_gesture(landmarks)
        self.assertEqual(gesture, GestureType.ERASE)
    
    def test_detect_recognize_gesture(self):
        """Test detection of recognize gesture (index + middle + ring up)"""
        landmarks = self.create_mock_landmarks([False, True, True, True, False])
        gesture = self.detector.detect_basic_gesture(landmarks)
        self.assertEqual(gesture, GestureType.RECOGNIZE)
    
    def test_detect_clear_gesture(self):
        """Test detection of clear gesture (all fingers up)"""
        landmarks = self.create_mock_landmarks([True, True, True, True, True])
        gesture = self.detector.detect_basic_gesture(landmarks)
        self.assertEqual(gesture, GestureType.CLEAR)
    
    def test_detect_save_gesture(self):
        """Test detection of save gesture (thumb + pinky up)"""
        landmarks = self.create_mock_landmarks([True, False, False, False, True])
        gesture = self.detector.detect_basic_gesture(landmarks)
        self.assertEqual(gesture, GestureType.SAVE)
    
    def test_detect_no_gesture(self):
        """Test detection when no specific gesture is made"""
        landmarks = self.create_mock_landmarks([False, False, False, False, False])
        gesture = self.detector.detect_basic_gesture(landmarks)
        self.assertEqual(gesture, GestureType.NONE)
    
    def test_gesture_stability(self):
        """Test gesture stability calculation"""
        landmarks = self.create_mock_landmarks([False, True, False, False, False])
        
        # Add same gesture multiple times
        for _ in range(5):
            self.detector.detect_basic_gesture(landmarks)
        
        stability = self.detector.get_gesture_stability(GestureType.DRAW)
        self.assertGreater(stability, 0.8)
    
    def test_hand_center_calculation(self):
        """Test hand center calculation"""
        landmarks = self.create_mock_landmarks([False, False, False, False, False])
        center = self.detector.get_hand_center(landmarks)
        
        # All landmarks are at (0.5, 0.5), so center should be (0.5, 0.5)
        self.assertAlmostEqual(center[0], 0.5, places=1)
        self.assertAlmostEqual(center[1], 0.5, places=1)

if __name__ == '__main__':
    unittest.main()