"""
Core Virtual Board implementation with all features
"""
import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple, List
import time

from src.config import config
from src.gesture_detector import GestureDetector, GestureType
from src.ml_models import ModelManager
from src.utils import (
    save_canvas, draw_text_with_background, 
    create_gradient_background, FPSCounter, smooth_points
)

class VirtualBoardCore:
    """Core Virtual Board with advanced features"""
    
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=config.hand_tracking.static_image_mode,
            max_num_hands=config.hand_tracking.max_num_hands,
            min_detection_confidence=config.hand_tracking.min_detection_confidence,
            min_tracking_confidence=config.hand_tracking.min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize components
        self.gesture_detector = GestureDetector()
        self.model_manager = ModelManager()
        self.fps_counter = FPSCounter()
        
        # Canvas and drawing state
        self.canvas = None
        self.background = None
        self.drawing_points = []
        self.current_stroke = []
        
        # Recognition area
        self.recognition_area = None
        self.last_prediction = ""
        self.prediction_confidence = 0.0
        
        # Drawing state
        self.is_drawing = False
        self.last_point = None
        self.current_gesture = GestureType.NONE
        
        # UI state
        self.show_landmarks = True
        self.show_fps = True
        self.show_instructions = True
        
    def initialize(self, width: int, height: int) -> None:
        """Initialize canvas and components"""
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self.background = create_gradient_background(width, height)
        
        # Setup recognition area (bottom-right corner)
        margin = 20
        area_size = 150
        self.recognition_area = {
            'x1': width - area_size - margin,
            'y1': height - area_size - margin,
            'x2': width - margin,
            'y2': height - margin
        }
        
        # Setup ML models
        print("Setting up ML models...")
        self.model_manager.setup_models()
        print("Virtual Board initialized!")
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame for hand detection and gesture recognition"""
        height, width = frame.shape[:2]
        
        # Initialize if needed
        if self.canvas is None:
            self.initialize(width, height)
        
        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process hands
        results = self.hands.process(rgb_frame)
        
        # Draw UI elements
        self._draw_ui_elements(frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks if enabled
                if self.show_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
                
                # Detect gesture
                gesture, gesture_info = self.gesture_detector.detect_advanced_gesture(
                    hand_landmarks.landmark
                )
                
                # Process gesture
                self._process_gesture(frame, gesture, gesture_info, width, height)
        
        # Update FPS
        fps = self.fps_counter.update()
        
        # Draw FPS if enabled
        if self.show_fps:
            draw_text_with_background(frame, f"FPS: {fps:.1f}", (10, 30))
        
        return frame
    
    def _process_gesture(self, frame: np.ndarray, gesture: GestureType, 
                        gesture_info: dict, width: int, height: int) -> None:
        """Process detected gesture"""
        self.current_gesture = gesture
        
        # Get finger tip position
        tip_x = int(gesture_info['index_tip'][0] * width)
        tip_y = int(gesture_info['index_tip'][1] * height)
        
        if gesture == GestureType.DRAW:
            self._handle_draw_gesture(frame, tip_x, tip_y, gesture_info)
        elif gesture == GestureType.ERASE:
            self._handle_erase_gesture(frame, tip_x, tip_y)
        elif gesture == GestureType.RECOGNIZE:
            self._handle_recognize_gesture(frame, tip_x, tip_y)
        elif gesture == GestureType.CLEAR:
            self._handle_clear_gesture(frame)
        elif gesture == GestureType.SAVE:
            self._handle_save_gesture(frame)
        else:
            self._handle_no_gesture()
    
    def _handle_draw_gesture(self, frame: np.ndarray, x: int, y: int, gesture_info: dict) -> None:
        """Handle drawing gesture"""
        # Draw cursor
        cv2.circle(frame, (x, y), 8, config.drawing.drawing_color, -1)
        
        # Draw gesture status
        draw_text_with_background(frame, "DRAW", (10, 70), color=(0, 255, 0))
        
        # Add to current stroke
        self.current_stroke.append((x, y))
        
        # Draw on canvas
        if self.last_point is not None and gesture_info['gesture_stability'] > 0.5:
            cv2.line(self.canvas, self.last_point, (x, y), 
                    config.drawing.drawing_color, config.drawing.brush_thickness)
        
        self.last_point = (x, y)
        self.is_drawing = True
    
    def _handle_erase_gesture(self, frame: np.ndarray, x: int, y: int) -> None:
        """Handle erasing gesture"""
        # Draw cursor
        cv2.circle(frame, (x, y), config.drawing.eraser_thickness, (0, 0, 255), 2)
        
        # Draw gesture status
        draw_text_with_background(frame, "ERASE", (10, 70), color=(0, 0, 255))
        
        # Erase on canvas
        cv2.circle(self.canvas, (x, y), config.drawing.eraser_thickness, 
                  config.drawing.eraser_color, -1)
        
        self.last_point = None
        self.is_drawing = False
    
    def _handle_recognize_gesture(self, frame: np.ndarray, x: int, y: int) -> None:
        """Handle recognition gesture"""
        # Draw cursor
        cv2.circle(frame, (x, y), 10, (255, 0, 255), -1)
        
        # Draw gesture status
        draw_text_with_background(frame, "RECOGNIZE", (10, 70), color=(255, 0, 255))
        
        # Perform recognition
        self._perform_recognition()
        
        self.last_point = None
        self.is_drawing = False
    
    def _handle_clear_gesture(self, frame: np.ndarray) -> None:
        """Handle clear gesture"""
        draw_text_with_background(frame, "CLEAR", (10, 70), color=(255, 255, 0))
        self.clear_canvas()
    
    def _handle_save_gesture(self, frame: np.ndarray) -> None:
        """Handle save gesture"""
        draw_text_with_background(frame, "SAVE", (10, 70), color=(0, 255, 255))
        self.save_current_canvas()
    
    def _handle_no_gesture(self) -> None:
        """Handle no gesture detected"""
        if self.is_drawing and self.current_stroke:
            # Smooth the completed stroke
            if len(self.current_stroke) > 2:
                smoothed_stroke = smooth_points(self.current_stroke)
                self.drawing_points.append(smoothed_stroke)
            self.current_stroke = []
        
        self.last_point = None
        self.is_drawing = False
    
    def _perform_recognition(self) -> None:
        """Perform digit/letter recognition on recognition area"""
        if self.recognition_area is None:
            return
        
        # Extract recognition area
        x1, y1 = self.recognition_area['x1'], self.recognition_area['y1']
        x2, y2 = self.recognition_area['x2'], self.recognition_area['y2']
        
        roi = self.canvas[y1:y2, x1:x2]
        
        if np.sum(roi) > 0:  # Check if there's something drawn
            try:
                prediction, confidence = self.model_manager.predict(roi)
                self.last_prediction = prediction
                self.prediction_confidence = confidence
                print(f"Predicted: {prediction} (Confidence: {confidence:.3f})")
            except Exception as e:
                print(f"Recognition error: {e}")
    
    def _draw_ui_elements(self, frame: np.ndarray) -> None:
        """Draw UI elements on frame"""
        height, width = frame.shape[:2]
        
        # Draw recognition area
        if self.recognition_area:
            cv2.rectangle(frame,
                         (self.recognition_area['x1'], self.recognition_area['y1']),
                         (self.recognition_area['x2'], self.recognition_area['y2']),
                         (255, 255, 0), 2)
            
            draw_text_with_background(frame, "Recognition Area",
                                    (self.recognition_area['x1'], self.recognition_area['y1'] - 10),
                                    font_scale=0.5, color=(255, 255, 0))
        
        # Draw prediction
        if self.last_prediction:
            prediction_text = f"Predicted: {self.last_prediction} ({self.prediction_confidence:.2f})"
            draw_text_with_background(frame, prediction_text,
                                    (10, height - 30), color=(255, 255, 255))
        
        # Draw current mode
        mode_text = f"Mode: {self.model_manager.current_mode.upper()}"
        draw_text_with_background(frame, mode_text, (10, height - 60), 
                                color=(200, 200, 200), font_scale=0.7)
        
        # Draw instructions if enabled
        if self.show_instructions:
            instructions = [
                "1 finger: Draw",
                "2 fingers: Erase", 
                "3 fingers: Recognize",
                "5 fingers: Clear",
                "Thumb+Pinky: Save"
            ]
            
            for i, instruction in enumerate(instructions):
                y_pos = 100 + i * 25
                draw_text_with_background(frame, instruction, (width - 200, y_pos),
                                        font_scale=0.5, color=(200, 200, 200))
    
    def get_combined_view(self, frame: np.ndarray) -> np.ndarray:
        """Get combined view of camera and canvas"""
        if self.canvas is not None:
            return cv2.addWeighted(frame, 0.7, self.canvas, 0.3, 0)
        return frame
    
    def get_canvas_view(self) -> np.ndarray:
        """Get canvas-only view"""
        if self.canvas is not None:
            return cv2.addWeighted(self.background, 0.1, self.canvas, 0.9, 0)
        return self.background if self.background is not None else np.zeros((480, 640, 3), dtype=np.uint8)
    
    def clear_canvas(self) -> None:
        """Clear the canvas"""
        if self.canvas is not None:
            self.canvas.fill(0)
            self.drawing_points.clear()
            self.current_stroke.clear()
            self.last_prediction = ""
            self.prediction_confidence = 0.0
            print("Canvas cleared!")
    
    def save_current_canvas(self) -> str:
        """Save current canvas to file"""
        if self.canvas is not None:
            filepath = save_canvas(self.canvas)
            print(f"Canvas saved to: {filepath}")
            return filepath
        return ""
    
    def toggle_landmarks(self) -> None:
        """Toggle hand landmarks display"""
        self.show_landmarks = not self.show_landmarks
    
    def toggle_fps(self) -> None:
        """Toggle FPS display"""
        self.show_fps = not self.show_fps
    
    def toggle_instructions(self) -> None:
        """Toggle instructions display"""
        self.show_instructions = not self.show_instructions
    
    def switch_recognition_mode(self) -> None:
        """Switch between digit and letter recognition"""
        current_mode = self.model_manager.current_mode
        new_mode = "letter" if current_mode == "digit" else "digit"
        self.model_manager.switch_mode(new_mode)
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        if hasattr(self, 'hands'):
            self.hands.close()