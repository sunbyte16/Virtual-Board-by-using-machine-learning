import cv2
import numpy as np
import mediapipe as mp
from collections import deque

class VirtualBoard:
    def __init__(self):
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Drawing parameters
        self.drawing_color = (0, 255, 0)  # Green
        self.eraser_color = (0, 0, 0)    # Black
        self.brush_thickness = 5
        self.eraser_thickness = 20
        
        # Canvas and drawing state
        self.canvas = None
        self.prev_point = None 
   def initialize_canvas(self, width, height):
        """Initialize the drawing canvas"""
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        
    def detect_gesture(self, landmarks):
        """Detect hand gestures for different modes"""
        # Get finger tip and pip coordinates
        index_tip = landmarks[8]
        index_pip = landmarks[6]
        middle_tip = landmarks[12]
        middle_pip = landmarks[10]
        
        # Check if index finger is up
        index_up = index_tip.y < index_pip.y
        middle_up = middle_tip.y < middle_pip.y
        
        if index_up and not middle_up:
            return "draw"
        elif index_up and middle_up:
            return "erase"
        else:
            return "none"    def pr
ocess_frame(self, frame):
        """Process each frame for hand detection and drawing"""
        height, width, _ = frame.shape
        
        if self.canvas is None:
            self.initialize_canvas(width, height)
            
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
                
                # Get gesture
                gesture = self.detect_gesture(hand_landmarks.landmark)
                
                # Get index finger tip coordinates
                index_tip = hand_landmarks.landmark[8]
                x = int(index_tip.x * width)
                y = int(index_tip.y * height)
                
                if gesture == "draw":
                    cv2.circle(frame, (x, y), 10, self.drawing_color, -1)
                    
                    if self.prev_point is not None:
                        cv2.line(self.canvas, self.prev_point, (x, y), 
                                self.drawing_color, self.brush_thickness)
                    self.prev_point = (x, y)
                    
                elif gesture == "erase":
                    cv2.circle(frame, (x, y), 15, (0, 0, 255), 2)
                    cv2.circle(self.canvas, (x, y), self.eraser_thickness, 
                              self.eraser_color, -1)
                    self.prev_point = None
                    
                else:
                    self.prev_point = None
                    
        else:
            self.prev_point = None
            
        return frame    def 
clear_canvas(self):
        """Clear the entire canvas"""
        if self.canvas is not None:
            self.canvas.fill(0)
            
    def run(self):
        """Main loop to run the virtual board"""
        cap = cv2.VideoCapture(0)
        
        print("Virtual Board Controls:")
        print("- Index finger up: Draw")
        print("- Index + Middle finger up: Erase")
        print("- Press 'c' to clear canvas")
        print("- Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process the frame
            processed_frame = self.process_frame(frame)
            
            # Combine camera feed and canvas
            if self.canvas is not None:
                # Create a combined view
                combined = cv2.addWeighted(processed_frame, 0.7, self.canvas, 0.3, 0)
                cv2.imshow('Virtual Board - Camera + Canvas', combined)
                cv2.imshow('Virtual Board - Canvas Only', self.canvas)
            else:
                cv2.imshow('Virtual Board - Camera', processed_frame)
                
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.clear_canvas()
                print("Canvas cleared!")
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    board = VirtualBoard()
    board.run()