import cv2
import numpy as np
import mediapipe as mp
from handwriting_recognition import HandwritingRecognizer
import os

class EnhancedVirtualBoard:
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
        self.recognition_area = None
        
        # Handwriting recognition
        self.recognizer = HandwritingRecognizer()
        self.setup_recognition_model()
        
        # Recognition state
        self.last_prediction = ""
        self.prediction_confidence = 0.0
        
    def setup_recognition_model(self):
        """Setup the handwriting recognition model"""
        model_path = "digit_model.h5"
        
        if os.path.exists(model_path):
            print("Loading pre-trained model...")
            self.recognizer.load_model(model_path)
        else:
            print("Training new model on MNIST...")
            self.recognizer.train_on_mnist()
            self.recognizer.save_model(model_path)
            print("Model saved!")
            
    def initialize_canvas(self, width, height):
        """Initialize the drawing canvas"""
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Define recognition area (bottom right corner)
        self.recognition_area = {
            'x1': width - 200,
            'y1': height - 200,
            'x2': width - 20,
            'y2': height - 20
        }
        
    def detect_gesture(self, landmarks):
        """Detect hand gestures for different modes"""
        # Get finger tip and pip coordinates
        index_tip = landmarks[8]
        index_pip = landmarks[6]
        middle_tip = landmarks[12]
        middle_pip = landmarks[10]
        ring_tip = landmarks[16]
        ring_pip = landmarks[14]
        
        # Check if fingers are up
        index_up = index_tip.y < index_pip.y
        middle_up = middle_tip.y < middle_pip.y
        ring_up = ring_tip.y < ring_pip.y
        
        if index_up and not middle_up and not ring_up:
            return "draw"
        elif index_up and middle_up and not ring_up:
            return "erase"
        elif index_up and middle_up and ring_up:
            return "recognize"
        else:
            return "none"   
 def extract_recognition_area(self):
        """Extract the recognition area for digit prediction"""
        if self.canvas is None or self.recognition_area is None:
            return None
            
        x1, y1 = self.recognition_area['x1'], self.recognition_area['y1']
        x2, y2 = self.recognition_area['x2'], self.recognition_area['y2']
        
        roi = self.canvas[y1:y2, x1:x2]
        return roi
        
    def recognize_digit(self):
        """Recognize digit in the recognition area"""
        roi = self.extract_recognition_area()
        if roi is None:
            return
            
        try:
            digit, confidence = self.recognizer.predict_digit(roi)
            self.last_prediction = str(digit)
            self.prediction_confidence = confidence
            print(f"Predicted: {digit} (Confidence: {confidence:.2f})")
        except Exception as e:
            print(f"Recognition error: {e}")
            
    def process_frame(self, frame):
        """Process each frame for hand detection and drawing"""
        height, width, _ = frame.shape
        
        if self.canvas is None:
            self.initialize_canvas(width, height)
            
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Draw recognition area
        if self.recognition_area:
            cv2.rectangle(frame, 
                         (self.recognition_area['x1'], self.recognition_area['y1']),
                         (self.recognition_area['x2'], self.recognition_area['y2']),
                         (255, 255, 0), 2)
            cv2.putText(frame, "Recognition Area", 
                       (self.recognition_area['x1'], self.recognition_area['y1'] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
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
                    cv2.putText(frame, "DRAW", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    if self.prev_point is not None:
                        cv2.line(self.canvas, self.prev_point, (x, y), 
                                self.drawing_color, self.brush_thickness)
                    self.prev_point = (x, y)
                    
                elif gesture == "erase":
                    cv2.circle(frame, (x, y), 15, (0, 0, 255), 2)
                    cv2.putText(frame, "ERASE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.circle(self.canvas, (x, y), self.eraser_thickness, 
                              self.eraser_color, -1)
                    self.prev_point = None
                    
                elif gesture == "recognize":
                    cv2.circle(frame, (x, y), 10, (255, 0, 255), -1)
                    cv2.putText(frame, "RECOGNIZE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                    self.recognize_digit()
                    self.prev_point = None
                    
                else:
                    self.prev_point = None
                    
        else:
            self.prev_point = None
            
        # Display prediction
        if self.last_prediction:
            cv2.putText(frame, f"Predicted: {self.last_prediction} ({self.prediction_confidence:.2f})", 
                       (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
        return frame
        
    def clear_canvas(self):
        """Clear the entire canvas"""
        if self.canvas is not None:
            self.canvas.fill(0)
            self.last_prediction = ""
            self.prediction_confidence = 0.0
            
    def run(self):
        """Main loop to run the enhanced virtual board"""
        cap = cv2.VideoCapture(0)
        
        print("Enhanced Virtual Board Controls:")
        print("- Index finger up: Draw")
        print("- Index + Middle finger up: Erase")
        print("- Index + Middle + Ring finger up: Recognize digit")
        print("- Press 'c' to clear canvas")
        print("- Press 'r' to recognize digit in recognition area")
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
                cv2.imshow('Enhanced Virtual Board - Camera + Canvas', combined)
                cv2.imshow('Enhanced Virtual Board - Canvas Only', self.canvas)
            else:
                cv2.imshow('Enhanced Virtual Board - Camera', processed_frame)
                
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.clear_canvas()
                print("Canvas cleared!")
            elif key == ord('r'):
                self.recognize_digit()
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    board = EnhancedVirtualBoard()
    board.run()