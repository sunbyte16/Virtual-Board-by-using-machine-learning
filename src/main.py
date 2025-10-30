"""
Main application entry point for Virtual Board
"""
import cv2
import argparse
import sys
from typing import Optional

from src.virtual_board_core import VirtualBoardCore
from src.config import config

class VirtualBoardApp:
    """Main Virtual Board Application"""
    
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.board = VirtualBoardCore()
        self.cap = None
        self.running = False
        
    def initialize_camera(self) -> bool:
        """Initialize camera capture"""
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_index}")
            return False
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, config.camera.fps)
        
        return True
    
    def print_instructions(self) -> None:
        """Print usage instructions"""
        print("\n" + "="*60)
        print("VIRTUAL BOARD - MACHINE LEARNING EDITION")
        print("="*60)
        print("\nHAND GESTURES:")
        print("  👆 Index finger up          → Draw")
        print("  ✌️  Index + Middle up        → Erase")
        print("  🤟 Index + Middle + Ring up → Recognize digit/letter")
        print("  🖐️  All fingers up          → Clear canvas")
        print("  🤘 Thumb + Pinky up         → Save canvas")
        print("\nKEYBOARD CONTROLS:")
        print("  'c' → Clear canvas")
        print("  'r' → Recognize in recognition area")
        print("  's' → Save canvas")
        print("  'm' → Switch recognition mode (digit/letter)")
        print("  'l' → Toggle hand landmarks")
        print("  'f' → Toggle FPS display")
        print("  'i' → Toggle instructions")
        print("  'q' → Quit application")
        print("\nTIPS:")
        print("  • Ensure good lighting for better hand detection")
        print("  • Keep hand steady for stable gesture recognition")
        print("  • Draw digits/letters in the yellow recognition area")
        print("  • Use smooth movements for better drawing experience")
        print("="*60)
    
    def handle_keyboard_input(self, key: int) -> bool:
        """Handle keyboard input"""
        if key == ord('q'):
            return False
        elif key == ord('c'):
            self.board.clear_canvas()
        elif key == ord('r'):
            self.board._perform_recognition()
        elif key == ord('s'):
            self.board.save_current_canvas()
        elif key == ord('m'):
            self.board.switch_recognition_mode()
        elif key == ord('l'):
            self.board.toggle_landmarks()
        elif key == ord('f'):
            self.board.toggle_fps()
        elif key == ord('i'):
            self.board.toggle_instructions()
        
        return True
    
    def run(self) -> None:
        """Main application loop"""
        if not self.initialize_camera():
            return
        
        self.print_instructions()
        self.running = True
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Could not read frame from camera")
                    break
                
                # Process frame
                processed_frame = self.board.process_frame(frame)
                
                # Get different views
                combined_view = self.board.get_combined_view(processed_frame)
                canvas_view = self.board.get_canvas_view()
                
                # Display windows
                cv2.imshow('Virtual Board - Combined View', combined_view)
                cv2.imshow('Virtual Board - Canvas Only', canvas_view)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if not self.handle_keyboard_input(key):
                    break
                    
        except KeyboardInterrupt:
            print("\nApplication interrupted by user")
        except Exception as e:
            print(f"Error during execution: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        self.running = False
        
        if self.cap is not None:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        if self.board is not None:
            self.board.cleanup()
        
        print("Virtual Board closed successfully!")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Virtual Board with Machine Learning')
    parser.add_argument('--camera', type=int, default=0, 
                       help='Camera index (default: 0)')
    parser.add_argument('--width', type=int, default=640,
                       help='Camera frame width (default: 640)')
    parser.add_argument('--height', type=int, default=480,
                       help='Camera frame height (default: 480)')
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    config.camera.camera_index = args.camera
    config.camera.frame_width = args.width
    config.camera.frame_height = args.height
    
    # Create and run application
    app = VirtualBoardApp(camera_index=args.camera)
    app.run()

if __name__ == "__main__":
    main()