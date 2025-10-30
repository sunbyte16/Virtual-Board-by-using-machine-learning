"""
Virtual Board Demo Script
Run this to test the basic virtual board functionality
"""

from virtual_board import VirtualBoard

def main():
    print("Starting Virtual Board Demo...")
    print("Make sure your webcam is connected and working!")
    
    try:
        board = VirtualBoard()
        board.run()
    except KeyboardInterrupt:
        print("\nDemo stopped by user")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have installed all requirements: pip install -r requirements.txt")

if __name__ == "__main__":
    main()