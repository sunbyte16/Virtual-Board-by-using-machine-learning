# Changelog

All notable changes to the Virtual Board ML project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-10-28

### Added
- Initial release of Virtual Board with Machine Learning
- Real-time hand tracking using MediaPipe
- Advanced gesture detection system
- CNN-based digit recognition (MNIST trained)
- Virtual canvas with drawing and erasing capabilities
- Multiple viewing modes (combined, canvas-only)
- Save and export functionality
- FPS monitoring and performance optimization
- Comprehensive configuration system
- Automated installation and setup scripts
- Unit tests for core functionality
- Make-based build system
- Professional project structure
- Detailed documentation and README

### Features
- **Hand Gestures**:
  - Index finger: Draw mode
  - Index + Middle: Erase mode
  - Index + Middle + Ring: Recognition mode
  - All fingers: Clear canvas
  - Thumb + Pinky: Save canvas

- **Keyboard Controls**:
  - 'c': Clear canvas
  - 'r': Recognize in area
  - 's': Save canvas
  - 'm': Switch recognition mode
  - 'l': Toggle landmarks
  - 'f': Toggle FPS
  - 'i': Toggle instructions
  - 'q': Quit

- **Technical Features**:
  - Real-time video processing at 30+ FPS
  - Advanced gesture stability detection
  - Confidence scoring for predictions
  - Gradient backgrounds and UI enhancements
  - Modular architecture for easy extension
  - Cross-platform compatibility (Windows, macOS, Linux)

### Technical Details
- Python 3.8+ support
- TensorFlow/Keras for ML models
- OpenCV for computer vision
- MediaPipe for hand tracking
- Comprehensive error handling
- Memory-efficient processing
- GPU acceleration support (optional)

### Project Structure
- Organized source code in `src/` directory
- Utility scripts in `scripts/` directory
- Unit tests in `tests/` directory
- Automated model training and setup
- Professional packaging with setup.py
- MIT license for open source use

## [Unreleased]

### Planned Features
- Letter recognition (A-Z) using EMNIST dataset
- Multi-hand tracking support
- 3D gesture recognition
- Web interface using Flask/FastAPI
- Mobile app development
- AR/VR integration
- Collaborative multi-user mode
- Voice command integration
- Shape recognition and correction
- Cloud synchronization
- Text-to-speech feedback

### Improvements
- Enhanced gesture recognition accuracy
- Better lighting adaptation
- Improved UI/UX design
- Performance optimizations
- Extended test coverage
- Documentation improvements
- Tutorial videos and examples