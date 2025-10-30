<div align="center">

# 🎨 Virtual Board using Machine Learning

### _An interactive digital platform where users can write or draw in the air using hand gestures, with real-time recognition powered by machine learning and computer vision._

---

[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen?style=for-the-badge&logo=checkmarx&logoColor=white)](https://github.com/sunbyte16)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-red?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-green?style=for-the-badge&logo=google&logoColor=white)](https://mediapipe.dev)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge&logo=opensourceinitiative&logoColor=white)](LICENSE)

[![GitHub Stars](https://img.shields.io/github/stars/sunbyte16/virtual-board-ml?style=social)](https://github.com/sunbyte16)
[![GitHub Forks](https://img.shields.io/github/forks/sunbyte16/virtual-board-ml?style=social)](https://github.com/sunbyte16)
[![GitHub Issues](https://img.shields.io/github/issues/sunbyte16/virtual-board-ml?style=social)](https://github.com/sunbyte16/virtual-board-ml/issues)

---

### 🚀 **Live Demo** | 📖 **Documentation** | 🐛 **Report Bug** | 💡 **Request Feature**

---

</div>

## ✨ **Key Features**

<div align="center">

| 🎯 **Core Features**           | 🚀 **Advanced Features**       | 🎨 **UI/UX Features**        |
| ------------------------------ | ------------------------------ | ---------------------------- |
| 🖐️ **Real-time Hand Tracking** | 🧠 **ML-Powered Recognition**  | 🎨 **Beautiful Gradient UI** |
| ✏️ **Gesture-based Drawing**   | 📊 **Performance Monitoring**  | 📱 **Responsive Interface**  |
| 🎯 **Multi-gesture Support**   | 💾 **Auto-save Functionality** | 🌈 **Customizable Themes**   |
| 🖼️ **Virtual Canvas**          | 🔄 **Real-time Processing**    | 📈 **Live FPS Counter**      |
| 🎮 **Multiple Drawing Modes**  | 🎪 **Smooth Animations**       | 🎭 **Interactive Elements**  |

</div>

### 🔥 **Feature Highlights**

<div align="center">

![Hand Tracking](https://img.shields.io/badge/🖐️%20Hand%20Tracking-MediaPipe%20Powered-blue?style=flat-square)
![Gesture Recognition](https://img.shields.io/badge/✋%20Gesture%20Recognition-5%20Gestures-green?style=flat-square)
![ML Recognition](https://img.shields.io/badge/🧠%20ML%20Recognition-CNN%20Based-purple?style=flat-square)
![Real Time](https://img.shields.io/badge/⚡%20Real%20Time-30%2B%20FPS-orange?style=flat-square)
![Cross Platform](https://img.shields.io/badge/🌐%20Cross%20Platform-Windows%20|%20macOS%20|%20Linux-red?style=flat-square)

</div>

## 🚀 **Quick Start Guide**

<div align="center">

### 🎯 **Get Started in 3 Simple Steps!**

</div>

### 📋 **Prerequisites**

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python&logoColor=white)
![Webcam](https://img.shields.io/badge/Webcam-Required-green?style=flat-square&logo=camera&logoColor=white)
![RAM](https://img.shields.io/badge/RAM-4GB%2B-orange?style=flat-square&logo=memory&logoColor=white)
![OS](https://img.shields.io/badge/OS-Windows%20|%20macOS%20|%20Linux-red?style=flat-square&logo=windows&logoColor=white)

</div>

### 🎯 **Installation Options**

<details>
<summary><b>🚀 Option 1: Automated Setup (Recommended)</b></summary>

```bash
# 🔽 Clone the repository
git clone https://github.com/sunbyte16/virtual-board-ml.git
cd virtual-board-ml

# 🛠️ Install everything automatically
make install

# 🧠 Train ML models
make train

# 🎮 Run the application
make run
```

</details>

<details>
<summary><b>⚙️ Option 2: Manual Setup</b></summary>

```bash
# 📦 Install dependencies
pip install -r requirements.txt

# 🔧 Run installation script
python scripts/install.py

# 🤖 Train ML models
python scripts/train_models.py

# 🚀 Launch the application
python -m src.main
```

</details>

<details>
<summary><b>🎪 Option 3: Quick Demo</b></summary>

```bash
# 🎯 Run basic demo (no ML training required)
python demo.py
```

</details>

### 🎮 **Launch Commands**

<div align="center">

| Command                     | Description             | Use Case                              |
| --------------------------- | ----------------------- | ------------------------------------- |
| `make run`                  | 🚀 **Full Application** | Complete experience with all features |
| `python demo.py`            | 🎪 **Quick Demo**       | Basic functionality testing           |
| `python -m src.main --help` | ❓ **Help & Options**   | View all available options            |

</div>

## 🎮 **Controls & Usage**

### 🖐️ **Hand Gestures**

<div align="center">

| Gesture                  | Visual                                                                           | Action           | Description                      |
| ------------------------ | -------------------------------------------------------------------------------- | ---------------- | -------------------------------- |
| � ** Index Finger Up**   | ![Draw](https://img.shields.io/badge/✏️-Draw-green?style=flat-square)            | **Draw Mode**    | Draw smooth lines on canvas      |
| ✌️ **Index + Middle Up** | ![Erase](https://img.shields.io/badge/🧽-Erase-red?style=flat-square)            | **Erase Mode**   | Remove content from canvas       |
| 🤟 **Three Fingers Up**  | ![Recognize](https://img.shields.io/badge/🧠-Recognize-purple?style=flat-square) | **Recognition**  | Identify digits/letters using AI |
| 🖐️ **All Fingers Up**    | ![Clear](https://img.shields.io/badge/🗑️-Clear-orange?style=flat-square)         | **Clear Canvas** | Reset entire drawing area        |
| 🤘 **Thumb + Pinky**     | ![Save](https://img.shields.io/badge/💾-Save-blue?style=flat-square)             | **Save Canvas**  | Export drawing as image          |

</div>

### ⌨️ **Keyboard Shortcuts**

<div align="center">

| Key | Action                     | Description                     |
| --- | -------------------------- | ------------------------------- |
| `C` | 🗑️ **Clear Canvas**        | Reset the drawing area          |
| `R` | 🧠 **Recognize**           | Analyze recognition area        |
| `S` | 💾 **Save Canvas**         | Export current drawing          |
| `M` | 🔄 **Switch Mode**         | Toggle digit/letter recognition |
| `L` | 🖐️ **Toggle Landmarks**    | Show/hide hand tracking points  |
| `F` | 📊 **Toggle FPS**          | Display performance metrics     |
| `I` | ℹ️ **Toggle Instructions** | Show/hide help overlay          |
| `Q` | 🚪 **Quit**                | Exit the application            |

</div>

## 📁 **Project Architecture**

<div align="center">

```
🏗️ virtual-board-ml/
├── 📂 src/                          # 🧠 Core Application Logic
│   ├── 🚀 main.py                   # Application entry point
│   ├── 🎨 virtual_board_core.py     # Main board implementation
│   ├── 🖐️ gesture_detector.py       # Advanced gesture recognition
│   ├── 🤖 ml_models.py              # Machine learning models
│   ├── ⚙️ config.py                 # Configuration management
│   └── 🛠️ utils.py                  # Utility functions
├── 📂 scripts/                      # 🔧 Automation Scripts
│   ├── 🛠️ install.py                # Automated installation
│   ├── 🧠 train_models.py           # ML model training
│   └── 🧪 run_tests.py              # Test execution
├── 📂 tests/                        # ✅ Quality Assurance
│   ├── 🧪 test_gesture_detection.py # Gesture testing
│   └── 📋 __init__.py               # Test package init
├── 📂 models/                       # 🤖 Trained AI Models (auto-generated)
├── 📂 saves/                        # 💾 Saved Drawings (auto-generated)
├── � reqeuirements.txt              # 📦 Python dependencies
├── ⚙️ setup.py                      # 📦 Package configuration
├── 🛠️ Makefile                      # 🔨 Build automation
└── 📖 README.md                     # 📚 This documentation
```

</div>

## 🛠️ **Development Toolkit**

### 🔨 **Available Make Commands**

<div align="center">

| Command        | Purpose            | Description                            |
| -------------- | ------------------ | -------------------------------------- |
| `make help`    | ❓ **Help**        | Show all available commands            |
| `make install` | 🛠️ **Setup**       | Install dependencies and setup project |
| `make train`   | 🧠 **AI Training** | Train machine learning models          |
| `make test`    | 🧪 **Testing**     | Run comprehensive test suite           |
| `make run`     | 🚀 **Launch**      | Start the Virtual Board application    |
| `make demo`    | 🎪 **Demo**        | Run basic demonstration                |
| `make dev`     | 👨‍💻 **Development** | Setup development environment          |
| `make clean`   | 🧹 **Cleanup**     | Remove generated files                 |
| `make format`  | 🎨 **Format**      | Format code with Black                 |
| `make lint`    | 🔍 **Lint**        | Check code quality with Flake8         |

</div>

### 🧪 **Testing & Quality**

```bash
# 🧪 Run all tests
make test

# 🔍 Check code quality
make lint

# 🎨 Format code
make format

# 🔄 Full quality check
make check
```

### 🤖 **Model Training**

```bash
# 🧠 Train all models
make train

# 🔢 Train digit recognition only
python scripts/train_models.py --model digit

# 🔤 Train letter recognition only
python scripts/train_models.py --model letter

# ⚡ Custom training with specific epochs
python scripts/train_models.py --model both --epochs 10
```

## 🔧 **Technical Deep Dive**

### 🖐️ **Hand Tracking Technology**

<div align="center">

![MediaPipe](https://img.shields.io/badge/MediaPipe-21%20Landmarks-blue?style=flat-square&logo=google)
![Accuracy](https://img.shields.io/badge/Accuracy-Sub--pixel-green?style=flat-square)
![Performance](https://img.shields.io/badge/Performance-Real--time-orange?style=flat-square)

</div>

- **🎯 MediaPipe Hands**: 21 landmark detection with sub-pixel accuracy
- **🔄 Gesture Recognition**: Advanced finger position analysis
- **📈 Smoothing**: Temporal smoothing for stable gesture detection
- **🎪 Multi-gesture Support**: Simultaneous detection of multiple gesture types

### 🧠 **Machine Learning Pipeline**

<div align="center">

![CNN](https://img.shields.io/badge/Architecture-CNN-purple?style=flat-square&logo=tensorflow)
![MNIST](https://img.shields.io/badge/Dataset-MNIST-blue?style=flat-square)
![Accuracy](https://img.shields.io/badge/Accuracy-99%25%2B-green?style=flat-square)

</div>

- **🏗️ CNN Architecture**: Custom convolutional neural networks
- **📊 MNIST Training**: Pre-trained on 70,000 digit samples
- **⚡ Real-time Inference**: Optimized for live prediction
- **📦 Batch Processing**: Support for multiple predictions
- **💾 Model Management**: Automatic model loading/saving

### 👁️ **Computer Vision Stack**

<div align="center">

![OpenCV](https://img.shields.io/badge/OpenCV-Video%20Processing-red?style=flat-square&logo=opencv)
![FPS](https://img.shields.io/badge/Performance-30%2B%20FPS-orange?style=flat-square)
![Resolution](https://img.shields.io/badge/Resolution-HD%20Ready-blue?style=flat-square)

</div>

- **📹 OpenCV Integration**: Real-time video processing
- **🖥️ Multi-window Display**: Separate camera and canvas views
- **🔄 Image Processing**: Advanced preprocessing for ML models
- **⚡ Performance Optimization**: 30+ FPS on standard hardware

## 🎯 **Advanced Features**

### 🎪 **Gesture Stability System**

<div align="center">

![Stability](https://img.shields.io/badge/Temporal-Smoothing-blue?style=flat-square)
![Confidence](https://img.shields.io/badge/Confidence-Scoring-green?style=flat-square)
![Adaptive](https://img.shields.io/badge/Adaptive-Thresholds-orange?style=flat-square)

</div>

- **📈 Temporal Smoothing**: Stable gesture recognition over time
- **🎯 Confidence Scoring**: Real-time gesture confidence metrics
- **🔄 Adaptive Thresholds**: Dynamic adjustment based on hand movement

### 🧠 **Recognition Modes**

<div align="center">

![Digits](https://img.shields.io/badge/Digits-0--9-blue?style=flat-square)
![Letters](https://img.shields.io/badge/Letters-A--Z-green?style=flat-square)
![Confidence](https://img.shields.io/badge/Confidence-Real--time-purple?style=flat-square)

</div>

- **🔢 Digit Mode**: 0-9 digit recognition with high accuracy
- **🔤 Letter Mode**: A-Z letter recognition (extensible)
- **📊 Confidence Scoring**: Real-time prediction confidence display

### 🎨 **UI Enhancement Suite**

<div align="center">

![FPS](https://img.shields.io/badge/FPS-Counter-orange?style=flat-square)
![Gradient](https://img.shields.io/badge/Gradient-Backgrounds-purple?style=flat-square)
![Overlay](https://img.shields.io/badge/Text-Overlays-blue?style=flat-square)

</div>

- **📊 FPS Counter**: Real-time performance monitoring
- **🌈 Gradient Backgrounds**: Beautiful visual effects
- **📝 Text Overlays**: Clear status and instruction display
- **🎯 Recognition Area**: Dedicated area for ML predictions

## 🚀 **Future Roadmap**

<div align="center">

### 🗺️ **Development Phases**

</div>

| Phase          | Status             | Timeline | Features                                                    |
| -------------- | ------------------ | -------- | ----------------------------------------------------------- |
| 🎯 **Phase 1** | ✅ **Complete**    | Q4 2024  | Basic hand tracking, gesture recognition, digit recognition |
| 🚀 **Phase 2** | 🔄 **In Progress** | Q1 2025  | Letter recognition, multi-hand support, improved UI         |
| 🌟 **Phase 3** | 📋 **Planned**     | Q2 2025  | Web interface, mobile app, AR integration                   |
| 🔮 **Phase 4** | 💭 **Future**      | Q3 2025  | 3D gestures, voice commands, collaborative mode             |

### 🎪 **Upcoming Features**

<div align="center">

![Multi-hand](https://img.shields.io/badge/🖐️-Multi--hand%20Support-blue?style=flat-square)
![3D](https://img.shields.io/badge/🎯-3D%20Gesture%20Recognition-green?style=flat-square)
![Voice](https://img.shields.io/badge/🎤-Voice%20Commands-purple?style=flat-square)
![AR](https://img.shields.io/badge/🥽-AR%20Integration-orange?style=flat-square)
![Web](https://img.shields.io/badge/🌐-Web%20Interface-red?style=flat-square)
![Mobile](https://img.shields.io/badge/📱-Mobile%20App-yellow?style=flat-square)

</div>

- [ ] **🖐️ Multi-hand Support**: Track multiple hands simultaneously
- [ ] **🎯 3D Gesture Recognition**: Depth-based gesture detection
- [ ] **🎤 Voice Commands**: Voice-controlled canvas operations
- [ ] **🥽 AR Integration**: Augmented reality overlay
- [ ] **🌐 Web Interface**: Browser-based virtual board
- [ ] **📱 Mobile App**: iOS/Android applications
- [ ] **👥 Collaborative Mode**: Multi-user shared canvas
- [ ] **🔷 Shape Recognition**: Automatic shape detection and correction
- [ ] **🔊 Text-to-Speech**: Audio feedback for recognized content
- [ ] **☁️ Cloud Sync**: Save and sync across devices

## 🐛 **Troubleshooting Guide**

### 🔧 **Common Issues & Solutions**

<details>
<summary><b>📷 Camera Not Detected</b></summary>

```bash
# 🔍 Check camera availability
python -c "import cv2; print('Camera OK' if cv2.VideoCapture(0).isOpened() else 'Camera Error')"

# 🛠️ Solutions:
# 1. Ensure camera is connected and not used by other apps
# 2. Try different camera index (0, 1, 2...)
# 3. Check camera permissions in system settings
# 4. Restart the application
```

</details>

<details>
<summary><b>📦 Import Errors</b></summary>

```bash
# 🔄 Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# 🆙 Upgrade pip
python -m pip install --upgrade pip

# 🧹 Clear cache
pip cache purge
```

</details>

<details>
<summary><b>🖐️ Poor Hand Detection</b></summary>

**Environment Setup:**

- ✅ Ensure good lighting conditions
- ✅ Use plain, contrasting background
- ✅ Keep hand within camera frame
- ✅ Check camera focus and cleanliness
- ✅ Maintain steady hand movements

</details>

<details>
<summary><b>🧠 Low Recognition Accuracy</b></summary>

**Optimization Tips:**

- ✅ Draw clearly in the yellow recognition area
- ✅ Use consistent stroke thickness
- ✅ Ensure good contrast between drawing and background
- ✅ Retrain models with custom data if needed
- ✅ Check lighting conditions

</details>

### ⚡ **Performance Optimization**

<div align="center">

![Performance](https://img.shields.io/badge/Target-30%2B%20FPS-green?style=flat-square)
![Memory](https://img.shields.io/badge/Memory-Optimized-blue?style=flat-square)
![GPU](https://img.shields.io/badge/GPU-Accelerated-purple?style=flat-square)

</div>

**For Better FPS:**

- � Close eother camera applications
- 📺 Use lower resolution if needed (`--width 480 --height 360`)
- �️\* Disable hand landmarks display (`L` key)
- 🎮 Use GPU acceleration if available
- 💾 Ensure sufficient RAM (4GB+ recommended)

## 📊 **System Requirements**

### 💻 **Minimum Requirements**

<div align="center">

![OS](https://img.shields.io/badge/OS-Windows%2010%20|%20macOS%2010.14%20|%20Ubuntu%2018.04-blue?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.8%2B-green?style=flat-square)
![RAM](https://img.shields.io/badge/RAM-4GB-orange?style=flat-square)
![CPU](https://img.shields.io/badge/CPU-Dual--core-red?style=flat-square)

</div>

| Component   | Minimum                               | Recommended          |
| ----------- | ------------------------------------- | -------------------- |
| **OS**      | Windows 10, macOS 10.14, Ubuntu 18.04 | Latest versions      |
| **Python**  | 3.8+                                  | 3.9+                 |
| **RAM**     | 4GB                                   | 8GB+                 |
| **CPU**     | Dual-core                             | Quad-core+           |
| **Camera**  | Any USB webcam                        | HD webcam (720p+)    |
| **GPU**     | Not required                          | NVIDIA GPU with CUDA |
| **Storage** | 2GB free space                        | 5GB+ for models      |

### 🚀 **Recommended Setup**

<div align="center">

![Recommended](https://img.shields.io/badge/Setup-Recommended-brightgreen?style=for-the-badge)

</div>

- **💻 OS**: Latest Windows 11, macOS Monterey, or Ubuntu 22.04
- **🐍 Python**: 3.9 or 3.10 for best compatibility
- **💾 RAM**: 8GB+ for smooth operation
- **🖥️ GPU**: NVIDIA GPU with CUDA support for ML acceleration
- **📷 Camera**: HD webcam (1080p) with good low-light performance
- **⚡ CPU**: Modern quad-core processor (Intel i5/AMD Ryzen 5+)

## 📊 **Project Statistics**

<div align="center">

![GitHub repo size](https://img.shields.io/github/repo-size/sunbyte16/virtual-board-ml?style=for-the-badge&logo=github&logoColor=white&color=blue)
![Lines of code](https://img.shields.io/tokei/lines/github/sunbyte16/virtual-board-ml?style=for-the-badge&logo=code&logoColor=white&color=green)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/sunbyte16/virtual-board-ml?style=for-the-badge&logo=git&logoColor=white&color=orange)
![GitHub last commit](https://img.shields.io/github/last-commit/sunbyte16/virtual-board-ml?style=for-the-badge&logo=github&logoColor=white&color=red)

</div>

## 🌟 **Show Your Support**

<div align="center">

If this project helped you, please consider giving it a ⭐️!

[![GitHub stars](https://img.shields.io/github/stars/sunbyte16/virtual-board-ml?style=for-the-badge&logo=github&logoColor=white&color=yellow)](https://github.com/sunbyte16/virtual-board-ml/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/sunbyte16/virtual-board-ml?style=for-the-badge&logo=github&logoColor=white&color=blue)](https://github.com/sunbyte16/virtual-board-ml/network)

</div>

## 📞 **Support & Contact**

<div align="center">

| Platform             | Link                                                                            | Description                    |
| -------------------- | ------------------------------------------------------------------------------- | ------------------------------ |
| 🐛 **Issues**        | [GitHub Issues](https://github.com/sunbyte16/virtual-board-ml/issues)           | Bug reports & feature requests |
| 💬 **Discussions**   | [GitHub Discussions](https://github.com/sunbyte16/virtual-board-ml/discussions) | Community discussions          |
| 📖 **Documentation** | [Wiki](https://github.com/sunbyte16/virtual-board-ml/wiki)                      | Detailed documentation         |
| 📧 **Email**         | [Contact](mailto:sunilsharma.dev@gmail.com)                                     | Direct contact                 |

</div>

## 👨‍💻 **About the Creator**

<div align="center">

### **Sunil Sharma**

_Full Stack Developer & AI Enthusiast_

[![GitHub](https://img.shields.io/badge/GitHub-sunbyte16-black?style=for-the-badge&logo=github&logoColor=white)](https://github.com/sunbyte16)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Sunil%20Kumar-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sunil-kumar-bb88bb31a/)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit%20Now-green?style=for-the-badge&logo=netlify&logoColor=white)](https://lively-dodol-cc397c.netlify.app)

---

### 🎯 **Skills & Technologies**

![Python](https://img.shields.io/badge/Python-Expert-blue?style=flat-square&logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Advanced-orange?style=flat-square&logo=tensorflow&logoColor=white)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Advanced-red?style=flat-square&logo=opencv&logoColor=white)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-Intermediate-purple?style=flat-square&logo=pytorch&logoColor=white)
![Web Development](https://img.shields.io/badge/Web%20Development-Full%20Stack-green?style=flat-square&logo=javascript&logoColor=white)

</div>

## 🤝 **Contributing**

<div align="center">

We welcome contributions from the community! 🎉

[![Contributors](https://img.shields.io/github/contributors/sunbyte16/virtual-board-ml?style=for-the-badge&logo=github&logoColor=white&color=brightgreen)](https://github.com/sunbyte16/virtual-board-ml/graphs/contributors)

### **How to Contribute:**

1. 🍴 **Fork** the repository
2. 🌿 **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. 💻 **Make** your changes
4. ✅ **Add** tests for new features
5. 🧪 **Run** the test suite (`make test`)
6. 📝 **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
7. 🚀 **Push** to the branch (`git push origin feature/AmazingFeature`)
8. 🔄 **Open** a Pull Request

</div>

## 🏆 **Achievements**

<div align="center">

![Achievement](https://img.shields.io/badge/🏆-Production%20Ready-gold?style=for-the-badge)
![Achievement](https://img.shields.io/badge/🎯-Real%20Time%20Processing-blue?style=for-the-badge)
![Achievement](https://img.shields.io/badge/🧠-ML%20Powered-purple?style=for-the-badge)
![Achievement](https://img.shields.io/badge/🖐️-Gesture%20Recognition-green?style=for-the-badge)

</div>

## 📄 **License**

<div align="center">

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge&logo=opensourceinitiative&logoColor=white)](LICENSE)

</div>

## 🙏 **Acknowledgments**

<div align="center">

Special thanks to the amazing open-source community and these incredible projects:

[![MediaPipe](https://img.shields.io/badge/MediaPipe-Google-blue?style=flat-square&logo=google)](https://mediapipe.dev)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-red?style=flat-square&logo=opencv)](https://opencv.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Machine%20Learning-orange?style=flat-square&logo=tensorflow)](https://tensorflow.org)
[![MNIST](https://img.shields.io/badge/MNIST-Dataset-green?style=flat-square)](http://yann.lecun.com/exdb/mnist/)

</div>

---

<div align="center">

### 💖 **Created By: [Sunil Sharma](https://github.com/sunbyte16)**

_Passionate about AI, Computer Vision, and creating innovative solutions that bridge the gap between humans and technology._

---

**⭐ Star this repository if you found it helpful!**

[![Made with Love](https://img.shields.io/badge/Made%20with-❤️-red?style=for-the-badge)](https://github.com/sunbyte16)
[![Open Source](https://img.shields.io/badge/Open%20Source-💚-brightgreen?style=for-the-badge)](https://opensource.org)
[![AI Powered](https://img.shields.io/badge/AI%20Powered-🤖-blue?style=for-the-badge)](https://tensorflow.org)

---

_© 2k25 Sunil Sharma. All rights reserved. Licensed under the MIT License._

</div>
#   V i r t u a l - B o a r d - b y - u s i n g - m a c h i n e - l e a r n i n g  
 #   V i r t u a l - B o a r d - b y - u s i n g - m a c h i n e - l e a r n i n g  
 #   V i r t u a l - B o a r d - b y - u s i n g - m a c h i n e - l e a r n i n g  
 