# Real-Time Visual Assistance for Blind People

AI-powered visual assistance system using YOLOv8, OpenCV, and Google Gemini 2.5 Flash for real-time environmental awareness.

## Features

- **YOLOv8 Object Detection**: Real-time detection with <1 second latency
- **Face Recognition**: Haar Cascade + custom KNN classifier with 90%+ accuracy
- **Spatial Analysis**: Distance estimation and clock-based directional positioning
- **Expression Recognition**: Emotion detection for social cues
- **Live Q&A**: Google Gemini 2.5 Flash conversational AI integration
- **Multi-Modal Feedback**: Text-to-speech, haptic alerts, emergency calling (Twilio)
- **Web Interface**: React/TypeScript live interaction dashboard

## Installation

```bash
# Clone repository
git clone https://github.com/temburuakhil/Real-Time-Visual-Assistance-for-Blind-People.git
cd Real-Time-Visual-Assistance-for-Blind-People

# Install dependencies
pip install -r requirements.txt
```

## Usage

**Python Backend:**
```bash
python "face object distance direction buzzer.py"
```

**Web Interface:**
```bash
cd "Live QnA"
npm install
npm start
```

## Project Structure

```
├── face object distance direction buzzer.py  # Main detection system
├── emergency.py                              # Emergency detection module
├── expression.py                             # Facial expression recognition
├── scene.py                                  # Scene description module
├── yolov8n.pt                               # YOLOv8 model
├── haarcascade_frontalface_alt.xml          # Face detection cascade
├── requirements.txt                          # Python dependencies
├── data/                                     # Face recognition training data
└── Live QnA/                                # React web interface
    ├── src/
    │   ├── components/                       # UI components
    │   ├── contexts/                         # API context management
    │   ├── hooks/                           # Custom React hooks
    │   └── lib/                             # Gemini API client
    └── public/
```

## Technical Stack

**Backend:** Python 3.8+, OpenCV, NumPy, YOLOv8, pyttsx3  
**Frontend:** React, TypeScript, WebSocket  
**AI/ML:** YOLOv8, Haar Cascade, KNN, Google Gemini 2.5 Flash  
**APIs:** Twilio (emergency calling), Google Gemini API

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLO for object detection
- OpenCV for computer vision capabilities
- Pygame for audio handling
- pyttsx3 for text-to-speech functionality 