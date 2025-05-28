# Real-Time Visual Assistance for Blind People

This project provides real-time visual assistance for visually impaired individuals by combining computer vision and audio feedback. It uses face recognition, object detection, and distance estimation to help users navigate their surroundings safely.

## Features

- **Face Recognition**: Identifies known faces and announces their presence
- **Object Detection**: Detects various objects in the environment using YOLO
- **Distance Estimation**: Calculates approximate distances to detected objects
- **Directional Audio**: Provides clock-based directional information
- **Emergency Alerts**: Warns about potential hazards and obstacles
- **Scene Description**: Describes the surrounding environment
- **Expression Recognition**: Detects facial expressions of people around

## Prerequisites

- Python 3.8 or higher
- Webcam
- Speakers/Headphones
- Required Python packages (install using `pip install -r requirements.txt`):
  - OpenCV (cv2)
  - Pygame
  - NumPy
  - pyttsx3
  - supervision
  - ultralytics (YOLO)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/temburuakhil/Real-Time-Visual-Assistance-for-Blind-People.git
cd Real-Time-Visual-Assistance-for-Blind-People
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Download the YOLO model:
```bash
# The model will be automatically downloaded on first run
```

## Usage

1. Run the main program:
```bash
python "face object distance direction buzzer.py"
```

2. The system will:
   - Initialize the camera
   - Start face recognition
   - Begin object detection
   - Provide audio feedback about:
     - Recognized faces and their locations
     - Objects in the environment
     - Potential hazards
     - Distance to objects
     - Direction of objects (using clock positions)

## Project Structure

- `face object distance direction buzzer.py`: Main program file
- `emergency.py`: Emergency detection and alert system
- `expression.py`: Facial expression recognition
- `scene.py`: Scene description module
- `haarcascade_frontalface_alt.xml`: Face detection model
- `Live QnA/`: Web interface for real-time assistance

## Features in Detail

### Face Recognition
- Uses Haar Cascade for face detection
- Implements KNN for face recognition
- Provides distance and direction information for recognized faces

### Object Detection
- Uses YOLO for real-time object detection
- Identifies various objects including vehicles, obstacles, and hazards
- Calculates approximate distances to detected objects

### Audio Feedback
- Text-to-speech announcements for detected objects and faces
- Directional information using clock positions
- Distance information in feet
- Emergency alerts for potential hazards

### Safety Features
- Vehicle detection and alerts
- Obstacle detection
- Emergency situation recognition
- Distance-based warnings

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLO for object detection
- OpenCV for computer vision capabilities
- Pygame for audio handling
- pyttsx3 for text-to-speech functionality 