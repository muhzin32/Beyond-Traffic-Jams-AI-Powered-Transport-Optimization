# Beyond Traffic Jams: AI-Powered Transport Optimization

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Flask](https://img.shields.io/badge/Flask-2.x-green.svg)

## Overview
This project uses multiple AI models including YOLO object detection and various neural networks (Custom, GRU, LSTM, CNN, MLP) to analyze traffic patterns and predict vehicle counts in real-time.

## Features
- Real-time vehicle detection using YOLOv8
- Multiple AI models for traffic prediction
- Web interface for video upload and analysis
- Live traffic count visualization
- Comparative prediction analysis across different models

## Setup
1. Install required dependencies:
```bash
pip install flask tensorflow opencv-python ultralytics
```

2. Ensure you have the following model files in the `models` directory:
- custom_model.h5
- gru_model.h5
- lstm_model.h5
- cnn_model.h5
- mlp_model.h5
- yolov8m.pt

3. Run the application:
```bash
python app.py
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/beyond-traffic-jams.git
cd beyond-traffic-jams
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Access the web interface at `http://localhost:5000`
2. Upload a traffic video through the interface
3. View real-time analysis including:
   - Current vehicle count
   - Predictions from different AI models
   - Annotated video feed with detection boxes

## Project Structure
- `/models` - Contains trained AI models
- `/videos` - Storage for uploaded video files
- `/templates` - HTML templates for web interface
- `app.py` - Main application file

## Technologies Used
- Flask
- TensorFlow
- OpenCV
- YOLOv8
- Python 3.x

## Contributing
1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make changes and commit (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- YOLOv8 by Ultralytics
- TensorFlow team
- Flask framework contributors
