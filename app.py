import os
import cv2
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from ultralytics import YOLO
from collections import deque
from flask import Flask, render_template, Response, request, redirect, url_for
from werkzeug.utils import secure_filename
from datetime import datetime
import signal
import atexit

# Custom Keras Layer
class CustomLayer(layers.Layer):
    def __init__(self, units=32, **kwargs):
        super(CustomLayer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer="random_normal", trainable=True)
        self.b = self.add_weight(shape=(self.units,), initializer="zeros", trainable=True)
        super(CustomLayer, self).build(input_shape)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):
        config = super(CustomLayer, self).get_config()
        config.update({"units": self.units})
        return config

# Load AI Models
custom_model = load_model('models/custom_model.h5', custom_objects={'CustomLayer': CustomLayer}, compile=False)
gru_model = load_model('models/gru_model.h5', compile=False)
lstm_model = load_model('models/lstm_model.h5', compile=False)
cnn_model = load_model('models/cnn_model.h5', compile=False)
mlp_model = load_model('models/mlp_model.h5', compile=False)

# YOLO Model for Object Detection
yolo_model = YOLO('yolov8m.pt')

# Initialize Flask App
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'videos'
app.config['VIDEO_SOURCE'] = os.path.join(app.config['UPLOAD_FOLDER'])

# Store previous counts
count_history = deque(maxlen=32)
predictions = {'custom': [], 'gru': [], 'lstm': [], 'cnn': [], 'mlp': []}

# Global variable for video capture
video_capture = None

def cleanup():
    global video_capture
    if video_capture is not None:
        video_capture.release()

def reset_video_capture():
    global video_capture
    if video_capture is not None:
        video_capture.release()
        video_capture = None

def signal_handler(signum, frame):
    cleanup()
    exit(0)

def get_video_capture():
    global video_capture
    try:
        if video_capture is None or not video_capture.isOpened():
            video_capture = cv2.VideoCapture(app.config['VIDEO_SOURCE'])
            if not video_capture.isOpened():
                raise Exception("Failed to open video source")
        return video_capture
    except Exception as e:
        print(f"Video capture error: {e}")
        return None

def preprocess_input(data, model_type):
    data = np.array(data, dtype=np.float32)
    return data.reshape(1, data.shape[0], 1) if model_type in ['gru', 'lstm', 'cnn'] else data.reshape(1, -1)

def generate_frames():
    global video_capture
    try:
        cap = get_video_capture()
        if not cap or not cap.isOpened():
            print("Error: Could not open video source")
            return
        
        while True:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            try:
                results = yolo_model.track(frame, persist=True, classes=[2, 3, 5, 7])
            except Exception as e:
                print("YOLO detection error:", e)
                results = None

            # Count unique vehicles
            current_count = len(set(results[0].boxes.id.int().tolist())) if results and hasattr(results[0], 'boxes') else 0
            count_history.append(current_count)

            # Predictions
            if len(count_history) == 32:
                try:
                    inputs = {model: preprocess_input(count_history, model) for model in predictions.keys()}
                    for model_name in predictions.keys():
                        predictions[model_name].append(float(eval(f"{model_name}_model").predict(inputs[model_name])[0][0]))
                except Exception as e:
                    print("Prediction error:", e)

            # Annotate Frame
            annotated_frame = results[0].plot() if results and hasattr(results[0], 'plot') else frame.copy()
            y_pos = 30
            cv2.putText(annotated_frame, f"Current Count: {current_count}", (20, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if predictions['custom']:
                colors = [(255, 0, 0), (0, 255, 0), (0, 255, 255), (255, 165, 0), (128, 0, 128)]
                for model_name, color in zip(predictions.keys(), colors):
                    y_pos += 30
                    cv2.putText(annotated_frame, f"{model_name.capitalize()} Pred: {predictions[model_name][-1]:.2f}",
                                (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            ret2, buffer = cv2.imencode('.jpg', annotated_frame)
            if not ret2:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    except Exception as e:
        print(f"Error in generate_frames: {e}")
        cleanup()
        return

@app.route('/main')
def main():
    return render_template('main.html', year=datetime.now().year)

@app.route('/')
def root():
    return redirect(url_for('main'))

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file uploaded", 400
    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400
    
    # Reset existing video capture
    reset_video_capture()
    
    try:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        app.config['VIDEO_SOURCE'] = file_path
        
        # Clear previous data
        count_history.clear()
        for key in predictions:
            predictions[key].clear()
            
        return redirect(url_for('output'))
    except Exception as e:
        print(f"Upload error: {e}")
        return "Error processing video", 500

@app.route('/output')
def output():
    return render_template('output.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Register cleanup handlers
atexit.register(cleanup)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == '__main__':
    try:
        app.run(debug=True, use_reloader=False)
    except Exception as e:
        print(f"Server error: {e}")
        cleanup()
