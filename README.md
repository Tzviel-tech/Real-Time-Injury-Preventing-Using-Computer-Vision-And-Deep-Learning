# Real-Time Gym Exercise Form Predictor

## Overview

This project uses Mediapipe and LSTM to predict in real-time whether a gym exercise is being performed correctly. It aims to help users improve their workout form and reduce the risk of injury during exercise.

## Features

- Real-time video processing using OpenCV
- Deep learning model for pose estimation and form analysis
- Support for multiple common gym exercises
- Visual feedback on exercise form

## Runing Example - Bicep - Curls Evaluator


https://github.com/user-attachments/assets/4f0e5bd9-2e84-4676-a5f3-c20b1cc62bf2

## Model Training Tracking Example - Bicep Curl


![Model Training Tracking Example](https://github.com/user-attachments/assets/1aa3383b-8350-454b-add2-0a08fcf3ab25)

## Prerequisites

- Python 3.7+
- OpenCV
- mediapipe
- TensorFlow 
- Webcam or video input device

## Installation

1. Clone the repository:
git clone https://github.com/Tzviel-tech/Real-Time-Injury-Preventing-Using-Computer-Vision-And-Deep-Learning.git cd gym-exercise-predictor


2. Install required packages:
pip install -r req.txt


## Usage

1. Run the main script (make sure to adjust the path to the model according to your directory here -
 ```python
   lstm_model = tf.keras.models.load_model(r'../models/model_bicep_curl_complete.keras'))
 ```

3. Position yourself in front of the camera, ensuring your full body is visible.

4. Start performing one of the supported exercises.

5. The application will provide real-time feedback on your form.

## Supported Exercises

- Shoulder Press
- Bicep Curls
- Plank

## How It Works

1. The application captures video input from your camera using OpenCV.
2. Each frame is processed through a pose estimation model to identify key body points.
3. These points are analyzed by our custom deep learning model to assess exercise form.
4. Real-time feedback is displayed on the video feed, indicating correct form or areas for improvement.

## Contributing

Contributions to improve the project are welcome. Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## Acknowledgements

- [OpenCV](https://opencv.org/)
- [TensorFlow](https://www.tensorflow.org/) or [PyTorch](https://pytorch.org/)
- [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) or similar pose estimation library
