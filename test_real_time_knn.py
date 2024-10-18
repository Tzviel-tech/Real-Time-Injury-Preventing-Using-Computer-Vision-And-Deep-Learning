import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Initialize MediaPipe Drawing and Pose solutions
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to calculate the angle between three points using vector math
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b  # vector from b to a
    bc = c - b  # vector from b to c

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # Clip the values to avoid numerical issues

    angle = np.arccos(cosine_angle)  # Calculate angle in radians
    return np.degrees(angle)  # Convert to degrees

# Function to calculate shoulder-hip-knee angle (for left side)
def calculate_shoulder_hip_knee_angle(landmarks):
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z]
    return calculate_angle(left_shoulder, left_hip, left_knee)

# Function to calculate wrist-shoulder-hip angle (for left side)
def calculate_wrist_shoulder_hip_angle(landmarks):
    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z]
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z]
    return calculate_angle(left_wrist, left_shoulder, left_hip)

# Load your trained LSTM model
lstm_model = tf.keras.models.load_model(r'C:\Users\alexc\Final_Project\Final-Project\model_bicep_curl_with_angles.keras')

# Initialize MediaPipe Pose
pose = mp_pose.Pose()

# Open the webcam
cap = cv2.VideoCapture(0)

# Set window size for prediction (e.g., 30 frames)
window_size = 30
keypoint_window = []

# List of landmark indices used during training
include_landmark_indices = [
    mp_pose.PoseLandmark.LEFT_SHOULDER.value,
    mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
    mp_pose.PoseLandmark.LEFT_ELBOW.value,
    mp_pose.PoseLandmark.RIGHT_ELBOW.value,
    mp_pose.PoseLandmark.LEFT_WRIST.value,
    mp_pose.PoseLandmark.RIGHT_WRIST.value,
    mp_pose.PoseLandmark.LEFT_HIP.value,
    mp_pose.PoseLandmark.RIGHT_HIP.value,
    mp_pose.PoseLandmark.LEFT_KNEE.value,  # Added knee landmark
    mp_pose.PoseLandmark.RIGHT_KNEE.value
]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to RGB for MediaPipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the image and extract pose landmarks
    result = pose.process(image_rgb)

    if result.pose_landmarks:
        keypoints = []
        landmarks = result.pose_landmarks.landmark

        # Extract raw x, y, z coordinates (no normalization needed for LSTM)
        for idx in include_landmark_indices:
            x = landmarks[idx].x
            y = landmarks[idx].y
            z = landmarks[idx].z
            keypoints.extend([x, y, z])  # Extract x, y, z as is

        # Calculate shoulder-hip-knee and wrist-shoulder-hip angles from keypoints
        shoulder_hip_knee_angle = calculate_shoulder_hip_knee_angle(landmarks)
        wrist_shoulder_hip_angle = calculate_wrist_shoulder_hip_angle(landmarks)

        # Add the new angles to the keypoints
        keypoints.append(shoulder_hip_knee_angle)
        keypoints.append(wrist_shoulder_hip_angle)

        # Add keypoints to the window
        keypoint_window.append(keypoints)

        # Keep only the latest 'window_size' frames
        if len(keypoint_window) > window_size:
            keypoint_window.pop(0)

        # If we have enough frames for a prediction (e.g., 30 frames)
        if len(keypoint_window) == window_size:
            # Convert to a numpy array and reshape to match LSTM input shape
            input_data = np.array(keypoint_window).reshape(1, window_size, len(keypoints))

            # Make a prediction using the LSTM model
            prediction = lstm_model.predict(input_data)
            form = "Correct" if prediction > 0.5 else "Incorrect"

            # Display the prediction on the frame
            cv2.putText(frame, f'Form: {form}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Draw landmarks and connections (skeleton)
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))

    # Show the video with skeleton and prediction
    cv2.imshow('Bicep Curl Form Detection', frame)

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
