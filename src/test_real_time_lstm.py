import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Initialize Mediapipe and model
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Dictionary to map prediction indices to labels
form_label_mapping = {
    0: 'Correct_Form',
    1: 'Incorrect Form Leaning Forward',
    2: 'Incorrect Form Leaning Backwards',
    3: 'Incorrect Form Loose Arms'
}

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

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

def calculate_elbow_shoulder_hip_angle(landmarks):
    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z]
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z]
    return calculate_angle(left_elbow, left_shoulder, left_hip)

# Load the multi-class model
lstm_model = tf.keras.models.load_model(r'C:\Users\alexc\Final_Project\Final-Project\model_bicep_curl_complete.keras')

pose = mp_pose.Pose()
cap = cv2.VideoCapture(0)

window_size = 30
keypoint_window = []

include_landmark_indices = [
    mp_pose.PoseLandmark.LEFT_SHOULDER.value,
    mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
    mp_pose.PoseLandmark.LEFT_ELBOW.value,
    mp_pose.PoseLandmark.RIGHT_ELBOW.value,
    mp_pose.PoseLandmark.LEFT_WRIST.value,
    mp_pose.PoseLandmark.RIGHT_WRIST.value,
    mp_pose.PoseLandmark.LEFT_HIP.value,
    mp_pose.PoseLandmark.RIGHT_HIP.value,
    mp_pose.PoseLandmark.LEFT_KNEE.value,
    mp_pose.PoseLandmark.RIGHT_KNEE.value
]

# Initialize default color for skeleton as green (Correct)
skeleton_color = (0, 255, 0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(image_rgb)

    if result.pose_landmarks:
        keypoints = []
        landmarks = result.pose_landmarks.landmark

        for idx in include_landmark_indices:
            x = landmarks[idx].x
            y = landmarks[idx].y
            z = landmarks[idx].z
            keypoints.extend([x, y, z])

        shoulder_hip_knee_angle = calculate_shoulder_hip_knee_angle(landmarks)
        elbow_shoulder_hip_angle = calculate_elbow_shoulder_hip_angle(landmarks)

        keypoints.append(shoulder_hip_knee_angle)
        keypoints.append(elbow_shoulder_hip_angle)

        keypoint_window.append(keypoints)

        if len(keypoint_window) > window_size:
            keypoint_window.pop(0)

        if len(keypoint_window) == window_size:
            input_data = np.array(keypoint_window).reshape(1, window_size, len(keypoints))
            prediction = lstm_model.predict(input_data)
            
            # Get the index of the highest probability in prediction
            predicted_index = np.argmax(prediction)
            form = form_label_mapping.get(predicted_index, "Unknown")

            # Check if the form is correct or incorrect
            if form == 'Correct_Form':
                text_color = (0, 255, 0)  # Green for correct
                skeleton_color = (0, 255, 0)
            else:
                text_color = (0, 0, 255)  # Red for incorrect
                skeleton_color = (0, 0, 255)

            # Display the form label on the video frame with smaller font size and red color for incorrect
            cv2.putText(frame, f'Form: {form}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)

        # Draw the skeleton in the specified color based on form correctness
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=skeleton_color, thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=skeleton_color, thickness=2, circle_radius=2))

    cv2.imshow('Bicep Curl Form Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

