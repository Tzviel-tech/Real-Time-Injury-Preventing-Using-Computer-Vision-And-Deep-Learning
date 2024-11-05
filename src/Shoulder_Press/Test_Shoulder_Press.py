import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

form_label_mapping = {
    0: 'Correct_Form',
    1: 'Unbalanced_Arms'
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

def calculate_right_elbow_left_elbow_right_wrist_angle(landmarks):
    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z]
    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z]
    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z]
    return calculate_angle(right_elbow, left_elbow, right_wrist)

def calculate_left_elbow_right_elbow_left_wrist_angle(landmarks):
    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z]
    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z]
    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z]
    return calculate_angle(left_elbow, right_elbow, left_wrist)

lstm_model = tf.keras.models.load_model(r'C:\Users\alexc\Final_Project\Final-Project\model_shoulder_press.keras')

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

        
        right_elbow_left_elbow_right_wrist_angle = calculate_right_elbow_left_elbow_right_wrist_angle(landmarks)
        left_elbow_right_elbow_left_wrist_angle = calculate_left_elbow_right_elbow_left_wrist_angle(landmarks)

        
        keypoints.append(right_elbow_left_elbow_right_wrist_angle)
        keypoints.append(left_elbow_right_elbow_left_wrist_angle)

        keypoint_window.append(keypoints)

        if len(keypoint_window) > window_size:
            keypoint_window.pop(0)

        if len(keypoint_window) == window_size:
            input_data = np.array(keypoint_window).reshape(1, window_size, len(keypoints))
            prediction = lstm_model.predict(input_data)
        
            predicted_index = np.argmax(prediction)
            form = form_label_mapping.get(predicted_index, "Unknown")

            if form == 'Correct_Form':
                text_color = (0, 255, 0)  
                skeleton_color = (0, 255, 0) 
            else:
                text_color = (0, 0, 255)  
                skeleton_color = (0, 0, 255) 

            cv2.putText(frame, f'Form: {form}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)

        mp_drawing.draw_landmarks(
            frame,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=skeleton_color, thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=skeleton_color, thickness=2, circle_radius=2)
        )

    cv2.imshow('Shoulder Press Form Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
