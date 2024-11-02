import cv2
import mediapipe as mp
import os
import json

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(r'C:\Users\alexc\Final_Project\Final-Project\data\Videos\plank.mp4')

pose = mp_pose.Pose()

frame_count = 0
keypoints_data = []

os.makedirs('./frames_plank/', exist_ok=True)

landmark_names = [lm.name for lm in mp_pose.PoseLandmark]

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

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to read frame. Stopping.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        print(f"Pose detected on frame {frame_count}")
        
        if frame_count % 500 == 0:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.imwrite(f'./frames_plank/frame_{frame_count}.png', frame)

        frame_keypoints = []
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            if idx not in include_landmark_indices:
                continue
            landmark_name = landmark_names[idx]
            frame_keypoints.append({
                'frame': frame_count,
                'landmark': landmark_name,
                'x': lm.x,
                'y': lm.y,
                'z': lm.z,
            })
        
        keypoints_data.extend(frame_keypoints)
    else:
        print(f"No pose detected on frame {frame_count}")

    frame_count += 1

cap.release()

if keypoints_data:
    keypoints_by_frame = {}
    for kp in keypoints_data:
        frame = kp['frame']
        if frame not in keypoints_by_frame:
            keypoints_by_frame[frame] = []
        keypoints_by_frame[frame].append({
            'landmark': kp['landmark'],
            'x': kp['x'],
            'y': kp['y'],
            'z': kp['z'],
        })
    with open('keypoints_plank.json', 'w') as json_file:
        json.dump(keypoints_by_frame, json_file, indent=4)
    print(f"Processed {frame_count} frames and saved keypoints to 'keypoints_plank.json'.")
else:
    print("No keypoints were detected. JSON file will not be created.")
