import cv2
import mediapipe as mp
import os
import json

# Initialize MediaPipe Pose model and drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Load video
cap = cv2.VideoCapture(r'C:\Users\alexc\Final_Project\Final-Project\Videos\squat2.mp4')

# Initialize Pose model
pose = mp_pose.Pose()

# Prepare to save frames and key points
frame_count = 0
keypoints_data = []

# Create directory for saved frames if it doesn't exist
os.makedirs('./frames/', exist_ok=True)

# List of landmark names from Mediapipe
landmark_names = [lm.name for lm in mp_pose.PoseLandmark]

# Indices of landmarks to exclude (face, hands, feet)
exclude_landmark_indices = list(range(0, 11)) + list(range(17, 24)) + list(range(29, 34))

# Loop through video frames
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to read frame. Stopping.")
        break

    # Convert frame to RGB (MediaPipe expects RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Get pose landmarks
    results = pose.process(rgb_frame)

    # Check if landmarks are detected
    if results.pose_landmarks:
        print(f"Pose detected on frame {frame_count}")
        
        # Save frame every 500 frames
        if frame_count % 500 == 0:
            # Draw pose landmarks on the frame
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            # Save frame with skeleton overlay
            cv2.imwrite(f'./frames/frame_{frame_count}.png', frame)

        # Extract key points and save them
        frame_keypoints = []
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            # Skip excluded landmarks
            if idx in exclude_landmark_indices:
                continue
            landmark_name = landmark_names[idx] if idx < len(landmark_names) else f'Landmark_{idx}'
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

    # Increment frame count
    frame_count += 1

# Release the video capture object
cap.release()

# Save keypoints data to JSON if keypoints were detected
if keypoints_data:
    # Organize keypoints by frame
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
    # Save to JSON file
    with open('keypoints_squat.json', 'w') as json_file:
        json.dump(keypoints_by_frame, json_file, indent=4)
    print(f"Processed {frame_count} frames and saved keypoints to 'keypoints_squat.json'.")
else:
    print("No keypoints were detected. JSON file will not be created.")




