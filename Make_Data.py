import cv2
import mediapipe as mp
import pandas as pd

# Initialize MediaPipe Pose model and drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Load video (handle the path correctly using a raw string)
cap = cv2.VideoCapture(r'C:\Users\alexc\Final_Project\Final-Project\Videos\squat2.mp4')

# Initialize Pose model
pose = mp_pose.Pose()

# Prepare to save frames and key points
frame_count = 0
keypoints_data = []

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
        # Draw pose landmarks on the frame
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Extract key points and save them
        frame_keypoints = []
        for id, lm in enumerate(results.pose_landmarks.landmark):
            frame_keypoints.append({
                'frame': frame_count,
                'id': id,
                'x': lm.x,
                'y': lm.y,
                'z': lm.z,
                'visibility': lm.visibility
            })
        
        keypoints_data.extend(frame_keypoints)
    else:
        print(f"No pose detected on frame {frame_count}")

    # Save frame with skeleton overlay
    cv2.imwrite(f'./frames/frame_{frame_count}.png', frame)

    # Increment frame count
    frame_count += 1

# Release the video capture object
cap.release()

# Save keypoints data to CSV if keypoints were detected
if keypoints_data:
    keypoints_df = pd.DataFrame(keypoints_data)
    keypoints_df.to_csv('keypoints_squat.csv', index=False)
    print(f"Processed {frame_count} frames and saved keypoints to 'keypoints.csv'.")
else:
    print("No keypoints were detected. CSV will not be created.")
