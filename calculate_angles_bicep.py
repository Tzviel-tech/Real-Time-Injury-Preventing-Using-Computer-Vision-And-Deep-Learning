import json
import numpy as np

# Load keypoint data from JSON (replace this with the actual bicep curl keypoints JSON file)
file_path = r'C:\Users\alexc\Final_Project\Final-Project\Labeled_Data\keypoints_bicep_curl_labeled.json'
with open(file_path, 'r') as json_file:
    keypoints_data = json.load(json_file)

# Function to calculate the angle between three points using vector math
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b  # vector ba: first to middle point
    bc = c - b  # vector bc: middle to last point
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # Clipping to avoid numerical issues
    
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

# Function to get specific landmark coordinates from the keypoints
def get_landmark_coordinates(keypoints, landmark_name):
    for keypoint in keypoints:
        if keypoint['landmark'] == landmark_name:
            return (keypoint['x'], keypoint['y'], keypoint['z'])
    return None

# Iterate over each frame and calculate angles, then update the JSON with the new data
for frame, data in keypoints_data.items():
    keypoints = data['keypoints']
    
    # Get coordinates of relevant landmarks for left side angles
    shoulder = get_landmark_coordinates(keypoints, 'LEFT_SHOULDER')
    hip = get_landmark_coordinates(keypoints, 'LEFT_HIP')
    elbow = get_landmark_coordinates(keypoints, 'LEFT_ELBOW')
    wrist = get_landmark_coordinates(keypoints, 'LEFT_WRIST')
    knee = get_landmark_coordinates(keypoints, 'LEFT_KNEE')

    # Calculate shoulder-hip-knee angle (left leg)
    if shoulder and hip and knee:
        shoulder_hip_knee_angle = calculate_angle(shoulder, hip, knee)
    else:
        shoulder_hip_knee_angle = 'N/A'

    # Calculate wrist-shoulder-hip angle (upper body)
    if wrist and shoulder and hip:
        wrist_shoulder_hip_angle = calculate_angle(wrist, shoulder, hip)
    else:
        wrist_shoulder_hip_angle = 'N/A'

    # Add angles to the keypoints data
    data['shoulder_hip_knee_angle'] = shoulder_hip_knee_angle
    data['wrist_shoulder_hip_angle'] = wrist_shoulder_hip_angle

# Save the updated JSON with angles
output_path = r'C:\Users\alexc\Final_Project\Final-Project\keypoints_bicep_curl_with_knee_and_hip_angles.json'
with open(output_path, 'w') as json_file:
    json.dump(keypoints_data, json_file, indent=4)

print(f"Angles calculated and added to JSON. Saved to '{output_path}'.")
