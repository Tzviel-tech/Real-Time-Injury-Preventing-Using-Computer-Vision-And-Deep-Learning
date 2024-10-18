import json
import numpy as np

file_path = r'C:\Users\alexc\Final_Project\Final-Project\Labeled_Data\keypoints_bicep_curl_labeled.json'
with open(file_path, 'r') as json_file:
    keypoints_data = json.load(json_file)

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

def get_landmark_coordinates(keypoints, landmark_name):
    for keypoint in keypoints:
        if keypoint['landmark'] == landmark_name:
            return (keypoint['x'], keypoint['y'], keypoint['z'])
    return None

for frame, data in keypoints_data.items():
    keypoints = data['keypoints']

    shoulder = get_landmark_coordinates(keypoints, 'LEFT_SHOULDER')
    hip = get_landmark_coordinates(keypoints, 'LEFT_HIP')
    elbow = get_landmark_coordinates(keypoints, 'LEFT_ELBOW')
    wrist = get_landmark_coordinates(keypoints, 'LEFT_WRIST')
    knee = get_landmark_coordinates(keypoints, 'LEFT_KNEE')

    if shoulder and hip and knee:
        shoulder_hip_knee_angle = calculate_angle(shoulder, hip, knee)
    else:
        shoulder_hip_knee_angle = 'N/A'

    if wrist and shoulder and hip:
        wrist_shoulder_hip_angle = calculate_angle(wrist, shoulder, hip)
    else:
        wrist_shoulder_hip_angle = 'N/A'

    data['shoulder_hip_knee_angle'] = shoulder_hip_knee_angle
    data['wrist_shoulder_hip_angle'] = wrist_shoulder_hip_angle

output_path = r'C:\Users\alexc\Final_Project\Final-Project\keypoints_bicep_curl_with_knee_and_hip_angles.json'
with open(output_path, 'w') as json_file:
    json.dump(keypoints_data, json_file, indent=4)

print(f"Angles calculated and added to JSON. Saved to '{output_path}'.")
