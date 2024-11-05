import json
import numpy as np

file_path = r'C:\Users\alexc\Final_Project\Final-Project\keypoints_shoulder_press_labeled.json'
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

    left_wrist = get_landmark_coordinates(keypoints, 'LEFT_WRIST')
    left_elbow = get_landmark_coordinates(keypoints, 'LEFT_ELBOW')
    
    right_wrist = get_landmark_coordinates(keypoints, 'RIGHT_WRIST')
    right_elbow = get_landmark_coordinates(keypoints, 'RIGHT_ELBOW')


    if right_elbow and left_elbow and right_wrist:
        right_elbow_left_elbow_right_wrist_angle = calculate_angle(right_elbow, left_elbow, right_wrist)
    else:
        right_elbow_left_elbow_right_wrist_angle = 'N/A'

    if left_elbow and right_elbow and left_wrist:
        left_elbow_right_elbow_left_wrist_angle = calculate_angle(left_elbow, right_elbow, left_wrist)
    else:
        left_elbow_right_elbow_left_wrist_angle = 'N/A'

    
    data['right_elbow_left_elbow_right_wrist_angle'] = right_elbow_left_elbow_right_wrist_angle
    data['left_elbow_right_elbow_left_wrist_angle'] = left_elbow_right_elbow_left_wrist_angle

output_path = r'C:\Users\alexc\Final_Project\Final-Project\keypoints_shoulder_press_with_angles.json'
with open(output_path, 'w') as json_file:
    json.dump(keypoints_data, json_file, indent=4)

print(f"Angles calculated and added to JSON. Saved to '{output_path}'.")
