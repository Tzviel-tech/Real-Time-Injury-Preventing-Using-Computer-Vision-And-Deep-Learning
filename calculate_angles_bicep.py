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
    
    ba = a - b  # vector ba: first to middle point (e.g. hip to shoulder)
    bc = c - b  # vector bc: middle to last point (e.g. shoulder to neck)
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
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
    
    # Get coordinates of relevant landmarks for back and elbow angles
    shoulder = get_landmark_coordinates(keypoints, 'LEFT_SHOULDER')  # Use 'RIGHT_SHOULDER' for the right arm
    hip = get_landmark_coordinates(keypoints, 'LEFT_HIP')
    elbow = get_landmark_coordinates(keypoints, 'LEFT_ELBOW')
    wrist = get_landmark_coordinates(keypoints, 'LEFT_WRIST')
    
    # Calculate the back angle (between hip-shoulder-neck)
    if shoulder and hip:
        neck = get_landmark_coordinates(keypoints, 'RIGHT_SHOULDER')  # Use the midpoint between shoulders as neck
        if neck:
            back_angle = calculate_angle(hip, shoulder, neck)
        else:
            back_angle = 'N/A'  # Mark missing angles
    else:
        back_angle = 'N/A'
    
    # Calculate elbow angle (between shoulder-elbow-wrist)
    if shoulder and elbow and wrist:
        elbow_angle = calculate_angle(shoulder, elbow, wrist)
    else:
        elbow_angle = 'N/A'

    # Add angles to the keypoints data
    data['back_angle'] = back_angle
    data['elbow_angle'] = elbow_angle

# Save the updated JSON with angles
output_path = r'C:\Users\alexc\Final_Project\Final-Project\keypoints_bicep_curl_with_angles.json'
with open(output_path, 'w') as json_file:
    json.dump(keypoints_data, json_file, indent=4)

print(f"Angles calculated and added to JSON. Saved to '{output_path}'.")
