import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os, csv

# Important landmarks and headers for bicep curl (right side, wrist control, and back posture)
IMPORTANT_LMS = [
    "RIGHT_SHOULDER",
    "RIGHT_ELBOW",
    "RIGHT_WRIST",
    "RIGHT_HIP",
    "RIGHT_KNEE",
    "NOSE"
]

HEADERS = ["label"]  # Label column

for lm in IMPORTANT_LMS:
    HEADERS += [f"{lm.lower()}_x", f"{lm.lower()}_y", f"{lm.lower()}_z", f"{lm.lower()}_v"]

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def rescale_frame(frame, percent=50):
    '''
    Rescale a frame to a certain percentage compared to its original frame
    '''
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

def init_csv(dataset_path: str):
    '''
    Create a blank csv file with just columns
    '''

    # Ignore if file already exists
    if os.path.exists(dataset_path):
        return

    # Write all the columns to an empty file
    with open(dataset_path, mode="w", newline="") as f:
        csv_writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(HEADERS)

def export_landmark_to_csv(dataset_path: str, results, action: str) -> None:
    '''
    Export Labeled Data from detected landmark to csv
    '''
    landmarks = results.pose_landmarks.landmark
    keypoints = []

    try:
        # Extract coordinates of important landmarks
        for lm in IMPORTANT_LMS:
            keypoint = landmarks[mp_pose.PoseLandmark[lm].value]
            keypoints.append([keypoint.x, keypoint.y, keypoint.z, keypoint.visibility])
        
        keypoints = list(np.array(keypoints).flatten())

        # Insert action as the label (first column)
        keypoints.insert(0, action)

        # Append new row to .csv file
        with open(dataset_path, mode="a", newline="") as f:
            csv_writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(keypoints)

    except Exception as e:
        print(e)
        pass

def describe_dataset(dataset_path: str):
    '''
    Describe dataset
    '''
    # Ensure the dataset exists before trying to read it
    if not os.path.exists(dataset_path):
        print(f"File {dataset_path} does not exist.")
        return None

    data = pd.read_csv(dataset_path)
    print(f"Headers: {list(data.columns.values)}")
    print(f'Number of rows: {data.shape[0]} \nNumber of columns: {data.shape[1]}\n')
    print(f"Labels: \n{data['label'].value_counts()}\n")
    print(f"Missing values: {data.isnull().values.any()}\n")
    
    duplicate = data[data.duplicated()]
    print(f"Duplicate Rows : {len(duplicate.sum(axis=1))}")

    return data

def remove_duplicate_rows(dataset_path: str):
    '''
    Remove duplicated data from the dataset then save it to another file
    '''
    df = pd.read_csv(dataset_path)
    df.drop_duplicates(keep="first", inplace=True)
    df.to_csv(f"cleaned_train.csv", sep=',', encoding='utf-8', index=False)

def concat_csv_files_with_same_headers(file_paths: list, saved_path: str):
    '''
    Concat different csv files
    '''
    all_df = []
    for path in file_paths:
        df = pd.read_csv(path, index_col=None, header=0)
        all_df.append(df)
    
    results = pd.concat(all_df, axis=0, ignore_index=True)
    results.to_csv(saved_path, sep=',', encoding='utf-8', index=False)

def extract_min_max(data):
    min_max = {}
    for label in data['label'].unique():
        min_max[label] = {}
        label_data = data[data['label'] == label]
        for col in data.columns:
            if col != 'label':
                min_max[label][col] = (label_data[col].min(), label_data[col].max())
    return min_max

def generate_synthetic_data(min_max, num_samples):
    synthetic_data = []
    for label in min_max.keys():
        for _ in range(num_samples):
            sample = [label]
            for col in HEADERS[1:]:  # Skip the 'label' column
                min_val, max_val = min_max[label][col]
                sample.append(np.random.uniform(min_val, max_val))
            synthetic_data.append(sample)
    return synthetic_data

# Main code to capture video from webcam and process frames
DATASET_PATH = "train.csv"

cap = cv2.VideoCapture(0)  # Capture video from the webcam
save_counts = 0

# Initialize the CSV file if it doesn't exist
init_csv(DATASET_PATH)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, image = cap.read()

        if not ret:
            break

        # Reduce size of a frame
        image = rescale_frame(image, 60)
        image = cv2.flip(image, 1)

        # Recolor image from BGR to RGB for mediapipe
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        if not results.pose_landmarks:
            print("Cannot detect pose - No human found")
            continue

        # Recolor image from BGR to RGB for mediapipe
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw landmarks and connections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(244, 117, 66), thickness=2, circle_radius=4), mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        # Display the saved count
        cv2.putText(image, f"Saved: {save_counts}", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow("CV2", image)

        # Pressed key for action
        k = cv2.waitKey(1) & 0xFF

        # Press C to save as correct form (good range of motion)
        if k == ord('c'):
            export_landmark_to_csv(DATASET_PATH, results, "C")
            save_counts += 1
        # Press B to save as back bad form
        elif k == ord("b"):
            export_landmark_to_csv(DATASET_PATH, results, "B")
            save_counts += 1
        # Press R to save as limited range of motion
        elif k == ord("r"):
            export_landmark_to_csv(DATASET_PATH, results, "R")
            save_counts += 1
        # Press W to save as wrist control
        elif k == ord("w"):
            export_landmark_to_csv(DATASET_PATH, results, "W")
            save_counts += 1
        # Press q to stop
        elif k == ord("q"):
            break
        else:
            continue

    cap.release()
    cv2.destroyAllWindows()

    # (Optional) Fix bugs cannot close windows in MacOS (https://stackoverflow.com/questions/6116564/destroywindow-does-not-close-window-on-mac-using-python-and-opencv)
    for i in range(1, 5):
        cv2.waitKey(1)

# Describe the dataset
df = describe_dataset(DATASET_PATH)

if df is not None:
    # Extract min and max values for each label
    min_max = extract_min_max(df)

    # Generate synthetic data
    synthetic_data = generate_synthetic_data(min_max, num_samples=10000)

    # Save synthetic data to CSV
    synthetic_df = pd.DataFrame(synthetic_data, columns=HEADERS)
    synthetic_df.to_csv("synthetic_data.csv", index=False)
