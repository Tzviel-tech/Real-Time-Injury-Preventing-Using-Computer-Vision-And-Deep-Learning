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

# Main code to capture video from webcam and process frames
DATASET_PATH = "train.csv"

cap = cv2.VideoCapture(0)  # Capture video from the webcam
save_counts_correct = 0
save_counts_incorrect = 0

# Initialize the CSV file if it doesn't exist
init_csv(DATASET_PATH)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, image = cap.read()

        if not ret:
            break

        # Reduce size of a frame
        image = rescale_frame(image, 130)
        image = cv2.flip(image, 1)

        # Recolor image from BGR to RGB for mediapipe
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        if not results.pose_landmarks:
            print("Cannot detect pose - No human found")
            continue

        # Recolor image from RGB back to BGR for OpenCV
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw landmarks and connections
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(244, 117, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

        # Display the saved counts
        start_x = 50
        start_y = 50
        font = cv2.FONT_HERSHEY_COMPLEX
        font_scale = 1.2
        font_color = (0, 0, 0)
        thickness = 2
        line_type = cv2.LINE_AA

        # First line of text
        text1 = f"Saved counts for correct form: {save_counts_correct}"
        (text_width1, text_height1), baseline1 = cv2.getTextSize(text1, font, font_scale, thickness)
        cv2.putText(image, text1, (start_x, start_y), font, font_scale, font_color, thickness, line_type)

        # Second line of text
        text2 = f"Saved counts for incorrect form: {save_counts_incorrect}"
        (text_width2, text_height2), baseline2 = cv2.getTextSize(text2, font, font_scale, thickness)
        line_spacing = 10  # Adjust spacing between lines if needed
        cv2.putText(image, text2, (start_x, start_y + text_height1 + line_spacing), font, font_scale, font_color, thickness, line_type)

        cv2.imshow("CV2", image)

        # Pressed key for action
        k = cv2.waitKey(1) & 0xFF

        # Press c to save as correct form 
        if k == ord('c'):
            export_landmark_to_csv(DATASET_PATH, results, "C")
            save_counts_correct += 1
        # Press l to save as incorrect form
        elif k == ord("l"):
            export_landmark_to_csv(DATASET_PATH, results, "L")
            save_counts_incorrect += 1
        # Press q to stop
        elif k == ord("q"):
            break
        else:
            continue
    cap.release()
    cv2.destroyAllWindows()

    for i in range(1, 5):
        cv2.waitKey(1)

# Describe the dataset
df = describe_dataset(DATASET_PATH)
