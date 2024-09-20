import pandas as pd

# Load your CSV file
file_path = r'C:\Users\alexc\Final_Project\Final-Project\keypoints_squat.csv'  # Replace with your file path
keypoints_squat_df = pd.read_csv(file_path)

# Define the labeling logic
def label_form(frame):
    if 0 <= frame <= 1478:
        return 'Correct'
    elif 1479 <= frame <= 1620:
        return 'Incorrect'
    elif 1621 <= frame <= 2911:
        return 'Correct'
    elif 2912 <= frame <= 3493:
        return 'Incorrect'
    elif 3494 <= frame <= 3577:
        return 'Correct'
    elif 3578 <= frame <= 3620:
        return 'Correct'
    elif 3621 <= frame <= 5395:
        return 'Incorrect'
    else:
        return 'Unknown'

# Apply the labeling function to the 'frame' column
keypoints_squat_df['Form'] = keypoints_squat_df['frame'].apply(label_form)

# Save the new CSV
output_path = 'keypoints_squat_labeled.csv'
keypoints_squat_df.to_csv(output_path, index=False)
