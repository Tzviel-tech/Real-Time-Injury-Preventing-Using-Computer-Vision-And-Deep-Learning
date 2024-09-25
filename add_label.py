import json

# Load your JSON file
file_path = r'C:\Users\alexc\Final_Project\Final-Project\keypoints_squat.json'
with open(file_path, 'r') as json_file:
    keypoints_data = json.load(json_file)

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

# Apply the labeling function
for frame_str, keypoints in keypoints_data.items():
    frame_number = int(frame_str)
    form_label = label_form(frame_number)
    # Update the data
    keypoints_data[frame_str] = {
        'Form': form_label,
        'keypoints': keypoints
    }

# Save the updated JSON file
output_path = 'keypoints_squat_labeled.json'
with open(output_path, 'w') as json_file:
    json.dump(keypoints_data, json_file, indent=4)

print(f"Labels added and saved to '{output_path}'.")

