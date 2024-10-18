import json

# Load your JSON file (replace the file path with your actual bicep curl keypoints file)
file_path = r'C:\Users\alexc\Final_Project\Final-Project\keypoints_bicep_curl.json'
with open(file_path, 'r') as json_file:
    keypoints_data = json.load(json_file)

# Define the labeling logic based on the frame tags from your image
def label_form(frame):
    if 0 <= frame <= 3656:
        return 'Correct_Form'
    elif 3657 <= frame <= 3657:
        return 'Incorrect_Form'
    elif 3658 <= frame <= 4656:
        return 'Incorrect_Form'
    else:
        return 'Unknown'

# Apply the labeling function
for frame_str, keypoints in keypoints_data.items():
    frame_number = int(frame_str)
    form_label = label_form(frame_number)
    # Update the data with the form label
    keypoints_data[frame_str] = {
        'Form': form_label,
        'keypoints': keypoints
    }

# Save the updated JSON file with the form labels
output_path = r'C:\Users\alexc\Final_Project\Final-Project\keypoints_bicep_curl_labeled.json'
with open(output_path, 'w') as json_file:
    json.dump(keypoints_data, json_file, indent=4)

print(f"Labels added and saved to '{output_path}'.")
