import json

file_path = r'C:\Users\alexc\Final_Project\Final-Project\keypoints_bicep_curl_new.json'
output_path = r'C:\Users\alexc\Final_Project\Final-Project\keypoints_bicep_curl_2_labeled.json'

with open(file_path, 'r') as json_file:
    keypoints_data = json.load(json_file)

# Updated label_form function with new frame ranges
def label_form(frame):
    if 0 <= frame <= 10749:
        return 'Correct_Form'
    elif 10750 <= frame <= 16148:
        return 'Incorrect_Form_Leaning_Forward'
    elif 16149 <= frame <= 16161:
        return 'Correct_Form'
    elif 16162 <= frame <= 21499:
        return 'Incorrect_Form_Loose_Arms'
    elif 21500 <= frame <= 21509:
        return 'Correct_Form'
    elif 21510 <= frame <= 27018:
        return 'Incorrect_Form_Leaning_Backwards'
    else:
        return 'Unknown'

# Process each frame in the JSON file
for frame_str, keypoints in keypoints_data.items():
    frame_number = int(frame_str)
    form_label = label_form(frame_number)
    keypoints_data[frame_str] = {
        'Form': form_label,
        'keypoints': keypoints
    }

# Save updated JSON with labels
with open(output_path, 'w') as json_file:
    json.dump(keypoints_data, json_file, indent=4)

print(f"Labels added and saved to '{output_path}'.")
