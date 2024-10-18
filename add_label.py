import json


file_path = r'C:\Users\alexc\Final_Project\Final-Project\keypoints_bicep_curl.json'
with open(file_path, 'r') as json_file:
    keypoints_data = json.load(json_file)

def label_form(frame):
    if 0 <= frame <= 2545:
        return 'Correct_Form'
    elif 2546 <= frame <= 5118:
        return 'Incorrect_Form'
    else:
        return 'Unknown'

for frame_str, keypoints in keypoints_data.items():
    frame_number = int(frame_str)
    form_label = label_form(frame_number)
    keypoints_data[frame_str] = {
        'Form': form_label,
        'keypoints': keypoints
    }

output_path = r'C:\Users\alexc\Final_Project\Final-Project\keypoints_bicep_curl_labeled.json'
with open(output_path, 'w') as json_file:
    json.dump(keypoints_data, json_file, indent=4)

print(f"Labels added and saved to '{output_path}'.")
