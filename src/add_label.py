import json

file_path = r'C:\Users\alexc\Final_Project\Final-Project\keypoints_plank.json'
output_path = r'C:\Users\alexc\Final_Project\Final-Project\keypoints_plank_labeled.json'

with open(file_path, 'r') as json_file:
    keypoints_data = json.load(json_file)

def label_form(frame):
    if 0 <= frame <= 2419:
        return 'Correct_Form'
    elif 2420 <= frame <= 4736:
        return 'High_Back'
    elif 4737 <= frame <= 7137:
        return 'Low_Back'
    else:
        return 'Unknown'

for frame_str, keypoints in keypoints_data.items():
    frame_number = int(frame_str)
    form_label = label_form(frame_number)
    keypoints_data[frame_str] = {
        'Form': form_label,
        'keypoints': keypoints
    }

with open(output_path, 'w') as json_file:
    json.dump(keypoints_data, json_file, indent=4)

print(f"Labels added and saved to '{output_path}'.")
