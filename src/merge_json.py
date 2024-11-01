import json

# Load the first JSON file
with open(r'C:\Users\alexc\Final_Project\Final-Project\keypoints_bicep_curl_labeled_with_angles_2.json', 'r') as file1:
    data1 = json.load(file1)

# Load the second JSON file
with open(r'C:\Users\alexc\Final_Project\Final-Project\keypoints_bicep_curl_labeled_with_angles.json', 'r') as file2:
    data2 = json.load(file2)

# Offset frames in the second JSON file by 27019
data2_adjusted = {str(int(frame) + 27019): value for frame, value in data2.items()}

# Combine the two dictionaries
merged_data = {**data1, **data2_adjusted}

# Save the merged data to a new JSON file
with open(r'C:\Users\alexc\Final_Project\Final-Project\bicep_data_combined.json', 'w') as output_file:
    json.dump(merged_data, output_file, indent=4)

print("Files have been merged and saved to 'path_to_output_file.json'")
