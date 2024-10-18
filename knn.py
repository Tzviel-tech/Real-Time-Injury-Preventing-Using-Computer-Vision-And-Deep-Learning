import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib  # To save and load the model

# Load JSON data and prepare it for KNN (Flattening the sequence)
def prepare_data_for_knn(json_path, window_size):
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)
    
    X, y = [], []
    frames = sorted([int(frame) for frame in data.keys()])
    
    for i in range(len(frames) - window_size):
        sequence = []
        for j in range(window_size):
            frame_data = data[str(frames[i + j])]
            keypoints = frame_data['keypoints']
            
            # Extract features (keypoints + angles)
            features = []
            for kp in keypoints:
                features.extend([kp['x'], kp['y'], kp['z']])
            
            # Append the calculated angles (elbow, back)
            features.append(frame_data['elbow_angle'])
            features.append(frame_data['back_angle'])
            
            sequence.extend(features)  # Flatten the sequence
        
        X.append(sequence)  # Flattened feature vector for KNN
        y.append(1 if frame_data['Form'] == 'Correct_Form' else 0)
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y

# Prepare the data
json_path = r'C:\Users\alexc\Final_Project\Final-Project\keypoints_bicep_curl_with_angles.json'
X, y = prepare_data_for_knn(json_path, window_size=30)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize KNeighborsClassifier with 5 neighbors (K=5)
knn_model = KNeighborsClassifier(n_neighbors=5)

# Train the KNN model
knn_model.fit(X_train, y_train)

# Save the trained KNN model to a file
model_filename = r'C:\Users\alexc\Final_Project\Final-Project\knn_bicep_curl_model.joblib'
joblib.dump(knn_model, model_filename)
print(f"Model saved to {model_filename}")

# Load the model when needed
loaded_model = joblib.load(model_filename)

# Make predictions with the loaded model
y_pred = loaded_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"KNN Model Accuracy: {accuracy * 100:.2f}%")
