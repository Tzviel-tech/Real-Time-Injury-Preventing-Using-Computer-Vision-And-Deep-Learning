import numpy as np
import json
import tensorflow as tf
from tensorflow import keras
from keras import layers



# Load JSON data and prepare it for ConvLSTM
def prepare_data(json_path, window_size):
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
            
            sequence.append(features)
        
        X.append(sequence)
        y.append(1 if frame_data['Form'] == 'Correct_Form' else 0)
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y

# Prepare the data (replace with your JSON file path)
json_path = r'C:\Users\alexc\Final_Project\Final-Project\keypoints_bicep_curl_with_angles.json'
X, y = prepare_data(json_path, window_size=30)

# Reshape the data to match ConvLSTM input (batch, time_steps, channels, height, width)
X = X.reshape((X.shape[0], X.shape[1], 1, X.shape[2], 1))

# Split into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    layers.LSTM(64, return_sequences=True, input_shape=(30, 26)),  # 30 time steps, 26 features
    layers.BatchNormalization(),
    layers.LSTM(64, return_sequences=False),
    layers.BatchNormalization(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
