import numpy as np
import json
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split

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
            
            # Extract features (keypoints only)
            features = []
            for kp in keypoints:
                features.extend([kp['x'], kp['y'], kp['z']])
            
            # Append the calculated angles (shoulder-hip-knee, wrist-shoulder-hip)
            features.append(frame_data.get('shoulder_hip_knee_angle', 0))  # Default to 0 if missing
            features.append(frame_data.get('wrist_shoulder_hip_angle', 0))  # Default to 0 if missing
            
            sequence.append(features)
        
        X.append(sequence)
        y.append(1 if frame_data['Form'] == 'Correct_Form' else 0)
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y

# Prepare the data (replace with your JSON file path)
json_path = r'C:\Users\alexc\Final_Project\Final-Project\keypoints_bicep_curl_with_angles.json'
X, y = prepare_data(json_path, window_size=30)

# Reshape the data for LSTM (batch, time_steps, num_features)
X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))  # Shape: (batch_size, 30, num_features)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build the LSTM model with dropout and reduced complexity
model = tf.keras.Sequential([
    layers.LSTM(32, return_sequences=True, input_shape=(30, X.shape[2])),  # Use dynamic num_features
    layers.BatchNormalization(),
    layers.Dropout(0.3),  # Dropout added
    layers.LSTM(32),
    layers.BatchNormalization(),
    layers.Dropout(0.3),  # Dropout added
    layers.Dense(32, activation='relu'),  # Reduced fully connected layer size
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model with learning rate scheduler
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Use EarlyStopping and ReduceLROnPlateau callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)

# Train the model with callbacks
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping, lr_scheduler])

# Save the model
model.save(r'C:\Users\alexc\Final_Project\Final-Project\model_bicep_curl_with_angles.keras')
