import numpy as np
import json
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split

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
            
            features = []
            for kp in keypoints:
                features.extend([kp['x'], kp['y'], kp['z']])
            
            features.append(frame_data.get('shoulder_hip_knee_angle', 0))
            features.append(frame_data.get('wrist_shoulder_hip_angle', 0))
            
            sequence.append(features)
        
        X.append(sequence)
        y.append(1 if frame_data['Form'] == 'Correct_Form' else 0)
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y

json_path = r'C:\Users\alexc\Final_Project\Final-Project\keypoints_bicep_curl_with_angles.json'
X, y = prepare_data(json_path, window_size=30)

X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = tf.keras.Sequential([
    layers.LSTM(32, return_sequences=True, input_shape=(30, X.shape[2])),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.LSTM(32),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)

model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping, lr_scheduler])

model.save(r'C:\Users\alexc\Final_Project\Final-Project\model_bicep_curl_with_angles.keras')
