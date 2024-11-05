import numpy as np
import json
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
import wandb
from wandb.integration.keras import WandbCallback

wandb.init(project="shoulder_press_detection", name="Shoulder Press Model")

form_label_mapping = {
    'Correct_Form': 0,
    'Unbalanced': 1,
}

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
            

            # Adding the new angles
            features.append(frame_data.get('right_elbow_left_elbow_right_wrist_angle', 0))
            features.append(frame_data.get('left_elbow_right_elbow_left_wrist_angle', 0))
            
            sequence.append(features)
        
        X.append(sequence)
        y.append(form_label_mapping.get(frame_data['Form'], -1))
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y

json_path = r'C:\Users\alexc\Final_Project\Final-Project\keypoints_shoulder_press_with_angles.json'
X, y = prepare_data(json_path, window_size=30)

X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = tf.keras.Sequential([
    layers.Bidirectional(layers.LSTM(64, return_sequences=True), input_shape=(30, X.shape[2])),
    layers.Dropout(0.3),
    layers.Bidirectional(layers.LSTM(32)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(form_label_mapping), activation='softmax')
])


class_weight_dict = {0: 1.5, 1: 1.0}
print("Custom class weights:", class_weight_dict)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7, verbose=1)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])


model.fit(X_train, y_train, epochs=50, batch_size=32, 
          validation_data=(X_test, y_test), 
          class_weight=class_weight_dict,
          callbacks=[early_stopping, lr_scheduler, WandbCallback(save_graph=False, save_model=False)])

model.save(r'C:\Users\alexc\Final_Project\Final-Project\model_shoulder_press.keras')