# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 14:48:36 2025

@author: Busiso
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
tf.config.run_functions_eagerly(True) 
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load datasets
data_success = pd.read_csv('C:/Users/Busiso/Desktop/60_G1_successful.csv')
data_fail = pd.read_csv('C:/Users/Busiso/Desktop/60PSI_G1_failed.csv')

# Rename columns
data_fail.columns = ['Voltage', 'Time']
data_success.columns = ['Voltage', 'Time']

# Add failure labels
data_fail['failure'] = 1  # Failed samples
data_success['failure'] = 0  # Successful samples

# Combine both datasets
data = pd.concat([data_fail, data_success], ignore_index=True)

# Define features and target
features = data[['Voltage', 'Time']]
target = data['failure']

# Scale the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

# Reshape data only if the model expects 3D input
if len(X_train.shape) == 2:  # Check if it's 2D (samples, features)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Try loading an existing model with error handling
try:
    model = load_model('C:/Users/Busiso/Desktop/failure_predictor_model.h5')
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading model:", e)
    exit()  # Stop execution if model loading fails

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Set up early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.001)

# Learning rate scheduler to reduce learning rate when training slows down
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, lr_scheduler]
)

# Make predictions
y_pred = model.predict(X_test)
y_pred_class = (y_pred > 0.5).astype('int32')

# Evaluate performance
print('Accuracy:', accuracy_score(y_test, y_pred_class))
print('Classification Report:')
print(classification_report(y_test, y_pred_class))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred_class))

# Save the trained model
model.save('C:/Users/Busiso/Desktop/trained_strain_gauge_model.h5')
print("Trained model saved as 'trained_strain_gauge_model.h5'")

# Plot training loss
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.show()
