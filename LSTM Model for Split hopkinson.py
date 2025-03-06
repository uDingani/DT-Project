# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 17:12:29 2025

@author: Busiso
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# 1. Load and Prepare the Data
successful_file = 'path_to_successful_tests.xlsx'  
failed_file = 'path_to_failed_tests.xlsx'        

# Load the Excel files
successful_data = pd.read_excel(successful_file)
failed_data = pd.read_excel(failed_file)


successful_data['Label'] = 0
failed_data['Label'] = 1

# Combine into one dataset
all_data = pd.concat([successful_data, failed_data], ignore_index=True)


print("Columns in data:", all_data.columns)

# 2. Feature Engineering

all_data['Voltage_Diff'] = all_data['Voltage'].diff().fillna(0)  # Rate of change
all_data['Rolling_Mean'] = all_data['Voltage'].rolling(window=10).mean().fillna(method='bfill')
all_data['Rolling_Std'] = all_data['Voltage'].rolling(window=10).std().fillna(method='bfill')


features = ['Voltage', 'Voltage_Diff', 'Rolling_Mean', 'Rolling_Std']
X = all_data[features]
y = all_data['Label']

#  Create Sequences for LSTM
# The LSTM needs 3D input to work i.e (samples, timesteps, features)
def create_sequences(X, y, time_steps=50):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X.iloc[i:(i + time_steps)].values)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 50 
X_seq, y_seq = create_sequences(X, y, time_steps)

# Split and Scale the Data
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)

# Scale the features 
scaler = StandardScaler()
# Reshape for scaling: (samples * timesteps, features)
X_train_2d = X_train.reshape(-1, X_train.shape[2])
X_test_2d = X_test.reshape(-1, X_test.shape[2])

X_train_scaled = scaler.fit_transform(X_train_2d).reshape(X_train.shape)
X_test_scaled = scaler.transform(X_test_2d).reshape(X_test.shape)

#  Build the LSTM Model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(time_steps, X_train.shape[2])))
model.add(Dropout(0.2))  # Prevent overfitting
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid'))  # Binary output: 0 (good) or 1 (loose)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()

# 6. Train the Model
history = model.fit(X_train_scaled, y_train, epochs=20, batch_size=32, 
                    validation_data=(X_test_scaled, y_test), verbose=1)

# 7. Evaluate the Model
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Plot training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 8. Predict on New Data (Example)
# Load new data for prediction (replace with your new test file)
new_data = pd.read_excel('path_to_new_test_data.xlsx')
new_data['Voltage_Diff'] = new_data['Voltage'].diff().fillna(0)
new_data['Rolling_Mean'] = new_data['Voltage'].rolling(window=10).mean().fillna(method='bfill')
new_data['Rolling_Std'] = new_data['Voltage'].rolling(window=10).std().fillna(method='bfill')
new_X = new_data[features]

# Create sequences for the new data
new_X_seq, _ = create_sequences(new_X, pd.Series([0]*len(new_X)), time_steps)  # Dummy y for shape
new_X_scaled = scaler.transform(new_X_seq.reshape(-1, new_X_seq.shape[2])).reshape(new_X_seq.shape)

# Make predictions
predictions = model.predict(new_X_scaled)
threshold = 0.5  # Adjust based on your needs
for i, pred in enumerate(predictions):
    if pred > threshold:
        print(f"Time step {i + time_steps}: Gauge might be loosening! (Probability: {pred[0]:.2f})")
    else:
        print(f"Time step {i + time_steps}: Gauge looks good. (Probability: {pred[0]:.2f})")

# 9. Save the Model (Optional)
model.save('strain_gauge_lstm_model.h5')
print("Model saved as 'strain_gauge_lstm_model.h5'")