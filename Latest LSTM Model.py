# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle

# Load the pre-trained model with error handling
try:
    model = load_model('C:/Users/Busiso/Desktop/failure_predictor_model.h5')
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: Model file 'failure_predictor_model.h5' not found!")
    exit()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Load the scaler (assumed to be saved during training)
try:
    with open('C:/Users/Busiso/Desktop/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("Scaler loaded successfully.")
except FileNotFoundError:
    print("Error: Scaler file 'scaler.pkl' not found! You need to load or fit a scaler.")
    exit()
except Exception as e:
    print(f"Error loading scaler: {e}")
    exit()

# Load new data
try:
    new_data = pd.read_excel('C:/Users/Busiso/Desktop/60_G1_successful.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: Data file '60_G1_successful.csv' not found!")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# Extract time and voltage
time = new_data['Time'].values
voltage = new_data['Voltage'].values  


new_data['Voltage_Diff'] = new_data['Voltage'].diff().fillna(0)
new_data['Rolling_Mean'] = new_data['Voltage'].rolling(window=10).mean().fillna(method='bfill')
new_data['Rolling_Std'] = new_data['Voltage'].rolling(window=10).std().fillna(method='bfill')
features = ['Voltage', 'Voltage_Diff', 'Rolling_Mean', 'Rolling_Std']
new_X = new_data[features]


def create_sequences(X, time_steps=50):
    Xs = []
    for i in range(len(X) - time_steps):
        Xs.append(X.iloc[i:(i + time_steps)].values)
    return np.array(Xs)


time_steps = 50
new_X_seq = create_sequences(new_X, time_steps)
print("new_X_seq shape:", new_X_seq.shape)


try:
    new_X_scaled = scaler.transform(new_X_seq.reshape(-1, new_X_seq.shape[2])).reshape(new_X_seq.shape)
    print("new_X_scaled shape:", new_X_scaled.shape)
except Exception as e:
    print(f"Error scaling data: {e}")
    exit()


try:
    predictions = model.predict(new_X_scaled)
    print("Predictions shape:", predictions.shape)
except Exception as e:
    print(f"Error during prediction: {e}")
    exit()

# Filter reliable points
threshold = 0.5
reliable_indices = [i + time_steps for i, pred in enumerate(predictions) if pred[0] <= threshold]

# Calculate strain
gauge_factor = 2.0
excitation_voltage = 5.0
strain = voltage / (gauge_factor * excitation_voltage)
reliable_strain = strain[reliable_indices]
reliable_time = time[reliable_indices]

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(time, strain, label="All Strain Data", alpha=0.5)
plt.plot(reliable_time, reliable_strain, 'ro', label="Reliable Strain (Gauge Tight)")
plt.xlabel("Time (s)")
plt.ylabel("Strain")
plt.title("Strain from Voltage with LSTM Filtering")
plt.legend()
plt.grid(True)
plt.show()

# Output results
print(f"Number of reliable points: {len(reliable_indices)} out of {len(voltage)}")
print(f"Sample reliable strain values: {reliable_strain[:5]}")