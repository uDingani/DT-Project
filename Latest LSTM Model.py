# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle
import sys
import os

# Load the model
try:
    model = load_model('C:/Users/Busiso/Documents/GitHub/DT-Project/trained_strain_gauge_model.h5')
    print("Model loaded successfully.")
    print("Expected input shape:", model.input_shape)
except FileNotFoundError:
    print("Error: Model file 'trained_strain_gauge_model.h5' not found!")
    sys.exit()
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit()

# Load training data
try:
    train_data = pd.read_csv('C:/Users/Busiso/Desktop/EXCEL FILES/SUCCESSFUL/Success.csv')
    print("Training data loaded successfully.")
    print("Columns in train_data:", train_data.columns.tolist())
    print("First 5 rows of train_data:\n", train_data.head())
except FileNotFoundError:
    print("Error: Training data file 'Success.csv' not found!")
    sys.exit()
except Exception as e:
    print(f"Error loading training data: {e}")
    sys.exit()

# Standardize column names to lowercase
train_data.columns = train_data.columns.str.lower().str.strip()

# Load new data
file_path = 'C:/Users/Busiso/Desktop/60_G1_successful.xlsx'
if not os.path.exists(file_path):
    print(f"Error: File '{file_path}' does not exist!")
    sys.exit()
try:
    new_data = pd.read_excel(file_path)
    print("New data loaded successfully as Excel.")
except Exception as e:
    print(f"Failed to load as Excel: {e}")
    sys.exit()

# Standardize new data column names
new_data.columns = new_data.columns.str.lower().str.strip()
print("Columns in new_data:", new_data.columns.tolist())

# Verify required columns
required_cols = ['time', 'voltage']
missing_cols = [col for col in required_cols if col not in new_data.columns]
if missing_cols:
    raise ValueError(f"Missing required columns in new_data: {missing_cols}")

# Extract time and voltage
time = new_data['time'].values
voltage = new_data['voltage'].values

# Feature engineering
train_data['voltage_diff'] = train_data['voltage'].diff().fillna(0)
train_data['rolling_mean'] = train_data['voltage'].rolling(window=10).mean().bfill()
train_data['rolling_std'] = train_data['voltage'].rolling(window=10).std().bfill()

new_data['voltage_diff'] = new_data['voltage'].diff().fillna(0)
new_data['rolling_mean'] = new_data['voltage'].rolling(window=10).mean().bfill()
new_data['rolling_std'] = new_data['voltage'].rolling(window=10).std().bfill()

# Check model's expected input features
train_features = ['voltage', 'voltage_diff', 'rolling_mean', 'rolling_std']
if model.input_shape[-1] == 1:
    train_features = ['voltage']  # Reduce features if model expects a single input
    print("Using only 'voltage' as input feature.")

train_X = train_data[train_features]
new_X = new_data[train_features]

# Create sequences
def create_sequences(X, time_steps=50):
    Xs = []
    for i in range(len(X) - time_steps):
        Xs.append(X.iloc[i:(i + time_steps)].values)
    return np.array(Xs)

time_steps = 50
train_X_seq = create_sequences(train_X, time_steps)
new_X_seq = create_sequences(new_X, time_steps)

print("train_X_seq shape:", train_X_seq.shape)
print("new_X_seq shape:", new_X_seq.shape)

# Scale the data
scaler = StandardScaler()
try:
    scaler.fit(train_X_seq.reshape(-1, train_X_seq.shape[2]))
    print("Scaler fitted to training data.")
except Exception as e:
    print(f"Error fitting scaler: {e}")
    sys.exit()

try:
    new_X_scaled = scaler.transform(new_X_seq.reshape(-1, new_X_seq.shape[2])).reshape(new_X_seq.shape)
    print("new_X_scaled shape:", new_X_scaled.shape)
except Exception as e:
    print(f"Error scaling new data: {e}")
    sys.exit()

# Reshape to match LSTM input
new_X_scaled = new_X_scaled.reshape((new_X_scaled.shape[0], time_steps, new_X_scaled.shape[2]))

# Make predictions
try:
    predictions = model.predict(new_X_scaled)
    print("Predictions shape:", predictions.shape)
except Exception as e:
    print(f"Error during prediction: {e}")
    sys.exit()

# Filter reliable points
threshold = 0.5
reliable_indices = [i + time_steps for i, pred in enumerate(predictions) if pred[0] <= threshold]

# Calculate strain
gauge_factor = 2.0
excitation_voltage = 5.0
strain = voltage / (gauge_factor * excitation_voltage)
reliable_strain = strain[reliable_indices]
reliable_time = time[reliable_indices]

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(time, strain, label="All Strain Data", alpha=0.5)
plt.plot(reliable_time, reliable_strain, 'ro', label="Reliable Strain (Gauge Tight)")
plt.xlabel("Time (s)")
plt.ylabel("Strain")
plt.title("Strain from Voltage with LSTM Filtering")
plt.legend()
plt.grid(True)
plt.show()

# Print results
print(f"Number of reliable points: {len(reliable_indices)} out of {len(voltage)}")
print(f"Sample reliable strain values: {reliable_strain[:5]}")

# Save the scaler
with open('C:/Users/Busiso/Desktop/reconstructed_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Scaler saved as 'reconstructed_scaler.pkl'.")
