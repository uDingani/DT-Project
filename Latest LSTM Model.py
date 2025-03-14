# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import pickle
import sys
import os
import psutil
import joblib 


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


train_data.columns = train_data.columns.str.lower().str.strip()


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


new_data.columns = new_data.columns.str.lower().str.strip()
print("Columns in new_data:", new_data.columns.tolist())


def identify_columns(df):
    """Identify columns containing 'time' or 'voltage' in their names."""
    time_cols = [col for col in df.columns if 'time' in col.lower()]
    voltage_cols = [col for col in df.columns if 'voltage' in col.lower()]
    return time_cols, voltage_cols

time_cols, voltage_cols = identify_columns(new_data)
print(f"Detected time columns: {time_cols}")
print(f"Detected voltage columns: {voltage_cols}")

if len(time_cols) < 1 or len(voltage_cols) < 1:
    raise ValueError("Could not find required time and/or voltage columns in the data")


missing_train_cols = [col for col in voltage_cols if col not in train_data.columns]
if missing_train_cols:
    raise ValueError(f"Voltage columns missing in training data: {missing_train_cols}")


time = new_data[time_cols[0]].values
voltages = {col: new_data[col].values for col in voltage_cols}


features = []
for v_col in voltage_cols:
    diff_col = f'{v_col}_diff'
    mean_col = f'{v_col}_rolling_mean'
    std_col = f'{v_col}_rolling_std'
    
    
    train_data[diff_col] = train_data[v_col].diff().fillna(0)
    train_data[mean_col] = train_data[v_col].rolling(window=10).mean().bfill()
    train_data[std_col] = train_data[v_col].rolling(window=10).std().bfill()
    
   
    new_data[diff_col] = new_data[v_col].diff().fillna(0)
    new_data[mean_col] = new_data[v_col].rolling(window=10).mean().bfill()
    new_data[std_col] = new_data[v_col].rolling(window=10).std().bfill()
    
    features.extend([v_col, diff_col, mean_col, std_col])


if len(voltage_cols) > 1:
    for i in range(len(voltage_cols)):
        for j in range(i + 1, len(voltage_cols)):
            ratio_col = f'ratio_{voltage_cols[i]}_{voltage_cols[j]}'
            diff_col = f'diff_{voltage_cols[i]}_{voltage_cols[j]}'
            
            
            train_data[ratio_col] = np.where(
                train_data[voltage_cols[j]] != 0,
                train_data[voltage_cols[i]] / train_data[voltage_cols[j]],
                0
            )
            train_data[diff_col] = train_data[voltage_cols[i]] - train_data[voltage_cols[j]]
            
            new_data[ratio_col] = np.where(
                new_data[voltage_cols[j]] != 0,
                new_data[voltage_cols[i]] / new_data[voltage_cols[j]],
                0
            )
            new_data[diff_col] = new_data[voltage_cols[i]] - new_data[voltage_cols[j]]
            
            features.extend([ratio_col, diff_col])


expected_features = model.input_shape[-1]
if len(features) != expected_features:
    print(f"Warning: Number of features ({len(features)}) does not match model input shape ({expected_features})")
    print(f"Features generated: {features}")

train_X = train_data[features]
new_X = new_data[features]


def create_sequences_chunked(X, time_steps=50, chunk_size=10000):
    """Create sequences for LSTM input in chunks to manage memory."""
    Xs = []
    try:
        for start in range(0, len(X) - time_steps, chunk_size):
            end = min(start + chunk_size, len(X) - time_steps)
            chunk_X = X.iloc[start:end + time_steps]
            for i in range(len(chunk_X) - time_steps):
                Xs.append(chunk_X.iloc[i:(i + time_steps)].values)
            print(f"Processed chunk {start} to {end}")
            print(f"Memory usage: {psutil.virtual_memory().percent}%")
    except MemoryError:
        print("Memory error occurred. Try reducing chunk_size or time_steps.")
        raise
    return np.array(Xs)

time_steps = 50
chunk_size = 20000 

try:
    print("Creating training sequences...")
    train_X_seq = create_sequences_chunked(train_X, time_steps, chunk_size)
    print("Creating new data sequences...")
    new_X_seq = create_sequences_chunked(new_X, time_steps, chunk_size)
except MemoryError:
    print("Failed to create sequences. Consider reducing chunk_size or time_steps.")
    sys.exit()

print("train_X_seq shape:", train_X_seq.shape)
print("new_X_seq shape:", new_X_seq.shape)


scaler_path = 'C:/Users/Busiso/Desktop/scaler.pkl'  
try:
    scaler = joblib.load(scaler_path)
    print("Pre-trained scaler loaded successfully from 'scaler.pkl'.")
except FileNotFoundError:
    print(f"Error: Scaler file '{scaler_path}' not found!")
    sys.exit()
except Exception as e:
    print(f"Error loading scaler: {e}")
    sys.exit()


try:
    new_X_scaled = scaler.transform(new_X_seq.reshape(-1, new_X_seq.shape[2])).reshape(new_X_seq.shape)
    print("new_X_scaled shape:", new_X_scaled.shape)
except Exception as e:
    print(f"Error scaling new data: {e}")
    sys.exit()


new_X_scaled = new_X_scaled.reshape((new_X_scaled.shape[0], time_steps, new_X_scaled.shape[2]))


try:
    predictions = model.predict(new_X_scaled)
    print("Predictions shape:", predictions.shape)
except Exception as e:
    print(f"Error during prediction: {e}")
    sys.exit()


threshold = 0.5
reliable_indices = [i + time_steps for i, pred in enumerate(predictions) if pred[0] <= threshold]


gauge_factor = 2.0
excitation_voltage = 5.0
strain = voltages[voltage_cols[0]] / (gauge_factor * excitation_voltage)
reliable_strain = strain[reliable_indices]
reliable_time = time[reliable_indices]


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
print(f"Number of reliable points: {len(reliable_indices)} out of {len(strain)}")
print(f"Sample reliable strain values: {reliable_strain[:5]}")