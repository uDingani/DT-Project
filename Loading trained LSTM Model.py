# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import joblib
import sys
import os
import psutil
import warnings

# Load pre-trained model
model_path = 'C:/Users/Busiso/Desktop/hanna results/strain_gauge_lstm_model.keras'
try:
    model = load_model(model_path)
    print("Model loaded successfully.")
    print("Expected input shape:", model.input_shape)
except FileNotFoundError:
    print(f"Error: Model file '{model_path}' not found!")
    sys.exit(1)
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# Load expected feature names
feature_names_path = 'C:/Users/Busiso/Desktop/hanna results/feature_names.pkl'
try:
    expected_features = joblib.load(feature_names_path)
    print(f"Loaded expected features: {expected_features}")
    expected_feature_count = len(expected_features)
    if expected_feature_count != model.input_shape[-1]:
        raise ValueError(f"Feature count in file ({expected_feature_count}) does not match model input ({model.input_shape[-1]})")
except FileNotFoundError:
    print(f"Error: Feature names file '{feature_names_path}' not found! Please provide the feature list from training.")
    sys.exit(1)
except Exception as e:
    print(f"Error loading feature names: {e}")
    sys.exit(1)

# Load training data from Excel
train_data_path = 'C:/Users/Busiso/Desktop/hanna results/70PSI_successful(2-4-25).xlsx'
if not os.path.exists(train_data_path):
    print(f"Error: Training data file '{train_data_path}' not found!")
    sys.exit(1)
try:
    train_data = pd.read_excel(train_data_path)
    print("Training data loaded successfully from Excel.")
    print("Columns in train_data:", train_data.columns.tolist())
except Exception as e:
    print(f"Error loading training data: {e}")
    sys.exit(1)

# Load new data from Excel
new_data_path = 'C:/Users/Busiso/Desktop/hanna results/80PSI_failed(2-20-25).xlsx'
if not os.path.exists(new_data_path):
    print(f"Error: File '{new_data_path}' does not exist!")
    sys.exit(1)
try:
    new_data = pd.read_excel(new_data_path)
    print("New data loaded successfully from Excel.")
except Exception as e:
    print(f"Failed to load new data as Excel: {e}")
    sys.exit(1)

# Standardize column names
train_data.columns = train_data.columns.str.lower().str.strip()
new_data.columns = new_data.columns.str.lower().str.strip()
print("Columns in new_data:", new_data.columns.tolist())

# Identify columns
def identify_columns(df):
    """Identify columns containing 'time' or 'voltage' in their names."""
    time_cols = [col for col in df.columns if 'time' in col.lower()]
    voltage_cols = [col for col in df.columns if 'voltage' in col.lower()]
    if not time_cols or not voltage_cols:
        raise ValueError("Could not find required time and/or voltage columns")
    return time_cols, voltage_cols

train_time_cols, train_voltage_cols = identify_columns(train_data)
new_time_cols, new_voltage_cols = identify_columns(new_data)
print(f"Training data - Detected time columns: {train_time_cols}")
print(f"Training data - Detected voltage columns: {train_voltage_cols}")
print(f"New data - Detected time columns: {new_time_cols}")
print(f"New data - Detected voltage columns: {new_voltage_cols}")

# Use only common voltage columns
common_voltage_cols = [col for col in new_voltage_cols if col in train_voltage_cols]
if not common_voltage_cols:
    raise ValueError("No common voltage columns between training and new data!")
if len(common_voltage_cols) < len(new_voltage_cols):
    warnings.warn(f"Some voltage columns in new data are missing in training data: "
                  f"{[col for col in new_voltage_cols if col not in common_voltage_cols]}")
print(f"Using common voltage columns: {common_voltage_cols}")

# Feature engineering
def engineer_features(df, voltage_cols):
    features = []
    for v_col in voltage_cols:
        diff_col = f'{v_col}_diff'
        mean_col = f'{v_col}_rolling_mean'
        std_col = f'{v_col}_rolling_std'
        df[diff_col] = df[v_col].diff().fillna(0)
        df[mean_col] = df[v_col].rolling(window=10).mean().bfill()
        df[std_col] = df[v_col].rolling(window=10).std().bfill()
        features.extend([v_col, diff_col, mean_col, std_col])

    if len(voltage_cols) > 1:
        for i in range(len(voltage_cols)):
            for j in range(i + 1, len(voltage_cols)):
                ratio_col = f'ratio_{voltage_cols[i]}_{voltage_cols[j]}'
                diff_col = f'diff_{voltage_cols[i]}_{voltage_cols[j]}'
                df[ratio_col] = np.where(df[voltage_cols[j]] != 0, df[voltage_cols[i]] / df[voltage_cols[j]], 0)
                df[diff_col] = df[voltage_cols[i]] - df[voltage_cols[j]]
                features.extend([ratio_col, diff_col])
    return df, features

train_data, train_features = engineer_features(train_data, common_voltage_cols)
new_data, new_features = engineer_features(new_data, common_voltage_cols)
print(f"Generated features: {new_features}")

# Ensure all expected features are present
missing_features = [f for f in expected_features if f not in new_features]
if missing_features:
    warnings.warn(f"Missing features from new data: {missing_features}. Filling with zeros.")
    for f in missing_features:
        train_data[f] = 0
        new_data[f] = 0
    new_features = expected_features  # Use the expected feature list

if len(new_features) != expected_feature_count:
    raise ValueError(f"After adjustment, feature count ({len(new_features)}) still does not match model input ({expected_feature_count})")

train_X = train_data[new_features]
new_X = new_data[new_features]

# Sequence creation
def create_sequences_chunked(X, time_steps, chunk_size=10000):
    """Create sequences for LSTM input in chunks."""
    if len(X) < time_steps:
        print(f"Warning: Data length ({len(X)}) is less than time_steps ({time_steps}). No sequences will be generated.")
        return np.array([])
    Xs = []
    try:
        for start in range(0, len(X) - time_steps + 1, chunk_size):
            end = min(start + chunk_size, len(X) - time_steps + 1)
            chunk_X = X.iloc[start:end + time_steps]
            for i in range(end - start):
                Xs.append(chunk_X.iloc[i:(i + time_steps)].values)
            print(f"Processed chunk {start} to {end}")
            print(f"Memory usage: {psutil.virtual_memory().percent}%")
    except MemoryError:
        print("Memory error occurred. Try reducing chunk_size or time_steps.")
        raise
    return np.array(Xs)

# Use model’s expected time_steps
time_steps = model.input_shape[1]
chunk_size = 20000

try:
    print("Creating training sequences...")
    train_X_seq = create_sequences_chunked(train_X, time_steps, chunk_size)
    print("Creating new data sequences...")
    new_X_seq = create_sequences_chunked(new_X, time_steps, chunk_size)
except MemoryError:
    print("Failed to create sequences. Consider reducing chunk_size or time_steps.")
    sys.exit(1)

if train_X_seq.size == 0 or new_X_seq.size == 0:
    print("Error: No sequences generated. Check data length and time_steps.")
    sys.exit(1)

print("train_X_seq shape:", train_X_seq.shape)
print("new_X_seq shape:", new_X_seq.shape)

# Load scaler
scaler_path = 'C:/Users/Busiso/Desktop/hanna results/scaler.pkl'
try:
    scaler = joblib.load(scaler_path)
    print("Pre-trained scaler loaded successfully.")
except FileNotFoundError:
    print(f"Error: Scaler file '{scaler_path}' not found!")
    sys.exit(1)
except Exception as e:
    print(f"Error loading scaler: {e}")
    sys.exit(1)

# Scale new data
try:
    new_X_scaled = scaler.transform(new_X_seq.reshape(-1, new_X_seq.shape[2])).reshape(new_X_seq.shape)
    print("new_X_scaled shape:", new_X_scaled.shape)
except Exception as e:
    print(f"Error scaling new data: {e}")
    sys.exit(1)

# Ensure shape matches model input
if new_X_scaled.shape[1:] != model.input_shape[1:]:
    print(f"Error: Scaled data shape {new_X_scaled.shape[1:]} does not match model input {model.input_shape[1:]}")
    sys.exit(1)

# Predict
try:
    predictions = model.predict(new_X_scaled)
    print("Predictions shape:", predictions.shape)
except Exception as e:
    print(f"Error during prediction: {e}")
    sys.exit(1)

# Process predictions
threshold = 0.5
reliable_indices = [i + time_steps - 1 for i, pred in enumerate(predictions) if pred[0] <= threshold]

# Calculate strain
gauge_factor = 2.0  # Adjust if needed
excitation_voltage = 5.0  # Adjust if needed
time = new_data[new_time_cols[0]].values
voltages = {col: new_data[col].values for col in common_voltage_cols}
strain = voltages[common_voltage_cols[0]] / (gauge_factor * excitation_voltage)
reliable_strain = strain[reliable_indices]
reliable_time = time[reliable_indices]

# Plot
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