# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 17:12:29 2025

@author: Busiso
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import joblib
import psutil

# Flexible file input
successful_file = input("Enter path to successful Excel file: ")
failed_file = input("Enter path to failed Excel file: ")

try:
    successful_data = pd.read_excel(successful_file)
    failed_data = pd.read_excel(failed_file)
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    exit(1)

successful_data['Label'] = 0
failed_data['Label'] = 1
all_data = pd.concat([successful_data, failed_data], ignore_index=True)

def identify_columns(df):
    """Identify columns containing 'time' or 'voltage' in their names."""
    time_cols = [col for col in df.columns if 'time' in col.lower()]
    voltage_cols = [col for col in df.columns if 'voltage' in col.lower()]
    return time_cols, voltage_cols

# Column validation
time_cols, voltage_cols = identify_columns(all_data)
print(f"Detected time columns: {time_cols}")
print(f"Detected voltage columns: {voltage_cols}")
if not voltage_cols:
    raise ValueError("No voltage columns found in the data.")
if not time_cols:
    raise ValueError("No time columns found in the data.")
if len(time_cols) != len(voltage_cols):
    print("Warning: Number of time and voltage columns mismatch.")

# Feature engineering
features = []
for v_col in voltage_cols:
    diff_col = f'{v_col}_Diff'
    mean_col = f'{v_col}_Rolling_Mean'
    std_col = f'{v_col}_Rolling_Std'
    all_data[diff_col] = all_data[v_col].diff().fillna(0)
    all_data[mean_col] = all_data[v_col].rolling(window=10).mean().fillna(method='bfill')
    all_data[std_col] = all_data[v_col].rolling(window=10).std().fillna(method='bfill')
    features.extend([v_col, diff_col, mean_col, std_col])

if len(voltage_cols) > 1:
    for i in range(len(voltage_cols)):
        for j in range(i + 1, len(voltage_cols)):
            ratio_col = f'Ratio_{voltage_cols[i]}_{voltage_cols[j]}'
            diff_col = f'Diff_{voltage_cols[i]}_{voltage_cols[j]}'
            all_data[ratio_col] = all_data[voltage_cols[i]] / all_data[voltage_cols[j]].replace(0, np.nan)
            all_data[diff_col] = all_data[voltage_cols[i]] - all_data[voltage_cols[j]]
            features.extend([ratio_col, diff_col])

X = all_data[features].fillna(0)  # Handle NaNs from division by zero
y = all_data['Label']

# Memory-optimized sequence creation
def create_sequences_chunked(X, y, time_steps=50, chunk_size=None):
    """Create sequences for LSTM input with dynamic chunk size."""
    if chunk_size is None:
        available_memory = psutil.virtual_memory().available / (1024 ** 2)  # MB
        chunk_size = min(10000, int(available_memory / (X.shape[1] * 8)))  # 8 bytes per float
    Xs, ys = [], []
    try:
        for start in range(0, len(X) - time_steps, chunk_size):
            end = min(start + chunk_size, len(X) - time_steps)
            chunk_X = X.iloc[start:end + time_steps]
            chunk_y = y.iloc[start:end + time_steps]
            for i in range(len(chunk_X) - time_steps):
                Xs.append(chunk_X.iloc[i:(i + time_steps)].values)
                ys.append(chunk_y.iloc[i + time_steps])
            print(f"Processed chunk {start} to {end}, Memory usage: {psutil.virtual_memory().percent}%")
    except MemoryError:
        print("Memory error occurred. Try reducing time_steps or available memory is too low.")
        raise
    return np.array(Xs), np.array(ys)

time_steps = 50
try:
    print("Creating training sequences...")
    X_seq, y_seq = create_sequences_chunked(X, y, time_steps)
except MemoryError:
    print("Failed to create sequences due to memory constraints.")
    exit(1)

# Data splitting and scaling
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[2])).reshape(X_train.shape)
X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[2])).reshape(X_test.shape)

# Model definition
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(time_steps, len(features))),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Training
history = model.fit(X_train_scaled, y_train, epochs=3, batch_size=32,
                    validation_data=(X_test_scaled, y_test), verbose=1)

# Enhanced evaluation
loss, accuracy = model.evaluate(X_test_scaled, y_test)
y_pred = (model.predict(X_test_scaled) > 0.5).astype(int)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")

# Enhanced plotting
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

# Prediction on new data
new_file = input("Enter path to new Excel file for prediction: ")
try:
    new_data = pd.read_excel(new_file)
    _, voltage_cols_new = identify_columns(new_data)
    new_features = []
    for v_col in voltage_cols_new:
        diff_col = f'{v_col}_Diff'
        mean_col = f'{v_col}_Rolling_Mean'
        std_col = f'{v_col}_Rolling_Std'
        new_data[diff_col] = new_data[v_col].diff().fillna(0)
        new_data[mean_col] = new_data[v_col].rolling(window=10).mean().fillna(method='bfill')
        new_data[std_col] = new_data[v_col].rolling(window=10).std().fillna(method='bfill')
        new_features.extend([v_col, diff_col, mean_col, std_col])
    if len(voltage_cols_new) > 1:
        for i in range(len(voltage_cols_new)):
            for j in range(i + 1, len(voltage_cols_new)):
                ratio_col = f'Ratio_{voltage_cols_new[i]}_{voltage_cols_new[j]}'
                diff_col = f'Diff_{voltage_cols_new[i]}_{voltage_cols_new[j]}'
                new_data[ratio_col] = new_data[voltage_cols_new[i]] / new_data[voltage_cols_new[j]].replace(0, np.nan)
                new_data[diff_col] = new_data[voltage_cols_new[i]] - new_data[voltage_cols_new[j]]
                new_features.extend([ratio_col, diff_col])
    new_X = new_data[new_features].fillna(0)
    print("Creating sequences for new data...")
    new_X_seq, _ = create_sequences_chunked(new_X, pd.Series([0]*len(new_X)), time_steps)
    new_X_scaled = scaler.transform(new_X_seq.reshape(-1, new_X_seq.shape[2])).reshape(new_X_seq.shape)
    predictions = model.predict(new_X_scaled)
    loosening_count = np.sum(predictions > 0.5)
    print(f"Predicted loosening instances: {loosening_count} out of {len(predictions)}")
except Exception as e:
    print(f"Error processing new data: {e}")
    exit(1)

# Model saving with error handling
try:
    model.save('strain_gauge_lstm_model.keras')
    print("Model saved as 'strain_gauge_lstm_model.keras'")
    joblib.dump(scaler, 'scaler.pkl')
    print("Scaler saved as 'scaler.pkl'")
except Exception as e:
    print(f"Error saving model or scaler: {e}")

# Verification
try:
    loaded_model = load_model('strain_gauge_lstm_model.keras')
    loaded_scaler = joblib.load('scaler.pkl')
    X_test_small = X_test_scaled[:10]
    y_test_small = y_test[:10]
    loaded_predictions = loaded_model.predict(X_test_small)
    print("\nVerification of loaded model:")
    for i, (pred, true) in enumerate(zip(loaded_predictions, y_test_small)):
        pred_value = pred[0]
        pred_label = 1 if pred_value > 0.5 else 0
        print(f"Sample {i}: Predicted={pred_label} (Prob: {pred_value:.2f}), True={true}")
except Exception as e:
    print(f"Error during verification: {e}")