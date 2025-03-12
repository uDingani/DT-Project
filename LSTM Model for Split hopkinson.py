# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 17:12:29 2025

@author: Busiso
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import joblib


successful_file = 'C:/Users/Busiso/Desktop/hanna results/60_G1_successful.xlsx'
failed_file = 'C:/Users/Busiso/Desktop/hanna results/60PSI_G1_failed.xlsx'


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

time_cols, voltage_cols = identify_columns(all_data)
print(f"Detected time columns: {time_cols}")
print(f"Detected voltage columns: {voltage_cols}")

if len(voltage_cols) < 1 or len(time_cols) < 1:
    raise ValueError("Could not find required time and/or voltage columns in the data")


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
            all_data[ratio_col] = all_data[voltage_cols[i]] / all_data[voltage_cols[j]]
            all_data[diff_col] = all_data[voltage_cols[i]] - all_data[voltage_cols[j]]
            features.extend([ratio_col, diff_col])

X = all_data[features]
y = all_data['Label']


def create_sequences(X, y, time_steps=50):
    """Create sequences for LSTM input."""
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X.iloc[i:(i + time_steps)].values)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 50
X_seq, y_seq = create_sequences(X, y, time_steps)


X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)

scaler = StandardScaler()
X_train_2d = X_train.reshape(-1, X_train.shape[2])
X_test_2d = X_test.reshape(-1, X_test.shape[2])

X_train_scaled = scaler.fit_transform(X_train_2d).reshape(X_train.shape)
X_test_scaled = scaler.transform(X_test_2d).reshape(X_test.shape)


model = Sequential()
model.add(LSTM(units=64, return_sequences=True, input_shape=(time_steps, len(features))))
model.add(Dropout(0.2))
model.add(LSTM(units=64))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()


history = model.fit(X_train_scaled, y_train, epochs=3, batch_size=32,
                    validation_data=(X_test_scaled, y_test), verbose=1)


loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


new_data = pd.read_excel('C:/Users/Busiso/Desktop/hanna results/70PSI_successful(2-4-25).xlsx')
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
            new_data[ratio_col] = new_data[voltage_cols_new[i]] / new_data[voltage_cols_new[j]]
            new_data[diff_col] = new_data[voltage_cols_new[i]] - new_data[voltage_cols_new[j]]
            new_features.extend([ratio_col, diff_col])

new_X = new_data[new_features]
new_X_seq, _ = create_sequences(new_X, pd.Series([0]*len(new_X)), time_steps)
new_X_scaled = scaler.transform(new_X_seq.reshape(-1, new_X_seq.shape[2])).reshape(new_X_seq.shape)

predictions = model.predict(new_X_scaled)
threshold = 0.5
for i, pred in enumerate(predictions):
    if pred > threshold:
        print(f"Time step {i + time_steps}: Gauge might be loosening! (Probability: {pred[0]:.2f})")
    else:
        print(f"Time step {i + time_steps}: Gauge looks good. (Probability: {pred[0]:.2f})")


model.save('C:/Users/Busiso/Desktop/strain_gauge_lstm_model.keras')
print("Model saved as 'strain_gauge_lstm_model.keras'")


joblib.dump(scaler, 'scaler.pkl')
print("Scaler saved as 'scaler.pkl'")



loaded_model = load_model('C:/Users/Busiso/Desktop/strain_gauge_lstm_model.keras')
loaded_scaler = joblib.load('C:/Users/Busiso/Desktop/scaler.pkl')


X_test_small = X_test_scaled[:10]
y_test_small = y_test[:10]


loaded_predictions = loaded_model.predict(X_test_small)
print("\nVerification of loaded model:")
for i, (pred, true) in enumerate(zip(loaded_predictions, y_test_small)):
    pred_value = pred[0]
    pred_label = 1 if pred_value > threshold else 0
    print(f"Sample {i}: Predicted={pred_label} (Prob: {pred_value:.2f}), True={true}")


original_predictions = model.predict(X_test_small)
print("\nComparison with original model:")
for i, (orig_pred, loaded_pred) in enumerate(zip(original_predictions, loaded_predictions)):
    print(f"Sample {i}: Original={orig_pred[0]:.4f}, Loaded={loaded_pred[0]:.4f}")