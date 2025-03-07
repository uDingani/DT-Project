# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle
import sys


try:
    model = load_model('C:/Users/Busiso/Documents/GitHub/DT-Project/trained_strain_gauge_model.h5')
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: Model file 'trained_strain_gauge_model.h5' not found!")
    sys.exit()
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit()


try:
    train_data = pd.read_csv('C:/Users/Busiso/Desktop/EXCEL FILES/SUCCESSFUL/Success.csv')  
    print("Training data loaded successfully.")
except FileNotFoundError:
    print("Error: Training data file 'Success.csv' not found!")
    sys.exit()
except Exception as e:
    print(f"Error loading training data: {e}")
    sys.exit()


try:
    new_data = pd.read_excel('C:/Users/Busiso/Desktop/60_G1_successful.xlsx')
    print("New data loaded successfully.")
except FileNotFoundError:
    print("Error: Data file '60_G1_successful.xlsx' not found!")
    sys.exit()
except Exception as e:
    print(f"Error loading data: {e}")
    sys.exit()


time = new_data['Time'].values
voltage = new_data['Voltage'].values  


train_data['Voltage_Diff'] = train_data['Voltage'].diff().fillna(0)
train_data['Rolling_Mean'] = train_data['Voltage'].rolling(window=10).mean().fillna(method='bfill')
train_data['Rolling_Std'] = train_data['Voltage'].rolling(window=10).std().fillna(method='bfill')
train_features = ['Voltage', 'Voltage_Diff', 'Rolling_Mean', 'Rolling_Std']
train_X = train_data[train_features]


new_data['Voltage_Diff'] = new_data['Voltage'].diff().fillna(0)
new_data['Rolling_Mean'] = new_data['Voltage'].rolling(window=10).mean().fillna(method='bfill')
new_data['Rolling_Std'] = new_data['Voltage'].rolling(window=10).std().fillna(method='bfill')
new_X = new_data[train_features]  # Use same features as training


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
strain = voltage / (gauge_factor * excitation_voltage)
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


print(f"Number of reliable points: {len(reliable_indices)} out of {len(voltage)}")
print(f"Sample reliable strain values: {reliable_strain[:5]}")


with open('C:/Users/Busiso/Desktop/reconstructed_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Scaler saved as 'reconstructed_scaler.pkl'.")
