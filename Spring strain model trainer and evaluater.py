# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 14:48:36 2025

@author: Busiso
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 13:32:19 2025

@author: a2yna
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
tf.config.run_functions_eagerly(True) 
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt


data_success = pd.read_csv('C:/Users/Busiso/Desktop/60_G1_successful.csv')
data_fail = pd.read_csv('C:/Users/Busiso/Desktop/60PSI_G1_failed.csv')

data_fail.columns = ['Voltage', 'Time']
data_success.columns = ['Voltage', 'Time']

data_fail['failure'] = 1
data_success['failure'] = 0


data = pd.concat([data_fail, data_success], ignore_index=True)


features = data[['Voltage', 'Time']]
target = data['failure']


scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)


X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)


X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))


model = load_model('C:/Users/Busiso/Desktop/failure_predictor_model.h5')
print("Model loaded from 'failure_predictor_model.h5'")


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


early_stopping = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.001)
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])


y_pred = model.predict(X_test)
y_pred_class = (y_pred > 0.5).astype('int32')


print('Accuracy:', accuracy_score(y_test, y_pred_class))
print('Classification Report:')
print(classification_report(y_test, y_pred_class))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred_class))

# Plotting with tweaks
X_test_2d = X_test.reshape(X_test.shape[0], X_test.shape[1])  
X_test_unscaled = scaler.inverse_transform(X_test_2d)

voltage_test = X_test_unscaled[:, 0]  
time_test = X_test_unscaled[:, 1]   

plt.figure(figsize=(12, 8)) 
for i in range(len(y_test)):
    if y_test.iloc[i] == 1:  
        plt.scatter(time_test[i], voltage_test[i], color='red', s=10, alpha=0.5, label='Failure' if i == 0 else "")
    else:  # Successes
        plt.scatter(time_test[i], voltage_test[i], color='green', s=10, alpha=0.5, label='Success' if i == 0 else "")

plt.axhline(y=14, color='blue', linestyle='--', label='Danger Voltage (14)')
plt.axvline(x=20, color='blue', linestyle='--', label='Danger Time (20)')

plt.xlabel('Time')
plt.ylabel('Voltage')
plt.title('Voltage vs Time: Failures (Red) vs Successes (Green)')
plt.legend()
plt.grid(True)
plt.show()


model.save('C:/Users/Busiso/Desktop/trained_strain_gauge_model.h5')
print("Trained model saved as 'trained_model.h5'")