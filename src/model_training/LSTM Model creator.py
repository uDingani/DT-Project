# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 17:12:29 2025
Updated on March 22, 2025 with enhancements

@author: Busiso
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import psutil
import logging
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Command-line arguments
parser = argparse.ArgumentParser(description='LSTM Model for Strain Gauge Data')
parser.add_argument('--time-steps', type=int, default=5, help='Number of time steps for sequences')
parser.add_argument('--sample-frac', type=float, default=1.0, help='Fraction of data to sample (0.0-1.0)')
args = parser.parse_args()

def validate_file_path(file_path):
    """Validate if file exists and is an Excel file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    if not file_path.lower().endswith(('.xlsx', '.xls')):
        raise ValueError(f"File must be an Excel file: {file_path}")
    return file_path

def load_data(successful_file, failed_file):
    """Load and label data from Excel files."""
    successful_file = validate_file_path(successful_file)
    failed_file = validate_file_path(failed_file)
    
    successful_data = pd.read_excel(successful_file)
    failed_data = pd.read_excel(failed_file)
    
    successful_data['Label'] = 0
    failed_data['Label'] = 1
    all_data = pd.concat([successful_data, failed_data], ignore_index=True)
    logging.info(f"Loaded {len(successful_data)} successful and {len(failed_data)} failed samples. Total: {len(all_data)}")
    return all_data

def identify_columns(df):
    """Identify time and voltage columns."""
    time_cols = [col for col in df.columns if 'time' in col.lower()]
    voltage_cols = [col for col in df.columns if 'voltage' in col.lower()]
    if not voltage_cols:
        raise ValueError("No voltage columns found in the data.")
    if not time_cols:
        logging.warning("No time columns found in the data.")
    if len(time_cols) != len(voltage_cols):
        logging.warning("Number of time and voltage columns mismatch.")
    return time_cols, voltage_cols

def engineer_features(df, voltage_cols):
    """Engineer features from voltage columns."""
    features = []
    for v_col in voltage_cols:
        diff_col = f'{v_col}_Diff'
        mean_col = f'{v_col}_Rolling_Mean'
        std_col = f'{v_col}_Rolling_Std'
        df[diff_col] = df[v_col].diff().fillna(0)
        df[mean_col] = df[v_col].rolling(window=10).mean().fillna(method='bfill')
        df[std_col] = df[v_col].rolling(window=10).std().fillna(method='bfill')
        features.extend([v_col, diff_col, mean_col, std_col])

    if len(voltage_cols) > 1:
        for i in range(len(voltage_cols)):
            for j in range(i + 1, len(voltage_cols)):
                ratio_col = f'Ratio_{voltage_cols[i]}_{voltage_cols[j]}'
                diff_col = f'Diff_{voltage_cols[i]}_{voltage_cols[j]}'
                df[ratio_col] = df[voltage_cols[i]] / df[voltage_cols[j]].replace(0, np.nan).fillna(0)
                df[diff_col] = df[voltage_cols[i]] - df[voltage_cols[j]]
                features.extend([ratio_col, diff_col])
    return df, features

def create_sequences_generator(X, y, time_steps, chunk_size=10000):
    """Generator for creating sequences in chunks."""
    if len(X) != len(y):
        raise ValueError(f"X and y lengths do not match: {len(X)} vs {len(y)}")
    logging.info(f"Creating sequences with {len(X)} rows, time_steps={time_steps}")
    if len(X) < time_steps:
        logging.warning(f"Data length {len(X)} < time_steps {time_steps}. Using all data as one sequence.")
        X_padded = np.pad(X.values, ((0, time_steps - len(X)), (0, 0)), mode='constant')
        yield X_padded[np.newaxis, :time_steps], np.array([y.iloc[-1]])
        return

    for start in range(0, len(X) - time_steps + 1, chunk_size):
        end = min(start + chunk_size, len(X) - time_steps + 1)
        chunk_end = min(end + time_steps - 1, len(X))
        chunk_X = X.iloc[start:chunk_end]
        chunk_y = y.iloc[start:chunk_end]
        Xs, ys = [], []
        logging.info(f"Chunk {start} to {end}, chunk_X size: {len(chunk_X)}, chunk_y size: {len(chunk_y)}")
        for i in range(min(end - start, len(chunk_X) - time_steps + 1)):
            Xs.append(chunk_X.iloc[i:(i + time_steps)].values)
            ys.append(chunk_y.iloc[i + time_steps - 1])
        if Xs:
            logging.info(f"Yielding chunk {start} to {end}, {len(Xs)} sequences, Memory: {psutil.virtual_memory().percent}%")
            yield np.array(Xs), np.array(ys)
        else:
            logging.warning(f"No sequences in chunk {start} to {end}")

def train_model(X, y, time_steps, features, sample_frac=1.0):
    """Train the LSTM model."""
    if sample_frac < 1.0:
        X = X.sample(frac=sample_frac, random_state=42)
        y = y.loc[X.index]
        logging.info(f"Subsampled data to {len(X)} rows")

    X_data = X[features].fillna(0)
    y_data = y

    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, shuffle=False)
    logging.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    scaler = StandardScaler()

    # Generator for training and testing data
    train_generator = create_sequences_generator(X_train, y_train, time_steps)
    test_generator = create_sequences_generator(X_test, y_test, time_steps)
    steps_per_epoch = max(1, (len(X_train) - time_steps) // 10000)
    validation_steps = max(1, (len(X_test) - time_steps) // 10000)

    # Scale a small sample to fit scaler
    sample_size = min(1000, len(X_train))
    sample_X = X_train.iloc[:sample_size][features].values.reshape(-1, len(features))
    scaler.fit(sample_X)

    # Model definition
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(time_steps, len(features))),
        BatchNormalization(),
        Dropout(0.2),
        LSTM(64),
        BatchNormalization(),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # Training with early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    history = model.fit(train_generator, steps_per_epoch=steps_per_epoch, epochs=10,
                        validation_data=test_generator, validation_steps=validation_steps,
                        callbacks=[early_stopping], verbose=1)

    return model, scaler, history, X_test, y_test, test_generator

def evaluate_model(model, X_test, y_test, test_generator, scaler):
    """Evaluate the model with detailed metrics."""
    X_test_seq, y_test_seq = [], []
    for X_chunk, y_chunk in test_generator:
        X_scaled = scaler.transform(X_chunk.reshape(-1, X_chunk.shape[2])).reshape(X_chunk.shape)
        X_test_seq.append(X_scaled)
        y_test_seq.append(y_chunk)

    if not X_test_seq:
        logging.error("No sequences generated from test data. Test set too small or time_steps too large.")
        print("Evaluation skipped due to insufficient test data.")
        return None

    X_test_seq = np.concatenate(X_test_seq)
    y_test_seq = np.concatenate(y_test_seq)
    logging.info(f"Evaluating on {len(X_test_seq)} test sequences")

    loss, accuracy = model.evaluate(X_test_seq, y_test_seq, verbose=0)
    y_pred = (model.predict(X_test_seq) > 0.5).astype(int)
    precision = precision_score(y_test_seq, y_pred)
    recall = recall_score(y_test_seq, y_pred)
    f1 = f1_score(y_test_seq, y_pred)

    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")

    # Confusion matrix
    cm = confusion_matrix(y_test_seq, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    return history

def plot_history(history):
    """Plot training history."""
    if history is None:
        logging.warning("No history to plot due to evaluation failure.")
        return
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

def predict_new_data(model, scaler, features, new_file, time_steps):
    """Predict on new data."""
    new_file = validate_file_path(new_file)
    new_data = pd.read_excel(new_file)
    _, voltage_cols_new = identify_columns(new_data)
    new_data, new_features = engineer_features(new_data, voltage_cols_new)

    # Ensure feature consistency
    missing_features = set(features) - set(new_features)
    if missing_features:
        logging.warning(f"Missing features in new data: {missing_features}")
        for f in missing_features:
            new_data[f] = 0
    new_X = new_data[features].fillna(0)

    new_generator = create_sequences_generator(new_X, pd.Series([0] * len(new_X)), time_steps)
    predictions = []
    for X_chunk, _ in new_generator:
        X_scaled = scaler.transform(X_chunk.reshape(-1, X_chunk.shape[2])).reshape(X_chunk.shape)
        pred = model.predict(X_scaled, verbose=0)
        predictions.extend(pred.flatten())
    
    if not predictions:
        logging.warning("No predictions made. New data may be too small.")
        return
    
    loosening_count = np.sum(np.array(predictions) > 0.5)
    print(f"Predicted loosening instances: {loosening_count} out of {len(predictions)}")

def save_model_and_scaler(model, scaler, features, base_path='C:/Users/Busiso/Desktop/hanna results/'):
    """Save model, scaler, and feature names."""
    try:
        model.save(f'{base_path}strain_gauge_lstm_model.keras')
        joblib.dump(scaler, f'{base_path}scaler.pkl')
        joblib.dump(features, f'{base_path}feature_names.pkl')
        logging.info("Model, scaler, and features saved successfully")
    except Exception as e:
        logging.error(f"Error saving model or scaler: {e}")

def verify_model(base_path='C:/Users/Busiso/Desktop/hanna results/'):
    """Verify saved model and scaler."""
    try:
        loaded_model = load_model(f'{base_path}strain_gauge_lstm_model.keras')
        loaded_scaler = joblib.load(f'{base_path}scaler.pkl')
        logging.info("Model and scaler loaded for verification")
    except Exception as e:
        logging.error(f"Error during verification: {e}")

def main():
    # User inputs
    successful_file = input("Enter path to successful Excel file: ")
    failed_file = input("Enter path to failed Excel file: ")
    new_file = input("Enter path to new Excel file for prediction: ")

    # Load and process data
    all_data = load_data(successful_file, failed_file)
    time_cols, voltage_cols = identify_columns(all_data)
    logging.info(f"Detected time columns: {time_cols}, voltage columns: {voltage_cols}")
    all_data, features = engineer_features(all_data, voltage_cols)

    # Train model
    model, scaler, history, X_test, y_test, test_generator = train_model(
        all_data, all_data['Label'], args.time_steps, features, args.sample_frac
    )

    # Evaluate and plot
    history = evaluate_model(model, X_test, y_test, test_generator, scaler)
    plot_history(history)

    # Predict on new data
    predict_new_data(model, scaler, features, new_file, args.time_steps)

    # Save and verify
    save_model_and_scaler(model, scaler, features)
    verify_model()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Program failed: {e}")
        sys.exit(1)