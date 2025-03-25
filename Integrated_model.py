# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 07:26:29 2025

@author: Busiso
"""

import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import os
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from database import Database
import yaml
from datetime import datetime
import matplotlib.pyplot as plt

#df
class HybridModel(BaseEstimator):
    def __init__(self, strain_model, shpb_model, strain_scaler, shpb_scaler_X, shpb_scaler_y):
        self.strain_model = strain_model
        self.shpb_model = shpb_model
        self.strain_scaler = strain_scaler
        self.shpb_scaler_X = shpb_scaler_X
        self.shpb_scaler_y = shpb_scaler_y
        self.reliability_threshold = 0.5
        self.max_iterations = 3
        self.convergence_threshold = 1e-6
        
    def predict_strain_reliability(self, sequences):
        # Ensure sequences is a 3D array (samples, time_steps, features)
        if len(sequences.shape) == 1:
            sequences = sequences.reshape(-1, 1)
        if len(sequences.shape) == 2:
            sequences = sequences.reshape(sequences.shape[0], sequences.shape[1], 1)
            
        # Create a DataFrame with all required features
        n_samples = sequences.shape[0]
        n_timesteps = sequences.shape[1]
        
        # Create a DataFrame with the strain data and derived features
        strain_data = sequences.reshape(-1, 1)
        strain_df = pd.DataFrame({
            'Voltage (V) - PXI1Slot4/ai0': strain_data.flatten(),
            'Voltage (V) - PXI1Slot4/ai0_Diff': np.diff(strain_data.flatten(), prepend=strain_data[0]),
            'Voltage (V) - PXI1Slot4/ai0_Rolling_Mean': pd.Series(strain_data.flatten()).rolling(window=10).mean(),
            'Voltage (V) - PXI1Slot4/ai0_Rolling_Std': pd.Series(strain_data.flatten()).rolling(window=10).std(),
            'Voltage (V) - PXI1Slot4/ai1': strain_data.flatten(),  # Using same data for both bars
            'Voltage (V) - PXI1Slot4/ai1_Diff': np.diff(strain_data.flatten(), prepend=strain_data[0]),
            'Voltage (V) - PXI1Slot4/ai1_Rolling_Mean': pd.Series(strain_data.flatten()).rolling(window=10).mean(),
            'Voltage (V) - PXI1Slot4/ai1_Rolling_Std': pd.Series(strain_data.flatten()).rolling(window=10).std(),
            'Ratio_Voltage (V) - PXI1Slot4/ai0_Voltage (V) - PXI1Slot4/ai1': np.ones_like(strain_data.flatten()),
            'Diff_Voltage (V) - PXI1Slot4/ai0_Voltage (V) - PXI1Slot4/ai1': np.zeros_like(strain_data.flatten())
        })
        
        # Fill NaN values
        strain_df = strain_df.ffill().bfill()
        
        # Scale the features
        scaled_features = self.strain_scaler.transform(strain_df)
        
        # Reshape back to 3D for LSTM model
        scaled_sequences = scaled_features.reshape(n_samples, n_timesteps, -1)
        
        # Make predictions
        return self.strain_model.predict(scaled_sequences)
    
    def predict_shpb(self, inputs):
        # Print input columns for debugging
        print("\nInput columns:", inputs.columns.tolist())
        print("Number of input features:", len(inputs.columns))
    
        # Print scaler information
        print("\nScaler information:")
        print("Number of features expected by scaler:", self.shpb_scaler_X.n_features_in_)
    
        # Load feature names from file if available
        try:
            feature_names = joblib.load('models/feature_names.pkl')
            print("\nExpected features:", feature_names)
            print("Number of expected features:", len(feature_names))
        except Exception as e:
            print("\nWarning: Could not load feature names:", str(e))
    
        # Ensure we have exactly the features the scaler expects
        if len(inputs.columns) != self.shpb_scaler_X.n_features_in_:
            print(f"\nWarning: Input has {len(inputs.columns)} features, but scaler expects {self.shpb_scaler_X.n_features_in_}")
        
            # If we have too many features, drop the extra ones
            if len(inputs.columns) > self.shpb_scaler_X.n_features_in_:
                # Remove the Diff_Voltage feature if it exists
                if 'Diff_Voltage (V) - PXI1Slot4/ai0_Voltage (V) - PXI1Slot4/ai1' in inputs.columns:
                    inputs = inputs.drop(columns=['Diff_Voltage (V) - PXI1Slot4/ai0_Voltage (V) - PXI1Slot4/ai1'])
                    print("Removed 'Diff_Voltage' feature to match scaler expectations.")
    
        scaled_inputs = self.shpb_scaler_X.transform(inputs)
        predictions = self.shpb_model.predict(scaled_inputs)
        return self.shpb_scaler_y.inverse_transform(predictions)

    
    def refine_predictions(self, strain_data, shpb_params, initial_stress):
        # Convert initial_stress to numpy array and ensure it's 1D
        current_stress = np.array(initial_stress).flatten()
        iteration = 0
        
        # Convert current_stress to numpy array and handle dimensions
        current_stress = np.array(current_stress)
        if len(current_stress.shape) == 2:
            current_stress = current_stress[:, 0]
        elif len(current_stress.shape) == 3:
            current_stress = current_stress[:, 0, 0]
        
        # Convert static_strength to float and ensure it's a scalar
        static_strength = float(shpb_params['static_strength'])
        
        while iteration < self.max_iterations:
            # Use stress predictions to refine strain reliability
            stress_factor = np.clip(current_stress / static_strength, 0, 1)
            reliability_adjustment = 1 - stress_factor
            
            # Update strain reliability predictions
            sequences = create_sequences_chunked(strain_data, time_steps=50, stride=1)
            base_reliability = self.predict_strain_reliability(sequences)
            
            # Ensure shapes match for multiplication
            if len(reliability_adjustment.shape) == 1:
                reliability_adjustment = reliability_adjustment.reshape(-1, 1)
            if len(base_reliability.shape) == 1:
                base_reliability = base_reliability.reshape(-1, 1)
                
            # Trim reliability_adjustment to match base_reliability shape if needed
            if reliability_adjustment.shape[0] > base_reliability.shape[0]:
                reliability_adjustment = reliability_adjustment[:base_reliability.shape[0]]
            elif reliability_adjustment.shape[0] < base_reliability.shape[0]:
                base_reliability = base_reliability[:reliability_adjustment.shape[0]]
                
            adjusted_reliability = base_reliability * reliability_adjustment
            
            # Filter reliable strains
            reliable_indices = [i + 50 for i, pred in enumerate(adjusted_reliability) if pred[0] <= self.reliability_threshold]
            reliable_strain = strain_data[reliable_indices]
            
            if not reliable_strain.size:
                break
                
            # Update SHPB inputs with refined strain data
            shpb_inputs = pd.DataFrame({
                'Voltage (V) - PXI1Slot4/ai0': reliable_strain / shpb_params['k'],  # Convert strain back to voltage
                'Voltage (V) - PXI1Slot4/ai0_Diff': np.diff(reliable_strain, prepend=reliable_strain[0]) / shpb_params['k'],
                'Voltage (V) - PXI1Slot4/ai0_Rolling_Mean': pd.Series(reliable_strain / shpb_params['k']).rolling(window=10).mean(),
                'Voltage (V) - PXI1Slot4/ai0_Rolling_Std': pd.Series(reliable_strain / shpb_params['k']).rolling(window=10).std(),
                'Voltage (V) - PXI1Slot4/ai1': reliable_strain / shpb_params['k'],  # Using same voltage for both bars as placeholder
                'Voltage (V) - PXI1Slot4/ai1_Diff': np.diff(reliable_strain, prepend=reliable_strain[0]) / shpb_params['k'],
                'Voltage (V) - PXI1Slot4/ai1_Rolling_Mean': pd.Series(reliable_strain / shpb_params['k']).rolling(window=10).mean(),
                'Voltage (V) - PXI1Slot4/ai1_Rolling_Std': pd.Series(reliable_strain / shpb_params['k']).rolling(window=10).std(),
                'Ratio_Voltage (V) - PXI1Slot4/ai0_Voltage (V) - PXI1Slot4/ai1': np.ones_like(reliable_strain)  # Using 1 as placeholder
            })
            
            # Fill NaN values
            shpb_inputs = shpb_inputs.ffill().bfill()
            
            # Get new stress predictions
            new_stress = self.predict_shpb(shpb_inputs)
            
            # Convert new_stress to numpy array and handle dimensions
            new_stress = np.array(new_stress)
            if len(new_stress.shape) == 2:
                new_stress = new_stress[:, 0]
            elif len(new_stress.shape) == 3:
                new_stress = new_stress[:, 0, 0]
            
            # Check convergence
            if np.abs(np.mean(new_stress) - np.mean(current_stress)) < self.convergence_threshold:
                break
                
            current_stress = new_stress
            iteration += 1
            
        return current_stress.reshape(-1, 1), reliable_strain

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_float_input(prompt):
    while True:
        try:
            value = float(input(prompt))
            if value <= 0:
                print("Value must be positive. Please try again.")
                continue
            return value
        except ValueError:
            print("Invalid input. Please enter a numeric value (e.g., 200e9 for 200 GPa).")

def identify_columns(data): 
    time_cols = [col for col in data.columns if 'time' in col.lower()]
    voltage_cols = [col for col in data.columns if 'voltage' in col.lower()]
    return time_cols, voltage_cols

def generate_strain_features(data, cols): 
    features = []
    for col in cols:
        features.extend([
            col,
            f'{col}_diff',
            f'{col}_rolling_mean',
            f'{col}_rolling_std'
        ])
    return features

def create_sequences_chunked(data, time_steps, stride=1):
    """Create sequences from data with proper reshaping."""
    # Ensure data is 1D
    if len(data.shape) > 1:
        data = data.flatten()
        
    # Create sequences
    sequences = []
    for i in range(0, len(data) - time_steps, stride):
        sequences.append(data[i:i + time_steps])
    
    # Convert to numpy array and reshape to 3D (samples, time_steps, features)
    sequences = np.array(sequences)
    if len(sequences.shape) == 2:
        sequences = sequences.reshape(sequences.shape[0], sequences.shape[1], 1)
    
    return sequences

def process_voltage_data(data, voltage_cols):
    """Process voltage data to create required features."""
    # Create features for incident bar (ai0)
    incident_col = voltage_cols[0]
    data[f'{incident_col}_Diff'] = data[incident_col].diff()
    data[f'{incident_col}_Rolling_Mean'] = data[incident_col].rolling(window=10).mean()
    data[f'{incident_col}_Rolling_Std'] = data[incident_col].rolling(window=10).std()
    
    # Create features for transmission bar (ai1)
    transmission_col = voltage_cols[1]
    data[f'{transmission_col}_Diff'] = data[transmission_col].diff()
    data[f'{transmission_col}_Rolling_Mean'] = data[transmission_col].rolling(window=10).mean()
    data[f'{transmission_col}_Rolling_Std'] = data[transmission_col].rolling(window=10).std()
    
    # Create ratio and difference features
    data['Ratio_Voltage (V) - PXI1Slot4/ai0_Voltage (V) - PXI1Slot4/ai1'] = data[incident_col] / data[transmission_col]
    data['Diff_Voltage (V) - PXI1Slot4/ai0_Voltage (V) - PXI1Slot4/ai1'] = data[incident_col] - data[transmission_col]
    
    # Fill NaN values with forward fill then backward fill
    data = data.ffill().bfill()
    
    return data


def main():
    # Create results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Load configuration
    config = load_config()
    
    # Initialize database
    db = Database()
    
    # Check if required files exist
    required_files = {
        'data/new_data.xlsx': 'Input data file',
        'models/strain_gauge_lstm_model.keras': 'Strain gauge model',
        'models/scaler.pkl': 'Strain scaler',
        'models/shpb_digital_twin_model.pkl': 'SHPB model',
        'models/scaler_X.pkl': 'SHPB input scaler',
        'models/scaler_y.pkl': 'SHPB output scaler'
    }
    
    for file, desc in required_files.items():
        if not os.path.exists(file):
            raise FileNotFoundError(f"{desc} not found: {file}")
    
    # Load models and scalers
    strain_model = load_model('models/strain_gauge_lstm_model.keras')
    strain_scaler = joblib.load('models/scaler.pkl')
    shpb_model = joblib.load('models/shpb_digital_twin_model.pkl')
    shpb_scaler_X = joblib.load('models/scaler_X.pkl')
    shpb_scaler_y = joblib.load('models/scaler_y.pkl')
    
    # Initialize hybrid model
    hybrid_model = HybridModel(strain_model, shpb_model, strain_scaler, shpb_scaler_X, shpb_scaler_y)
    
    # Get SHPB parameters from config or user input
    shpb_params = config['models']['shpb']['parameters']
    
    # Load and process voltage data
    data = pd.read_excel('data/new_data.xlsx')
    time_cols, voltage_cols = identify_columns(data)
    
    if len(voltage_cols) < 2:
        raise ValueError("Input data must contain voltage readings from both incident and transmission bars")
    
    # Process voltage data to create required features
    processed_data = process_voltage_data(data, voltage_cols)
    
    
    # Create input features for SHPB model
    shpb_inputs_all = pd.DataFrame({
    'Voltage (V) - PXI1Slot4/ai0': processed_data[voltage_cols[0]],
    'Voltage (V) - PXI1Slot4/ai0_Diff': processed_data[f'{voltage_cols[0]}_Diff'],
    'Voltage (V) - PXI1Slot4/ai0_Rolling_Mean': processed_data[f'{voltage_cols[0]}_Rolling_Mean'],
    'Voltage (V) - PXI1Slot4/ai0_Rolling_Std': processed_data[f'{voltage_cols[0]}_Rolling_Std'],
    'Voltage (V) - PXI1Slot4/ai1': processed_data[voltage_cols[1]],
    'Voltage (V) - PXI1Slot4/ai1_Diff': processed_data[f'{voltage_cols[1]}_Diff'],
    'Voltage (V) - PXI1Slot4/ai1_Rolling_Mean': processed_data[f'{voltage_cols[1]}_Rolling_Mean'],
    'Voltage (V) - PXI1Slot4/ai1_Rolling_Std': processed_data[f'{voltage_cols[1]}_Rolling_Std'],
    'Ratio_Voltage (V) - PXI1Slot4/ai0_Voltage (V) - PXI1Slot4/ai1': processed_data['Ratio_Voltage (V) - PXI1Slot4/ai0_Voltage (V) - PXI1Slot4/ai1'],
    'Diff_Voltage (V) - PXI1Slot4/ai0_Voltage (V) - PXI1Slot4/ai1': processed_data['Diff_Voltage (V) - PXI1Slot4/ai0_Voltage (V) - PXI1Slot4/ai1']
    })
    #A version with only 9 features
    shpb_inputs = shpb_inputs_all.drop(columns=['Diff_Voltage (V) - PXI1Slot4/ai0_Voltage (V) - PXI1Slot4/ai1'])
    # Get initial stress predictions
    initial_stress = hybrid_model.predict_shpb(shpb_inputs)
    
    # Calculate strain data from voltage
    strain_data = processed_data[voltage_cols[0]].values * shpb_params['k']
    
    # Refine predictions using hybrid model
    final_stress, reliable_strain = hybrid_model.refine_predictions(strain_data, shpb_params, initial_stress)

    # Get reliable indices from the last iteration
    sequences = create_sequences_chunked(strain_data, time_steps=50, stride=1)
    base_reliability = hybrid_model.predict_strain_reliability(sequences)
    
    final_stress_array = np.asarray(final_stress, dtype=np.float64)
    static_strength = float(shpb_params['static_strength'])
    
    print(f"Shape of final_stress_array: {final_stress_array.shape}")
    print(f"Shape of base_reliability: {base_reliability.shape}")
    
    if len(final_stress_array.shape) == 2:
        final_stress_values = final_stress_array.flatten()
    
    else:
        final_stress_values = final_stress_array
        
    if len(final_stress_values) < len(base_reliability):
        padding_length = len(base_reliability) - len(final_stress_array)
        
        if len(final_stress_values) > 0:  
            padding_value = final_stress_values[-1]
            padding_array = np.full(padding_length, padding_value)
        
            final_stress_padded = np.concatenate([final_stress_values, padding_array])
        else:
            final_stress_padded = np.zeros(len(base_reliability))
    else:
        final_stress_padded = final_stress_values[:len(base_reliability)]
    
    if len(base_reliability.shape) == 2 and len(final_stress_padded.shape) == 1:
        final_stress_padded = final_stress_padded.reshape(-1, 1)    
    
    
    stress_factor = np.clip(final_stress_padded / static_strength, 0, 1)
    reliability_adjustment = 1 - stress_factor
    
    print(f"Shape of reliability_adjustment: {reliability_adjustment.shape}")
    
    min_length = min(len(base_reliability), len(reliability_adjustment))
    base_reliability_trimmed = base_reliability[:min_length]
    reliability_adjustment_trimmed = reliability_adjustment[:min_length]
    
    
    adjusted_reliability = base_reliability_trimmed * reliability_adjustment_trimmed
    reliable_indices = [i + 50 for i, pred in enumerate(adjusted_reliability) if pred[0] <= hybrid_model.reliability_threshold]
    
    print(f"Length of reliable_indices: {len(reliable_indices)}")
    print(f"Length of reliable_strain: {len(reliable_strain)}")
    
    
    # Handle final_stress array dimensions dynamically
    if len(final_stress.shape) == 1:
        stress_values = final_stress
    elif len(final_stress.shape) == 2:
        stress_values = final_stress[:, 0]
    elif len(final_stress.shape) == 3:
        stress_values = final_stress[:, 0, 0]
    else:
        raise ValueError(f"Unexpected number of dimensions in final_stress: {len(final_stress.shape)}")
    
    print(f"Length of stress_values: {len(stress_values)}")
    min_reliable_length = min(len(reliable_indices), len(reliable_strain), len(stress_values))
    reliable_indices = reliable_indices[:min_reliable_length]
    reliable_strain = reliable_strain[:min_reliable_length]
    stress_values = stress_values[:min_reliable_length]
    
    # Prepare results for storage
    results_df = pd.DataFrame({
        'Time': data[time_cols[0]].values,
        'Strain': strain_data,
        'Reliable_Strain': np.nan,
        'Stress': np.nan
    })
    
    
    
    for i, idx in enumerate(reliable_indices):
        if idx < len(results_df):
            results_df.loc[idx, 'Reliable_Strain'] = reliable_strain[i]
            results_df.loc[idx, 'Stress'] = stress_values[i]
    
    results_df.loc[reliable_indices, 'Reliable_Strain'] = reliable_strain
    results_df.loc[reliable_indices, 'Stress'] = stress_values
    
    # Save experiment data
    experiment_id = db.save_experiment(
        parameters=shpb_params,
        results=results_df,
        metadata={
            'experiment_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'model_version': '1.0',
            'reliability_threshold': hybrid_model.reliability_threshold,
            'iterations': hybrid_model.max_iterations
        }
    )
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'Time': data[time_cols[0]].values[reliable_indices],
        'Strain': reliable_strain,
        'Stress': stress_values
    })
    
    # Create results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Create stress-strain curve plot
    plt.figure(figsize=(10, 6))
    
    # Plot stress vs strain
    plt.plot(reliable_strain[:len(reliable_indices)], stress_values[:len(reliable_indices)], 'b-', label='Stress-Strain Curve')
    
    # Add labels and title
    plt.xlabel('Strain (ε)')
    plt.ylabel('Stress (Pa)')
    plt.title('Stress-Strain Curve')
    plt.grid(True)
    plt.legend()
    
    # Save the plot
    plot_path = os.path.join('results', f'stress_strain_curve_{experiment_id}.png')
    plt.savefig(plot_path)
    plt.close()
    
    # Save predictions
    db.save_predictions(
        model_name='hybrid_model',
        predictions=predictions_df,
        actual_values=results_df,
        experiment_id=experiment_id
    )
    
    # Output results
    print("\nFinal Results:")
    print(f"Number of reliable strain measurements: {len(reliable_strain)}")
    print("\nPredicted stress (Pa):")
    for i, stress in enumerate(stress_values):
        print(f"Sample {i+1}: {stress:.2e} Pa")
    
    # Calculate and display statistics
    mean_stress = np.mean(stress_values)
    std_stress = np.std(stress_values)
    print(f"\nStatistics:")
    print(f"Mean stress: {mean_stress:.2e} Pa")
    print(f"Standard deviation: {std_stress:.2e} Pa")
    
    print(f"\nResults saved to experiment ID: {experiment_id}")
    print(f"Stress-strain curve saved to: {plot_path}")

if __name__ == "__main__":
    main() 