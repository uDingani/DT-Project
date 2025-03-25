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
        scaled_sequences = self.strain_scaler.transform(sequences.reshape(-1, sequences.shape[2])).reshape(sequences.shape)
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
            
            # Check for missing features
            missing_features = set(feature_names) - set(inputs.columns)
            if missing_features:
                print("\nMissing features:", missing_features)
                
            # Check for extra features
            extra_features = set(inputs.columns) - set(feature_names)
            if extra_features:
                print("\nExtra features:", extra_features)
        except Exception as e:
            print("\nWarning: Could not load feature names:", str(e))
        
        scaled_inputs = self.shpb_scaler_X.transform(inputs)
        predictions = self.shpb_model.predict(scaled_inputs)
        return self.shpb_scaler_y.inverse_transform(predictions)
    
    def refine_predictions(self, strain_data, shpb_params, initial_stress):
        current_stress = initial_stress
        iteration = 0
        
        while iteration < self.max_iterations:
            # Use stress predictions to refine strain reliability
            stress_factor = np.clip(current_stress / shpb_params['static_strength'], 0, 1)
            reliability_adjustment = 1 - stress_factor
            
            # Update strain reliability predictions
            sequences = create_sequences_chunked(strain_data, time_steps=50, stride=1)
            base_reliability = self.predict_strain_reliability(sequences)
            adjusted_reliability = base_reliability * reliability_adjustment
            
            # Filter reliable strains
            reliable_indices = [i + 50 for i, pred in enumerate(adjusted_reliability) if pred[0] <= self.reliability_threshold]
            reliable_strain = strain_data[reliable_indices]
            
            if not reliable_strain.size:
                break
                
            # Update SHPB inputs with refined strain data
            shpb_inputs = pd.DataFrame({
                'E_bar': [shpb_params['E_bar']] * len(reliable_strain),
                'A_bar': [shpb_params['A_bar']] * len(reliable_strain),
                'A_specimen': [shpb_params['A_specimen']] * len(reliable_strain),
                'L_specimen': [shpb_params['L_specimen']] * len(reliable_strain),
                'c0': [shpb_params['c0']] * len(reliable_strain),
                'static_strength': [shpb_params['static_strength']] * len(reliable_strain),
                'L_bar': [shpb_params['L_bar']] * len(reliable_strain),
                'eps_t': reliable_strain
            })
            
            # Get new stress predictions
            new_stress = self.predict_shpb(shpb_inputs)
            
            # Check convergence
            if np.abs(np.mean(new_stress) - np.mean(current_stress)) < self.convergence_threshold:
                break
                
            current_stress = new_stress
            iteration += 1
            
        return current_stress, reliable_strain

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
    return np.array([data[i:i+time_steps] for i in range(0, len(data)-time_steps, stride)])

def main():
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
    
    # Load and process strain gauge data
    data = pd.read_excel('data/new_data.xlsx')
    time_cols, voltage_cols = identify_columns(data)
    features = generate_strain_features(data, voltage_cols)
    strain_data = data[voltage_cols[0]].values * shpb_params['k']
    
    # Initial SHPB prediction
    initial_shpb_inputs = pd.DataFrame({
        'E_bar': [shpb_params['E_bar']] * len(strain_data),
        'A_bar': [shpb_params['A_bar']] * len(strain_data),
        'A_specimen': [shpb_params['A_specimen']] * len(strain_data),
        'L_specimen': [shpb_params['L_specimen']] * len(strain_data),
        'c0': [shpb_params['c0']] * len(strain_data),
        'static_strength': [shpb_params['static_strength']] * len(strain_data),
        'L_bar': [shpb_params['L_bar']] * len(strain_data),
        'eps_t': strain_data
    })
    initial_stress = hybrid_model.predict_shpb(initial_shpb_inputs)
    
    # Refine predictions using hybrid model
    final_stress, reliable_strain = hybrid_model.refine_predictions(strain_data, shpb_params, initial_stress)
    
    # Get reliable indices from the last iteration
    sequences = create_sequences_chunked(strain_data, time_steps=50, stride=1)
    base_reliability = hybrid_model.predict_strain_reliability(sequences)
    stress_factor = np.clip(final_stress / shpb_params['static_strength'], 0, 1)
    reliability_adjustment = 1 - stress_factor
    adjusted_reliability = base_reliability * reliability_adjustment
    reliable_indices = [i + 50 for i, pred in enumerate(adjusted_reliability) if pred[0] <= hybrid_model.reliability_threshold]
    
    # Prepare results for storage
    results_df = pd.DataFrame({
        'Time': data[time_cols[0]].values,
        'Strain': strain_data,
        'Reliable_Strain': np.nan,
        'Stress': np.nan
    })
    results_df.loc[reliable_indices, 'Reliable_Strain'] = reliable_strain
    results_df.loc[reliable_indices, 'Stress'] = final_stress[:, 0]
    
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
        'Stress': final_stress[:, 0]
    })
    
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
    for i, stress in enumerate(final_stress[:, 0]):
        print(f"Sample {i+1}: {stress:.2e} Pa")
    
    # Calculate and display statistics
    mean_stress = np.mean(final_stress[:, 0])
    std_stress = np.std(final_stress[:, 0])
    print(f"\nStatistics:")
    print(f"Mean stress: {mean_stress:.2e} Pa")
    print(f"Standard deviation: {std_stress:.2e} Pa")
    
    print(f"\nResults saved to experiment ID: {experiment_id}")

if __name__ == "__main__":
    main() 