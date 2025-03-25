# Fix for the missing feature in predict_shpb method
def predict_shpb(self, inputs):
    # Add the missing feature
    if 'Diff_Voltage (V) - PXI1Slot4/ai0_Voltage (V) - PXI1Slot4/ai1' not in inputs.columns:
        # Calculate the difference between the two voltage columns
        inputs['Diff_Voltage (V) - PXI1Slot4/ai0_Voltage (V) - PXI1Slot4/ai1'] = (
            inputs['Voltage (V) - PXI1Slot4/ai0'] - inputs['Voltage (V) - PXI1Slot4/ai1']
        )
    
    # Print input columns for debugging
    print("\nInput columns:", inputs.columns.tolist())
    print("Number of input features:", len(inputs.columns))
    
    # Rest of the method remains the same...
    scaled_inputs = self.shpb_scaler_X.transform(inputs)
    predictions = self.shpb_model.predict(scaled_inputs)
    return self.shpb_scaler_y.inverse_transform(predictions)

# Fix for the type error in main function
def main():
    # ... existing code ...
    
    # Fix the type error when calculating stress_factor
    # Ensure final_stress is a numpy array of float64
    final_stress_array = np.asarray(final_stress, dtype=np.float64)
    
    # Ensure static_strength is a float
    static_strength = float(shpb_params['static_strength'])
    
    # Calculate stress_factor with proper type conversion
    stress_factor = np.clip(final_stress_array / static_strength, 0, 1)
    
    # ... rest of the function ...

# Also fix the fillna deprecation warning
def refine_predictions(self, strain_data, shpb_params, initial_stress):
    # Convert initial_stress to numpy array and ensure it's 1D
    current_stress = np.array(initial_stress, dtype=np.float64).flatten()
    iteration = 0
    
    # Convert static_strength to float and ensure it's a scalar
    static_strength = float(shpb_params['static_strength'])
    
    # Update the deprecated fillna method
    strain_df = strain_df.ffill().bfill()  # Instead of fillna(method='ffill')
    
    # ... rest of the method ...

# Fix in process_voltage_data function
def process_voltage_data(data, voltage_cols):
    # ... existing code ...
    
    # Add the missing feature
    data['Diff_Voltage (V) - PXI1Slot4/ai0_Voltage (V) - PXI1Slot4/ai1'] = (
        data[voltage_cols[0]] - data[voltage_cols[1]]
    )
    
    # Update the deprecated fillna method
    data = data.ffill().bfill()  # Instead of ffill().bfill()
    
    return data

# Fix in the shpb_inputs creation in main
def main():
    # ... existing code ...
    
    # Ensure static_strength is a float
    shpb_params['static_strength'] = float(shpb_params['static_strength'])
    
    # ... existing code ...
    
    # Create input features for SHPB model with the missing feature
    shpb_inputs = pd.DataFrame({
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
