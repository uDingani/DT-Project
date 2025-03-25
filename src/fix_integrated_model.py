# In the refine_predictions method, add type conversion:
def refine_predictions(self, strain_data, shpb_params, initial_stress):
    # Convert initial_stress to numpy array and ensure it's 1D
    current_stress = np.array(initial_stress, dtype=np.float64).flatten()
    iteration = 0
    
    # Convert static_strength to float and ensure it's a scalar
    static_strength = float(shpb_params['static_strength'])
    
    # Rest of the method...

# In the main function, add explicit type conversion:
def main():
    # ... existing code ...
    
    # Ensure static_strength is a float
    shpb_params['static_strength'] = float(shpb_params['static_strength'])
    
    # ... existing code ...
    
    # When calculating stress_factor, ensure both operands are float arrays
    final_stress_float = np.array(final_stress, dtype=np.float64)
    static_strength_float = float(shpb_params['static_strength'])
    stress_factor = np.clip(final_stress_float / static_strength_float, 0, 1)
    
    # ... rest of the function ...
