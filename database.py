import pandas as pd
import numpy as np
from datetime import datetime
import yaml
import os
import json

class Database:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_dir = self.config['paths']['data_dir']
        self.results_dir = self.config['paths']['results_dir']
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
    
    def save_experiment(self, parameters, results, metadata=None):
        """Save experiment data to Excel file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{self.data_dir}/experiment_{timestamp}.xlsx"
        
        # Create Excel writer
        with pd.ExcelWriter(filename) as writer:
            # Save parameters
            pd.DataFrame([parameters]).to_excel(writer, sheet_name='Parameters', index=False)
            
            # Save results
            if isinstance(results, pd.DataFrame):
                results.to_excel(writer, sheet_name='Results', index=False)
            else:
                pd.DataFrame([results]).to_excel(writer, sheet_name='Results', index=False)
            
            # Save metadata if provided
            if metadata:
                pd.DataFrame([metadata]).to_excel(writer, sheet_name='Metadata', index=False)
        
        return filename
    
    def save_predictions(self, model_name, predictions, actual_values, experiment_id):
        """Save model predictions to Excel file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{self.results_dir}/predictions_{model_name}_{timestamp}.xlsx"
        
        # Create Excel writer
        with pd.ExcelWriter(filename) as writer:
            # Save predictions
            if isinstance(predictions, pd.DataFrame):
                predictions.to_excel(writer, sheet_name='Predictions', index=False)
            else:
                pd.DataFrame(predictions).to_excel(writer, sheet_name='Predictions', index=False)
            
            # Save actual values
            if isinstance(actual_values, pd.DataFrame):
                actual_values.to_excel(writer, sheet_name='Actual_Values', index=False)
            else:
                pd.DataFrame(actual_values).to_excel(writer, sheet_name='Actual_Values', index=False)
            
            # Save experiment ID
            pd.DataFrame({'experiment_id': [experiment_id]}).to_excel(
                writer, sheet_name='Metadata', index=False)
        
        return filename
    
    def get_experiment(self, experiment_id):
        """Retrieve experiment data by ID from Excel file."""
        # Find the experiment file
        experiment_files = [f for f in os.listdir(self.data_dir) 
                          if f.startswith('experiment_') and f.endswith('.xlsx')]
        
        for file in experiment_files:
            try:
                # Read parameters sheet
                parameters = pd.read_excel(os.path.join(self.data_dir, file), 
                                         sheet_name='Parameters').to_dict('records')[0]
                
                # Check if this is the experiment we're looking for
                if parameters.get('experiment_id') == experiment_id:
                    # Read all sheets
                    results = pd.read_excel(os.path.join(self.data_dir, file), 
                                          sheet_name='Results')
                    metadata = pd.read_excel(os.path.join(self.data_dir, file), 
                                           sheet_name='Metadata').to_dict('records')[0] \
                        if 'Metadata' in pd.ExcelFile(os.path.join(self.data_dir, file)).sheet_names \
                        else None
                    
                    return {
                        'id': experiment_id,
                        'timestamp': datetime.strptime(file.split('_')[1].split('.')[0], 
                                                     '%Y%m%d_%H%M%S'),
                        'parameters': parameters,
                        'results': results,
                        'metadata': metadata
                    }
            except Exception as e:
                print(f"Error reading file {file}: {e}")
                continue
        
        return None
    
    def get_predictions(self, experiment_id):
        """Retrieve all predictions for an experiment from Excel files."""
        prediction_files = [f for f in os.listdir(self.results_dir) 
                          if f.startswith('predictions_') and f.endswith('.xlsx')]
        
        predictions = []
        for file in prediction_files:
            try:
                # Read metadata sheet
                metadata = pd.read_excel(os.path.join(self.results_dir, file), 
                                       sheet_name='Metadata')
                
                # Check if this prediction belongs to our experiment
                if metadata['experiment_id'].iloc[0] == experiment_id:
                    # Read predictions and actual values
                    pred_data = pd.read_excel(os.path.join(self.results_dir, file), 
                                            sheet_name='Predictions')
                    actual_data = pd.read_excel(os.path.join(self.results_dir, file), 
                                              sheet_name='Actual_Values')
                    
                    predictions.append({
                        'id': len(predictions) + 1,
                        'timestamp': datetime.strptime(file.split('_')[2].split('.')[0], 
                                                     '%Y%m%d_%H%M%S'),
                        'model': file.split('_')[1],
                        'predictions': pred_data,
                        'actual_values': actual_data,
                        'experiment_id': experiment_id
                    })
            except Exception as e:
                print(f"Error reading file {file}: {e}")
                continue
        
        return predictions
    
    def get_recent_experiments(self, limit=10):
        """Get the most recent experiments from Excel files."""
        experiment_files = [f for f in os.listdir(self.data_dir) 
                          if f.startswith('experiment_') and f.endswith('.xlsx')]
        
        # Sort files by timestamp
        experiment_files.sort(reverse=True)
        
        experiments = []
        for file in experiment_files[:limit]:
            try:
                # Read parameters sheet
                parameters = pd.read_excel(os.path.join(self.data_dir, file), 
                                         sheet_name='Parameters').to_dict('records')[0]
                
                # Read results sheet
                results = pd.read_excel(os.path.join(self.data_dir, file), 
                                      sheet_name='Results')
                
                # Read metadata if available
                metadata = pd.read_excel(os.path.join(self.data_dir, file), 
                                       sheet_name='Metadata').to_dict('records')[0] \
                    if 'Metadata' in pd.ExcelFile(os.path.join(self.data_dir, file)).sheet_names \
                    else None
                
                experiments.append({
                    'id': parameters.get('experiment_id'),
                    'timestamp': datetime.strptime(file.split('_')[1].split('.')[0], 
                                                 '%Y%m%d_%H%M%S'),
                    'parameters': parameters,
                    'results': results,
                    'metadata': metadata
                })
            except Exception as e:
                print(f"Error reading file {file}: {e}")
                continue
        
        return experiments 