import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np
from datetime import datetime
import yaml
import threading
import queue
import logging
from typing import Optional
from matplotlib.figure import Figure
from data_acquisition import DataAcquisition
from database import Database
from Integrated_model import HybridModel
import os
from datetime import datetime
class DigitalTwinGUI:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.data_acquisition = DataAcquisition(config_path)
        self.database = Database(config_path)
        self.hybrid_model = None  # Will be initialized when needed
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("SHPB Digital Twin")
        self.root.geometry("1200x800")
        
        # Create main container
        self.main_container = ttk.Frame(self.root, padding="10")
        self.main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create control panel
        self._create_control_panel()
        
        # Create display area
        self._create_display_area()
        
        # Initialize data queue for real-time updates
        self.data_queue = queue.Queue()
        
        # Start update thread
        self.is_running = True
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.start()
    
    def _create_control_panel(self):
   
        control_frame = ttk.LabelFrame(self.main_container, text="Control Panel", padding="5")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
    
        # Data acquisition controls
        ttk.Label(control_frame, text="Data Acquisition").grid(row=0, column=0, columnspan=2, pady=5)
        self.start_button = ttk.Button(control_frame, text="Start Acquisition", 
                                command=self._start_acquisition)
        self.start_button.grid(row=1, column=0, padx=5, pady=2)
    
        self.stop_button = ttk.Button(control_frame, text="Stop Acquisition", 
                                command=self._stop_acquisition, state=tk.DISABLED)
        self.stop_button.grid(row=1, column=1, padx=5, pady=2)
    
        # Model controls
        ttk.Label(control_frame, text="Model Controls").grid(row=2, column=0, columnspan=2, pady=5)
        self.load_model_button = ttk.Button(control_frame, text="Load Models", 
                                    command=self._load_models)
        self.load_model_button.grid(row=3, column=0, columnspan=2, padx=5, pady=2)
    
        # Add Run Hybrid Model button here
        self.run_model_button = ttk.Button(control_frame, text="Run Hybrid Model", 
                                    command=self._run_hybrid_model)
        self.run_model_button.grid(row=4, column=0, columnspan=2, padx=5, pady=2)
    
        # SHPB parameters
        ttk.Label(control_frame, text="SHPB Parameters").grid(row=5, column=0, columnspan=2, pady=5)
        self.param_entries = {}
        params = ['E_bar', 'A_bar', 'A_specimen', 'L_specimen', 'c0', 'static_strength', 'L_bar', 'k']
        for i, param in enumerate(params):
            ttk.Label(control_frame, text=param).grid(row=6+i, column=0, padx=5, pady=2)
            self.param_entries[param] = ttk.Entry(control_frame)
            self.param_entries[param].grid(row=6+i, column=1, padx=5, pady=2)

    
    def _create_display_area(self):
        """Create the display area with plots and status."""
        display_frame = ttk.LabelFrame(self.main_container, text="Display Area", padding="5")
        display_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        # Create matplotlib figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=display_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Status display
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(display_frame, textvariable=self.status_var)
        status_label.pack(fill=tk.X, pady=5)
    
    def _start_acquisition(self):
        """Start data acquisition."""
        self.data_acquisition.start_acquisition()
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_var.set("Data acquisition running...")
    
    def _stop_acquisition(self):
        """Stop data acquisition."""
        self.data_acquisition.stop_acquisition()
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_var.set("Data acquisition stopped")
    
    def _load_models(self):
        """Load the hybrid model and update status."""
        try:
            self.hybrid_model = HybridModel(
                strain_model=load_model('trained_strain_gauge_model.h5'),
                shpb_model=joblib.load('shpb_digital_twin_model.pkl'),
                strain_scaler=joblib.load('scaler.pkl'),
                shpb_scaler_X=joblib.load('scaler_X.pkl'),
                shpb_scaler_y=joblib.load('scaler_y.pkl')
            )
            self.status_var.set("Models loaded successfully")
            messagebox.showinfo("Success", "Models loaded successfully")
        except Exception as e:
            self.status_var.set(f"Error loading models: {str(e)}")
            messagebox.showerror("Error", f"Failed to load models: {str(e)}")
    
    def _update_loop(self):
        """Main update loop for real-time display."""
        while self.is_running:
            try:
                # Get latest data
                data = self.data_acquisition.get_latest_data()
                if data is not None:
                    self._update_plots(data)
                
                # Sleep to maintain update rate
                self.root.after(100)  # Update every 100ms
                
            except Exception as e:
                logging.error(f"Error in update loop: {e}")
                self.root.after(1000)  # Wait longer on error
    
    def _update_plots(self, data: pd.DataFrame):
        """Update the plots with new data."""
        try:
            # Clear previous plots
            self.ax1.clear()
            self.ax2.clear()
            
            # Plot strain vs time
            self.ax1.plot(data['timestamp'], data['strain_gauge_1'], label='Strain Gauge 1')
            self.ax1.set_title('Strain vs Time')
            self.ax1.set_xlabel('Time')
            self.ax1.set_ylabel('Strain')
            self.ax1.grid(True)
            self.ax1.legend()
            
            # Plot stress vs strain if model is loaded
            if self.hybrid_model is not None:
                # Get predictions
                strain_data = data['strain_gauge_1'].values
                shpb_params = {k: float(v.get()) for k, v in self.param_entries.items()}
                
                # Make predictions
                stress, reliable_strain = self.hybrid_model.refine_predictions(
                    strain_data, shpb_params, None)
                
                self.ax2.plot(reliable_strain, stress[:, 0], label='Stress-Strain')
                self.ax2.set_title('Stress vs Strain')
                self.ax2.set_xlabel('Strain')
                self.ax2.set_ylabel('Stress (Pa)')
                self.ax2.grid(True)
                self.ax2.legend()
            
            # Update canvas
            self.canvas.draw()
            
        except Exception as e:
            logging.error(f"Error updating plots: {e}")
    def _run_hybrid_model(self):
    
        try:
            # Get data from data acquisition
            time_data = self.data_acquisition.time_buffer
            voltage_data = self.data_acquisition.data_buffer
        
            # Create a DataFrame from the acquired data
            data = pd.DataFrame()
        
            # Add time column
            data['Time'] = time_data
        
            # Add voltage columns
            for channel_name, values in voltage_data.items():
                data[f'Voltage (V) - {channel_name}'] = values
        
            # Import the hybrid model
            from Integrated_model import main as run_hybrid_model
        
            # Run the model with the acquired data
            results_df, predictions_df, experiment_id = run_hybrid_model(input_data=data)
        
            # Update display with results
            self._update_results_display(results_df, predictions_df)
        
            # Show success message
            messagebox.showinfo("Model Run Complete", 
                        f"Hybrid model analysis complete.\nExperiment ID: {experiment_id}")
        
        except Exception as e:
            # Show error message
            messagebox.showerror("Model Error", f"Error running hybrid model: {str(e)}")
            # Log the error
            logging.error(f"Error running hybrid model: {str(e)}", exc_info=True)
    def _update_results_display(self, results_df, predictions_df):
        # Clear previous plots
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
    
        # Create a figure for the plot
        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
    
        # Plot stress-strain curve
        ax.plot(predictions_df['Strain'], predictions_df['Stress'], 'b-', label='Stress-Strain Curve')
    
        # Add labels and title
        ax.set_xlabel('Strain (ε)')
        ax.set_ylabel('Stress (Pa)')
        ax.set_title('Stress-Strain Curve')
        ax.grid(True)
        ax.legend()
    
        # Create a canvas to display the plot
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
        # Add toolbar
        toolbar = NavigationToolbar2Tk(canvas, self.plot_frame)
        toolbar.update()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def run(self):
        """Start the GUI main loop."""
        self.root.mainloop()
        self.is_running = False
        self.update_thread.join()
    

if __name__ == "__main__":
    gui = DigitalTwinGUI()
    gui.run() 