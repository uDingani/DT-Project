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
    def __init__(self, config_path=None):
        if config_path is None:
            # Try to find the config file in various locations
            possible_paths = [
                'config.yaml',                      # Current directory
                '../config.yaml',                   # Parent directory
                os.path.join(os.path.dirname(__file__), 'config.yaml'),  # Same directory as script
                os.path.join(os.path.dirname(__file__), '../config.yaml')  # Parent of script directory
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break
        
        if config_path is None:
            # If config file not found, create a default one
            config_path = 'config.yaml'
            default_config = {
                'data_acquisition': {
                    'sampling_rate': 1000,
                    'buffer_size': 10000,
                    'channels': [
                        {'name': 'PXI1Slot4/ai0', 'description': 'Incident Bar Strain Gauge', 'calibration_factor': 1.0},
                        {'name': 'PXI1Slot4/ai1', 'description': 'Transmission Bar Strain Gauge', 'calibration_factor': 1.0}
                    ]
                },
                'database': {
                    'type': 'file',
                    'path': 'data/experiments'
                },
                'models': {
                    'shpb': {
                        'parameters': {
                            'E_bar': 200e9,
                            'A_bar': 0.0005,
                            'A_specimen': 0.0001,
                            'L_specimen': 0.01,
                            'c0': 5000,
                            'static_strength': 500e6,
                            'L_bar': 2.0,
                            'k': 2.11
                        }
                    }
                },
                'paths': {
                    'data_dir': 'data',
                    'models_dir': 'models',
                    'logs_dir': 'logs',
                    'results_dir': 'results'
                }
            }
            
            # Create directories if they don't exist
            for dir_path in ['data', 'models', 'logs', 'results']:
                os.makedirs(dir_path, exist_ok=True)
            
            # Write default config
            with open(config_path, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False)
            
            print(f"Created default config file at {config_path}")
    
   
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
        
        self._apply_custom_style()
        
        
        self.main_container = ttk.Frame(self.root, padding="15", style="TFrame")
        self.main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.main_container.columnconfigure(0, weight=1)
        self.main_container.columnconfigure(1, weight=3)  # Display area gets more space
        self.main_container.rowconfigure(0, weight=1)


        
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
    
    def _apply_custom_style(self):
        """Apply a custom purple and grey theme to the GUI with enhanced styling."""
        style = ttk.Style()
    
        # Define colors
        primary_color = "#6a1b9a"  # Deep purple
        secondary_color = "#9c27b0"  # Purple
        accent_color = "#d1c4e9"  # Light purple
        text_color = "#212121"  # Dark text (almost black)
        light_text = "#f5f5f5"  # Light text
        bg_color = "#424242"  # Dark grey
        frame_bg = "#616161"  # Medium grey
        highlight_color = "#ce93d8"  # Light purple highlight
    
        # Configure the theme
        style.configure("TFrame", background=frame_bg)
    
        # Create fancy labelframe style with boxed headings
        style.configure("TLabelframe", background=frame_bg, borderwidth=2, relief="groove")
        style.configure("TLabelframe.Label", 
                foreground=light_text, 
                background=primary_color, 
                font=('Arial', 10, 'bold'),
                padding=(10, 5),
                relief="raised",
                borderwidth=1)
    
        # Regular labels
        style.configure("TLabel", foreground=light_text, background=frame_bg, font=('Arial', 9))
    
        # Section heading style
        style.configure("Heading.TLabel", 
                foreground=light_text, 
                background=secondary_color, 
                font=('Arial', 10, 'bold'),
                padding=(5, 3),
                relief="raised",
                borderwidth=1)
    
        # Button styles with black text and hover effects
        style.configure("TButton", 
                foreground=text_color, 
                background=accent_color, 
                font=('Arial', 9, 'bold'),
                relief="raised",
                borderwidth=2,
                padding=(10, 5))
    
        # Map hover and pressed states
        style.map("TButton",
            foreground=[('pressed', text_color), ('active', text_color)],
            background=[('pressed', "#b39ddb"), ('active', highlight_color)],
            relief=[('pressed', 'sunken'), ('active', 'raised')],
            borderwidth=[('active', 3)])
    
        # Entry fields
        style.configure("TEntry", 
                foreground=text_color, 
                fieldbackground=accent_color, 
                font=('Arial', 9),
                borderwidth=2,
                relief="sunken")
    
        # Configure the main window
        self.root.configure(background=bg_color)
    
        # Create a custom style for the control panel
        style.configure("Control.TLabelframe", 
                background=frame_bg, 
                padding=10,
                borderwidth=3,
                relief="ridge")
    
        style.configure("Control.TLabelframe.Label", 
                foreground=light_text, 
                background=primary_color, 
                font=('Arial', 11, 'bold'),
                padding=(15, 5),
                relief="raised")
    
        
        style.configure("Display.TLabelframe", 
                background=frame_bg, 
                padding=10,
                borderwidth=3,
                relief="ridge")
    
        style.configure("Display.TLabelframe.Label", 
                foreground=light_text, 
                background=primary_color, 
                font=('Arial', 11, 'bold'),
                padding=(15, 5),
                relief="raised")

    
        style.configure("Start.TButton", 
                foreground=text_color, 
                background="#a5d6a7",  # Light green
                font=('Arial', 9, 'bold'),
                padding=(10, 5),
                relief="raised",
                borderwidth=2)
    
        style.map("Start.TButton",
                foreground=[('pressed', text_color), ('active', text_color)],
                background=[('pressed', "#81c784"), ('active', "#c8e6c9")],
                relief=[('pressed', 'sunken'), ('active', 'raised')],
                borderwidth=[('active', 3)])
    
    
        style.configure("Stop.TButton", 
                foreground=text_color, 
                background="#ef9a9a",  # Light red
                font=('Arial', 9, 'bold'),
                padding=(10, 5),
                relief="raised",
                borderwidth=2)
    
        style.map("Stop.TButton",
                foreground=[('pressed', text_color), ('active', text_color)],
                background=[('pressed', "#e57373"), ('active', "#ffcdd2")],
                relief=[('pressed', 'sunken'), ('active', 'raised')],
                borderwidth=[('active', 3)])
    
    
        style.configure("Model.TButton", 
                foreground=text_color, 
                background="#ce93d8",  # Light purple
                font=('Arial', 9, 'bold'),
                padding=(10, 5),
                relief="raised",
                borderwidth=2)
    
        style.map("Model.TButton",
                foreground=[('pressed', text_color), ('active', text_color)],
                background=[('pressed', "#ba68c8"), ('active', "#e1bee7")],
                relief=[('pressed', 'sunken'), ('active', 'raised')],
                borderwidth=[('active', 3)])

    def _create_control_panel(self):
        """Create the control panel with buttons and inputs."""
        control_frame = ttk.LabelFrame(self.main_container, text="Control Panel", style="Control.TLabelframe")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=12, pady=12)
    
    
        inner_frame = ttk.Frame(control_frame, style="TFrame", padding=(10, 10, 10, 10))
        inner_frame.pack(fill=tk.BOTH, expand=True)
    
    
        section_frame1 = ttk.Frame(inner_frame, style="TFrame")
        section_frame1.pack(fill=tk.X, pady=(0, 15))
    
        heading1 = ttk.Label(section_frame1, text="Data Acquisition", style="Heading.TLabel")
        heading1.pack(fill=tk.X, pady=(0, 8))
    
        button_frame1 = ttk.Frame(section_frame1, style="TFrame")
        button_frame1.pack(fill=tk.X)
    
        self.start_button = ttk.Button(button_frame1, text="Start Acquisition", 
                                command=self._start_acquisition, style="Start.TButton")
        self.start_button.pack(side=tk.LEFT, padx=5, pady=5, expand=True, fill=tk.X)
    
        self.stop_button = ttk.Button(button_frame1, text="Stop Acquisition", 
                                command=self._stop_acquisition, state=tk.DISABLED, style="Stop.TButton")
        self.stop_button.pack(side=tk.LEFT, padx=5, pady=5, expand=True, fill=tk.X)
    
 
        section_frame2 = ttk.Frame(inner_frame, style="TFrame")
        section_frame2.pack(fill=tk.X, pady=(0, 15))
    
        heading2 = ttk.Label(section_frame2, text="Model Controls", style="Heading.TLabel")
        heading2.pack(fill=tk.X, pady=(0, 8))
    
        self.load_model_button = ttk.Button(section_frame2, text="Load Models", 
                                    command=self._load_models, style="Model.TButton")
        self.load_model_button.pack(fill=tk.X, padx=5, pady=5)
    
    
        self.run_model_button = ttk.Button(section_frame2, text="Run Hybrid Model", 
                                    command=self._run_hybrid_model, style="Model.TButton")
        self.run_model_button.pack(fill=tk.X, padx=5, pady=5)
    
    
        section_frame3 = ttk.Frame(inner_frame, style="TFrame")
        section_frame3.pack(fill=tk.X)
    
        heading3 = ttk.Label(section_frame3, text="SHPB Parameters", style="Heading.TLabel")
        heading3.pack(fill=tk.X, pady=(0, 8))
    
        params_frame = ttk.Frame(section_frame3, style="TFrame")
        params_frame.pack(fill=tk.X)
    
        self.param_entries = {}
        params = ['E_bar', 'A_bar', 'A_specimen', 'L_specimen', 'c0', 'static_strength', 'L_bar', 'k']
    
        for i, param in enumerate(params):
            param_frame = ttk.Frame(params_frame, style="TFrame")
            param_frame.pack(fill=tk.X, pady=3)

            ttk.Label(param_frame, text=param).pack(side=tk.LEFT, padx=(5, 10), pady=2)
            self.param_entries[param] = ttk.Entry(param_frame)
            self.param_entries[param].pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5, pady=2)
        for i, param in enumerate(params):
            param_frame = ttk.Frame(params_frame, style="TFrame")
            param_frame.pack(fill=tk.X, pady=3)
        
            ttk.Label(param_frame, text=param).pack(side=tk.LEFT, padx=(5, 10), pady=2)
            self.param_entries[param] = ttk.Entry(param_frame)
            self.param_entries[param].pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5, pady=2)

    def _create_display_area(self):
        """Create the display area with plots and status."""
        display_frame = ttk.LabelFrame(self.main_container, text="Display Area", style="Display.TLabelframe")
        display_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=12, pady=12)
    
    
        inner_frame = ttk.Frame(display_frame, style="TFrame", padding=(10, 10, 10, 10))
        inner_frame.pack(fill=tk.BOTH, expand=True)
    
   
        plt.style.use('dark_background') 
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 8), facecolor='#424242')
    
    
        for ax in [self.ax1, self.ax2]:
            ax.set_facecolor('#616161')  # Set plot background
            ax.tick_params(colors='#f5f5f5')  # Set tick colors
            ax.xaxis.label.set_color('#f5f5f5')  # Set x-axis label color
            ax.yaxis.label.set_color('#f5f5f5')  # Set y-axis label color
            ax.title.set_color('#f5f5f5')  # Set title color
            ax.spines['bottom'].set_color('#9c27b0')  # Set axis color
            ax.spines['top'].set_color('#9c27b0')
            ax.spines['left'].set_color('#9c27b0')
            ax.spines['right'].set_color('#9c27b0')
            ax.grid(True, linestyle='--', alpha=0.7, color='#9e9e9e')
    
    
        plot_frame = ttk.Frame(inner_frame, style="TFrame", borderwidth=2, relief="groove")
        plot_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
    
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    
        status_frame = ttk.Frame(inner_frame, style="TFrame", borderwidth=2, relief="groove")
        status_frame.pack(fill=tk.X, pady=5)
    
        status_label_frame = ttk.Frame(status_frame, style="TFrame", padding=(5, 5, 5, 5))
        status_label_frame.pack(fill=tk.X)
    
        ttk.Label(status_label_frame, text="Status:", 
            font=('Arial', 9, 'bold'), 
            foreground="#f5f5f5").pack(side=tk.LEFT, padx=5)
    
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(status_label_frame, 
                            textvariable=self.status_var, 
                            font=('Arial', 9),
                            foreground="#f5f5f5")
        status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
   
    

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