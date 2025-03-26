import numpy as np
import pandas as pd
import yaml
import time
from datetime import datetime
import logging
from typing import List, Dict, Optional
import threading
from queue import Queue
import os

class DataAcquisition:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.sampling_rate = self.config['data_acquisition']['sampling_rate']
        self.buffer_size = self.config['data_acquisition']['buffer_size']
        self.channels = self.config['data_acquisition']['channels']
        
        # Initialize data buffer
        self.data_buffer = {channel['name']: [] for channel in self.channels}
        self.time_buffer = []
        
        # Initialize thread control
        self.is_running = False
        self.data_queue = Queue()
        
        # Create logs directory if it doesn't exist
        logs_dir = self.config['paths']['logs_dir']
        if not os.path.isabs(logs_dir):
            logs_dir = os.path.join(os.path.dirname(__file__), logs_dir)
        os.makedirs(logs_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename=os.path.join(logs_dir, 'data_acquisition.log')
        )
    
    def start_acquisition(self):
        """Start data acquisition in a separate thread."""
        if self.is_running:
            logging.warning("Data acquisition already running")
            return
        
        self.is_running = True
        self.acquisition_thread = threading.Thread(target=self._acquisition_loop)
        self.acquisition_thread.start()
        logging.info("Data acquisition started")
    
    def stop_acquisition(self):
        """Stop data acquisition."""
        self.is_running = False
        if hasattr(self, 'acquisition_thread'):
            self.acquisition_thread.join()
        logging.info("Data acquisition stopped")
    
    def _acquisition_loop(self):
        """Main acquisition loop."""
        while self.is_running:
            try:
                # Simulate data acquisition (replace with actual hardware interface)
                timestamp = datetime.now()
                data = self._read_sensors()
                
                # Add to buffer
                self.time_buffer.append(timestamp)
                for channel, value in data.items():
                    self.data_buffer[channel].append(value)
                
                # Check if buffer is full
                if len(self.time_buffer) >= self.buffer_size:
                    self._process_buffer()
                
                # Sleep to maintain sampling rate
                time.sleep(1 / self.sampling_rate)
                
            except Exception as e:
                logging.error(f"Error in acquisition loop: {e}")
                time.sleep(1)  # Wait before retrying
    
    def _read_sensors(self) -> Dict[str, float]:
        """Read data from sensors (simulated for now)."""
        # Replace this with actual hardware interface
        return {
            channel['name']: np.random.normal(0, 1)  # Simulated voltage readings
            for channel in self.channels
        }
    
    def _process_buffer(self):
        """Process and save the data buffer."""
        try:
            # Create DataFrame from buffer
            data_dict = {
                'timestamp': self.time_buffer,
                **{channel: self.data_buffer[channel] for channel in self.channels}
            }
            df = pd.DataFrame(data_dict)
            
            # Save to file
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.config['paths']['data_dir']}/data_{timestamp_str}.csv"
            df.to_csv(filename, index=False)
            
            # Clear buffer
            self.time_buffer = []
            for channel in self.channels:
                self.data_buffer[channel['name']] = []
            
            logging.info(f"Data saved to {filename}")
            
        except Exception as e:
            logging.error(f"Error processing buffer: {e}")
    
    def get_latest_data(self, n_samples: int = 100) -> Optional[pd.DataFrame]:
        """Get the latest n samples of data."""
        if not self.time_buffer:
            return None
        
        data_dict = {
            'timestamp': self.time_buffer[-n_samples:],
            **{channel: self.data_buffer[channel][-n_samples:] 
               for channel in self.channels}
        }
        return pd.DataFrame(data_dict)
    
    def get_channel_data(self, channel_name: str, 
                        start_time: Optional[datetime] = None,
                        end_time: Optional[datetime] = None) -> Optional[pd.Series]:
        """Get data for a specific channel within a time range."""
        if channel_name not in self.data_buffer:
            logging.error(f"Channel {channel_name} not found")
            return None
        
        # Convert timestamps to indices if time range is specified
        if start_time and end_time:
            start_idx = next((i for i, t in enumerate(self.time_buffer) 
                            if t >= start_time), 0)
            end_idx = next((i for i, t in enumerate(self.time_buffer) 
                          if t > end_time), len(self.time_buffer))
            return pd.Series(self.data_buffer[channel_name][start_idx:end_idx],
                           index=self.time_buffer[start_idx:end_idx])
        
        return pd.Series(self.data_buffer[channel_name],
                        index=self.time_buffer)
    
    def calibrate(self, channel_name: str, reference_value: float):
        """Calibrate a specific channel using a reference value."""
        if channel_name not in self.data_buffer:
            logging.error(f"Channel {channel_name} not found")
            return False
        
        # Implement calibration logic here
        # This is a placeholder for actual calibration code
        logging.info(f"Calibrating channel {channel_name} with reference value {reference_value}")
        return True 