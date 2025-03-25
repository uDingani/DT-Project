import os
import shutil
import sys

def create_directory_structure():
    """Create the required directory structure."""
    directories = ['data', 'models', 'results', 'src']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def organize_files():
    """Organize files into their proper locations."""
    # Define file mappings
    file_moves = {
        # Core files stay in root
        'Integrated model': './',
        'config.yaml': './',
        'requirements.txt': './',
        
        # Source files go to src
        'database.py': 'src/',
        'gui.py': 'src/',
        'data_acquisition.py': 'src/',
        
        # Model files
        'LSTM Model creator.py': 'src/model_training/',
        'SHPB data and fit an empirical DIF.py': 'src/model_training/',
        'Spring strain model trainer and evaluater.py': 'src/model_training/',
        'Loading trained LSTM Model.py': 'src/model_training/'
    }
    
    # Create model training directory
    os.makedirs('src/model_training', exist_ok=True)
    
    # Move files to their destinations
    for source, dest in file_moves.items():
        if os.path.exists(source):
            if dest != './':  # Only move if not staying in root
                os.makedirs(dest, exist_ok=True)
                shutil.move(source, os.path.join(dest, source))
                print(f"Moved {source} to {dest}")
        else:
            print(f"Warning: {source} not found")

def main():
    print("Organizing Digital Twin System files...")
    
    # Create directory structure
    create_directory_structure()
    
    # Organize files
    organize_files()
    
    print("\nFile organization complete!")
    print("\nDirectory structure:")
    print("DT-Project/")
    print("├── data/                  # For input data files")
    print("├── models/               # For trained models")
    print("├── results/              # For output results")
    print("├── src/                  # Source code")
    print("│   ├── database.py")
    print("│   ├── gui.py")
    print("│   ├── data_acquisition.py")
    print("│   └── model_training/   # Model training scripts")
    print("├── Integrated model      # Main script")
    print("├── config.yaml           # Configuration")
    print("└── requirements.txt      # Dependencies")
    
    print("\nNext steps:")
    print("1. Place your input data file (new_data.xlsx) in the data/ directory")
    print("2. Place your trained models and scalers in the models/ directory")
    print("3. Run: python \"Integrated model\"")

if __name__ == "__main__":
    main() 