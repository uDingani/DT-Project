import os
import shutil
import sys

def create_directory_structure():
    """Create the required directory structure."""
    directories = ['data', 'models', 'results']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def verify_required_files():
    """Verify that all required files exist in their correct locations."""
    required_files = {
        'data/new_data.xlsx': 'Input data file',
        'models/strain_gauge_lstm_model.keras': 'Strain gauge model',
        'models/scaler.pkl': 'Strain scaler',
        'models/shpb_digital_twin_model.pkl': 'SHPB model',
        'models/scaler_X.pkl': 'SHPB input scaler',
        'models/scaler_y.pkl': 'SHPB output scaler',
        'Integrated model': 'Main script',
        'database.py': 'Database module',
        'config.yaml': 'Configuration file'
    }
    
    missing_files = []
    for file_path, description in required_files.items():
        if not os.path.exists(file_path):
            missing_files.append(f"{description} ({file_path})")
    
    if missing_files:
        print("\nMissing required files:")
        for file in missing_files:
            print(f"- {file}")
        return False
    return True

def main():
    print("Setting up Digital Twin System...")
    
    # Create directory structure
    create_directory_structure()
    
    # Verify required files
    if not verify_required_files():
        print("\nPlease ensure all required files are in place before running the system.")
        print("Required files should be organized as follows:")
        print("\nDT-Project/")
        print("├── data/")
        print("│   └── new_data.xlsx")
        print("├── models/")
        print("│   ├── strain_gauge_lstm_model.keras")
        print("│   ├── scaler.pkl")
        print("│   ├── shpb_digital_twin_model.pkl")
        print("│   ├── scaler_X.pkl")
        print("│   └── scaler_y.pkl")
        print("├── results/")
        print("├── Integrated model")
        print("├── database.py")
        print("└── config.yaml")
        sys.exit(1)
    
    print("\nSystem setup complete! All required files and directories are in place.")
    print("\nTo run the system:")
    print("1. Ensure your input data is in data/new_data.xlsx")
    print("2. Run: python \"Integrated model\"")
    print("3. Results will be saved in the results/ directory")

if __name__ == "__main__":
    main() 