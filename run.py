import os
import sys

# Add src directory to Python path
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
sys.path.append(src_dir)

# Import and run the main script
from Integrated_model import main

if __name__ == "__main__":
    main() 