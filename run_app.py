"""
run_app.py - Main entry point for the Cryptocurrency Prediction AI Bot
Run with: python run_app.py
"""

import os
import sys
import subprocess
from pathlib import Path

# Get the project root directory (where this file is located)
project_root = Path(__file__).parent.absolute()

# Add project root to Python path so imports work correctly
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Change to project root directory to ensure relative paths work
os.chdir(project_root)

# Run the streamlit app
print(f"Starting Streamlit app from: {project_root}")
print("=" * 50)
print("Cryptocurrency Prediction AI Bot")
print("=" * 50)
print("\nThe app will open in your browser at http://localhost:8501")
print("Press Ctrl+C to stop the server\n")

try:
    # Use subprocess instead of os.system for better control
    subprocess.run([
        sys.executable, 
        "-m", 
        "streamlit", 
        "run", 
        "src/app/streamlit_app.py",
        "--server.headless", "false"
    ], cwd=project_root)
except KeyboardInterrupt:
    print("\n\nShutting down...")
except Exception as e:
    print(f"\nError starting app: {e}")
    print("\nTrying alternative method...")
    # Fallback to os.system if subprocess fails
    os.system(f'streamlit run src/app/streamlit_app.py')
