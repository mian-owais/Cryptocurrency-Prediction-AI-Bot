import os
import sys
from pathlib import Path

# Add the src directory to the Python path
src_dir = Path(__file__).parent / 'src'
sys.path.append(str(src_dir))

# Run the streamlit app
os.system('streamlit run src/app/streamlit_app.py')
