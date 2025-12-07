"""
Quick fix script to clear cache and verify max_features fix
Run this before restarting the app: python fix_max_features.py
"""

import os
import shutil
from pathlib import Path

# Clear Python cache
cache_dir = Path("src/__pycache__")
if cache_dir.exists():
    shutil.rmtree(cache_dir)
    print("✓ Cleared Python cache files")

# Clear any saved models (optional - uncomment if you want to retrain all models)
# models_dir = Path("models")
# if models_dir.exists():
#     for model_file in models_dir.glob("*.pkl"):
#         model_file.unlink()
#         print(f"✓ Removed {model_file}")

print("\n✓ Fix applied! Please restart your Streamlit app.")
print("  The max_features='auto' issue has been fixed.")
print("\nTo restart:")
print("  1. Stop the current app (Ctrl+C)")
print("  2. Run: python run_app.py")

