# Step-by-Step Guide: How to Run the Cryptocurrency Prediction AI Bot

## Prerequisites Checklist

Before running the application, make sure you have:

- ✅ Python 3.8 or higher installed
- ✅ pip (Python package installer) available
- ✅ Internet connection (for CoinGecko API)

---

## Step 1: Open Terminal/Command Prompt

### Option A: Using Windows PowerShell
1. Press `Windows Key + X`
2. Select "Windows PowerShell" or "Terminal"
3. Or search for "PowerShell" in the Start menu

### Option B: Using Command Prompt
1. Press `Windows Key + R`
2. Type `cmd` and press Enter

---

## Step 2: Navigate to Project Directory

Navigate to the project folder. Based on your setup, run:

```powershell
cd "C:\Users\DELL\Downloads\Cryptocurrency-Prediction-AI-Bot-main (4)\Cryptocurrency-Prediction-AI-Bot-main"
```

**Or if you're already in the Downloads folder:**
```powershell
cd ".\Cryptocurrency-Prediction-AI-Bot-main"
```

**Verify you're in the right directory:**
```powershell
dir
```

You should see files like:
- `run_app.py`
- `requirements.txt`
- `src/` folder
- `README.md`

---

## Step 3: Check Python Installation

Verify Python is installed and accessible:

```powershell
python --version
```

**Expected output:** `Python 3.8.x` or higher

**If Python is not found:**
- Install Python from [python.org](https://www.python.org/downloads/)
- Make sure to check "Add Python to PATH" during installation

---

## Step 4: Install Required Packages

Install all required dependencies:

```powershell
pip install -r requirements.txt
```

**This will install:**
- streamlit (web framework)
- pandas, numpy (data processing)
- scikit-learn, xgboost (machine learning)
- plotly (visualization)
- requests, pycoingecko (API access)
- And other dependencies...

**Note:** This may take a few minutes. You'll see progress bars for each package.

**If you get permission errors:**
```powershell
pip install --user -r requirements.txt
```

---

## Step 5: Verify Installation

Quick check to ensure key packages are installed:

```powershell
python -c "import streamlit; print('Streamlit:', streamlit.__version__)"
python -c "import pandas; print('Pandas:', pandas.__version__)"
```

---

## Step 6: Run the Application

### Method 1: Using run_app.py (Recommended)

```powershell
python run_app.py
```

### Method 2: Direct Streamlit Command

```powershell
streamlit run src/app/streamlit_app.py
```

---

## Step 7: Access the Dashboard

After running the command, you should see:

```
Starting Streamlit app from: C:\Users\DELL\Downloads\...
==================================================
Cryptocurrency Prediction AI Bot
==================================================

The app will open in your browser at http://localhost:8501
Press Ctrl+C to stop the server
```

**The browser should open automatically.** If not:

1. Open your web browser (Chrome, Firefox, Edge, etc.)
2. Navigate to: `http://localhost:8501`

---

## Step 8: Using the Dashboard

### Live Trading Tab

1. **Select Cryptocurrency**: Choose from Bitcoin, Ethereum, Dogecoin, Solana, Cardano
2. **Select Timeframe**: 
   - "Next 5 Minutes" - Short-term prediction
   - "Next Day" - Daily prediction
   - "Next Week" - Weekly prediction
3. **Select Model**:
   - **RandomForest**: Ensemble learning model
   - **XGBoost**: Gradient boosting model
   - **RL_Agent**: Reinforcement learning agent
   - **Random**: Simple rule-based fallback
4. **Enable "Retrain Model with Fresh Data"**: 
   - ✅ Check this to train models with latest CoinGecko data
   - Models will combine fresh + historical data
5. **View Predictions**: See real-time predictions with confidence scores

### Evaluation Tab

- View model performance metrics
- Analyze prediction accuracy
- Check error analysis
- Use RL Agent controls

---

## Troubleshooting

### Problem: "ModuleNotFoundError" or "No module named 'src'"

**Solution:**
```powershell
# Make sure you're in the project root directory
cd "C:\Users\DELL\Downloads\Cryptocurrency-Prediction-AI-Bot-main (4)\Cryptocurrency-Prediction-AI-Bot-main"

# Verify you're in the right place
dir run_app.py
```

### Problem: "streamlit: command not found"

**Solution:**
```powershell
pip install streamlit
```

### Problem: Port 8501 already in use

**Solution:**
```powershell
# Use a different port
streamlit run src/app/streamlit_app.py --server.port 8502
```

### Problem: CoinGecko API rate limit errors

**Solution:**
- The app automatically retries with delays
- Falls back to mock data if API is unavailable
- Wait a few minutes and try again

### Problem: Import errors for model_manager

**Solution:**
```powershell
# Make sure you're running from the project root
python -c "import sys; sys.path.insert(0, '.'); from src.model_manager import predict_with_model; print('OK')"
```

---

## Stopping the Application

To stop the Streamlit server:

1. Go back to the terminal/PowerShell window
2. Press `Ctrl + C`
3. Confirm by pressing `Ctrl + C` again if prompted

---

## Quick Start Summary

```powershell
# 1. Navigate to project
cd "C:\Users\DELL\Downloads\Cryptocurrency-Prediction-AI-Bot-main (4)\Cryptocurrency-Prediction-AI-Bot-main"

# 2. Install dependencies (first time only)
pip install -r requirements.txt

# 3. Run the app
python run_app.py

# 4. Open browser to http://localhost:8501
```

---

## What Happens When You Run the App

1. ✅ **Application starts** - Streamlit server initializes
2. ✅ **Browser opens** - Dashboard loads automatically
3. ✅ **Data fetching** - When you select a model/timeframe:
   - Fetches fresh data from CoinGecko API
   - Loads historical data from cache
   - Combines datasets
   - Trains selected model
   - Makes predictions
4. ✅ **Display results** - Shows predictions with confidence scores

---

## Next Steps

Once the app is running:

1. **Try different models** - Compare RandomForest vs XGBoost predictions
2. **Change timeframes** - See how predictions vary for different horizons
3. **Enable retraining** - Watch models train on fresh data
4. **Check evaluation tab** - Analyze model performance over time

---

## Need Help?

If you encounter any issues:
1. Check the error message in the terminal
2. Verify all dependencies are installed
3. Make sure you're in the correct directory
4. Check your internet connection (for CoinGecko API)

Happy trading! 📈

