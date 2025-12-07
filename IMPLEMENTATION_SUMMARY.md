# Implementation Summary: Real-time Model Training and Prediction

## Overview
This document summarizes the changes made to enable real-time model training and prediction when users interact with the cryptocurrency prediction dashboard.

## Key Changes

### 1. New Module: `model_manager.py`
Created a comprehensive module that handles:
- **Fresh Data Fetching**: Automatically fetches the latest data from CoinGecko API
- **Data Combination**: Merges fresh data with historical cached data
- **Model Training**: Trains models (RandomForest, XGBoost, RL_Agent) on combined dataset
- **Prediction**: Makes predictions using trained models
- **Data Persistence**: Saves processed data and trained models for future use

### 2. Updated Streamlit App (`streamlit_app.py`)
Enhanced the trading dashboard to:
- **Model Selection**: Added model selection dropdown (RandomForest, XGBoost, RL_Agent, Random)
- **Automatic Retraining**: When user selects a model or changes timeframe, the system:
  1. Fetches fresh data from CoinGecko API
  2. Loads historical data (if available)
  3. Combines old + new data
  4. Trains the selected model on combined dataset
  5. Makes predictions using the trained model
- **Real-time Updates**: Models retrain with fresh data whenever settings change

## How It Works

### CoinGecko API Updates
- **Update Frequency**: CoinGecko API provides real-time data when called (not on a fixed schedule)
- **Data Freshness**: Each time a prediction is requested, fresh data is fetched from the API
- **Historical Data**: Previously processed data is cached and combined with fresh data for training

### Model Training Flow

1. **User Action**: User selects a model (e.g., RandomForest) or changes timeframe (e.g., "Next Day")
2. **Data Fetching**: 
   - Fetches fresh price data from CoinGecko API
   - Loads historical processed data from cache (if exists)
3. **Data Preparation**:
   - Adds technical indicators (RSI, MACD, SMA, EMA, Bollinger Bands, etc.)
   - Combines old + new data, removing duplicates
   - Sorts by timestamp
4. **Model Training**:
   - Trains selected model on combined dataset using walk-forward validation
   - Saves trained model for future use
5. **Prediction**:
   - Uses trained model to predict next price movement
   - Returns prediction with confidence score

### Supported Models

1. **RandomForest**: Ensemble learning model using random forest regressor
2. **XGBoost**: Gradient boosting model for regression
3. **RL_Agent**: Reinforcement learning agent (DQN) for trend prediction
4. **Random**: Simple rule-based fallback predictor

### Timeframes

- **Next 5 Minutes**: Uses 0.1 days of data, 1-minute candles
- **Next Day**: Uses 2 days of data, 30-minute candles  
- **Next Week**: Uses 14 days of data, 6-hour candles

## File Structure

```
src/
├── model_manager.py          # NEW: Handles training and prediction
├── app/
│   └── streamlit_app.py     # UPDATED: Integrated model selection and training
├── model_train.py           # Existing: Model training utilities
├── model_predict.py         # Existing: Prediction utilities
├── data_fetcher.py          # Existing: CoinGecko API wrapper
└── ...
```

## Key Features

### ✅ Automatic Data Refresh
- Fetches fresh data from CoinGecko API on every prediction request
- Combines with historical data for comprehensive training

### ✅ Model Retraining
- Models retrain automatically when:
  - User selects a different model
  - User changes timeframe
  - User enables "Retrain Model with Fresh Data" option

### ✅ Data Persistence
- Trained models saved to `models/` directory
- Processed data cached in `data/processed/crypto_data.csv`
- Prediction logs saved to `data/logs/prediction_log.csv`

### ✅ Error Handling
- Graceful fallback to rule-based prediction if ML model fails
- Handles API rate limits and network errors
- Validates data before training

## Usage

### Running the Application

**Main Entry Point (Recommended):**
```bash
python run_app.py
```

**Alternative Method:**
```bash
streamlit run src/app/streamlit_app.py
```

The `run_app.py` file is the main entry point that:
- Sets up proper Python paths
- Ensures correct working directory
- Launches the Streamlit dashboard

### Using the Dashboard

1. **Select Cryptocurrency**: Choose from Bitcoin, Ethereum, Dogecoin, Solana, Cardano
2. **Select Timeframe**: Choose "Next 5 Minutes", "Next Day", or "Next Week"
3. **Select Model**: Choose RandomForest, XGBoost, RL_Agent, or Random
4. **Enable Retraining**: Toggle "Retrain Model with Fresh Data" to always use latest data
5. **View Predictions**: See real-time predictions with confidence scores

### What Happens Behind the Scenes

When you change any setting:
1. System fetches fresh data from CoinGecko API
2. Loads historical data from cache
3. Combines datasets
4. Trains selected model (if retraining enabled)
5. Makes prediction
6. Displays results with confidence metrics

## Technical Details

### Data Flow
```
User Action → Fetch Fresh Data → Load Historical Data → Combine → 
Add Indicators → Train Model → Make Prediction → Display Results
```

### Model Training
- Uses walk-forward time-series cross-validation
- Hyperparameter tuning via Optuna (if available)
- Saves best model based on RMSE

### Prediction Output
```python
{
    'predicted_label': 'Increase' | 'Decrease' | 'Constant',
    'predicted_price': float,
    'last_price': float,
    'predicted_pct_change': float,
    'confidence': float (0-100),
    'model_name': str,
    'timestamp': ISO datetime string
}
```

## Error Resolution

### Fixed Issues
1. ✅ Models now train on fresh + historical data
2. ✅ Predictions use actual trained ML models (not just rule-based)
3. ✅ Data fetching handles API updates correctly
4. ✅ Proper data persistence and caching
5. ✅ Deprecated pandas methods updated
6. ✅ Index handling for multi-index DataFrames

### Known Limitations
- CoinGecko free tier has rate limits (handled with retries)
- Model training can take a few seconds (shows spinner)
- RL_Agent requires PyTorch for full functionality

## Future Enhancements

Potential improvements:
- [ ] Add LSTM model support
- [ ] Implement model ensemble predictions
- [ ] Add more technical indicators
- [ ] Real-time streaming data updates
- [ ] Model performance comparison dashboard
- [ ] Automated hyperparameter optimization

## Testing

To test the implementation:
1. Run the Streamlit app
2. Select different models and timeframes
3. Observe that models retrain with fresh data
4. Check `models/` directory for saved models
5. Check `data/processed/` for cached data
6. Verify predictions change based on fresh data

## Conclusion

The system now fully supports:
- ✅ Real-time data fetching from CoinGecko API
- ✅ Automatic model retraining with fresh + historical data
- ✅ Model selection and prediction in the dashboard
- ✅ Proper error handling and fallbacks
- ✅ Data persistence and caching

Users can now interact with the dashboard and get predictions from models trained on the latest available data combined with historical context.

