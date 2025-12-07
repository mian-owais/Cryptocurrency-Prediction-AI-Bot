# Cryptocurrency Prediction AI Bot

An advanced AI-powered cryptocurrency trading bot with real-time predictions, evaluation dashboards, and a self-learning reinforcement learning agent.

## ÌæØ Key Features

### Ì≥ä Live Trading Dashboard
- **Real-time candlestick charts** for Bitcoin, Ethereum, Dogecoin, Solana, Cardano
- **Multiple timeframes**: 5-minute, daily, weekly predictions
- **Auto-refresh capability** every 60 seconds
- **Dark-themed Plotly visualizations**

### Ì≥à Evaluation & Error Analysis
- **Classification metrics**: Accuracy, Precision, Recall, F1-Score
- **Advanced diagnostics**: Confusion matrix, ROC curves, calibration curves
- **Rolling accuracy metrics** and backtest performance analysis
- **Error analysis** identifying worst predictions

### Ì¥ñ Self-Learning RL Agent
- **DQN-based reinforcement learning** for continuous policy improvement
- **Automatic prediction logging** with timestamps and confidence scores
- **Verification system**: Cross-validates predictions after timeframe expires
- **Reward computation** based on prediction accuracy
- **Offline training** using PyTorch (optional)
- **Interactive controls**: Verify Now, Re-learn Now, Make Prediction

### Ì¥Ñ Robust Data Pipeline
- **CoinGecko API integration** with automatic retries on rate limits
- **Mock data fallback** when API unavailable
- **15-second retry delays** for graceful degradation
- **OHLCV aggregation** with configurable resampling

## Ì≥Å Project Structure

```
AI_Trading_Bot/
‚îú‚îÄ‚îÄ src/
‚îÇ   ‚îú‚îÄ‚îÄ app/
‚îÇ   ‚îÇ   ‚îú‚îÄ‚îÄ streamlit_app.py          # Main Streamlit UI
‚îÇ   ‚îÇ   ‚îî‚îÄ‚îÄ __init__.py
‚îÇ   ‚îú‚îÄ‚îÄ eval_utils.py                 # Evaluation metrics & plots
‚îÇ   ‚îú‚îÄ‚îÄ rl_agent.py                   # DQN agent implementation
‚îÇ   ‚îú‚îÄ‚îÄ data_fetcher.py               # CoinGecko API wrapper
‚îÇ   ‚îú‚îÄ‚îÄ model_predict_trend.py        # Prediction logging
‚îÇ   ‚îú‚îÄ‚îÄ self_learning_loop.py         # Verification & training
‚îÇ   ‚îú‚îÄ‚îÄ model_train.py                # Model training utilities
‚îÇ   ‚îú‚îÄ‚îÄ trading_env.py                # Trading environment
‚îÇ   ‚îî‚îÄ‚îÄ __init__.py
‚îú‚îÄ‚îÄ tests/
‚îÇ   ‚îú‚îÄ‚îÄ test_eval_utils.py
‚îÇ   ‚îú‚îÄ‚îÄ test_imports.py
‚îÇ   ‚îú‚îÄ‚îÄ check_plot.py
‚îÇ   ‚îî‚îÄ‚îÄ test_trading_env.py
‚îú‚îÄ‚îÄ data/
‚îÇ   ‚îî‚îÄ‚îÄ logs/                         # Generated at runtime
‚îÇ       ‚îú‚îÄ‚îÄ prediction_log.csv
‚îÇ       ‚îî‚îÄ‚îÄ verification_log.csv
‚îú‚îÄ‚îÄ docs/
‚îú‚îÄ‚îÄ setup.py
‚îú‚îÄ‚îÄ requirements.txt
‚îú‚îÄ‚îÄ run_app.py
‚îî‚îÄ‚îÄ README.md
```

## Ì∫Ä Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/mian-owais/Cryptocurrency-Prediction-AI-Bot.git
cd AI_Trading_Bot
```

### 2. Setup Environment
```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -e .
```

### 3. Run Dashboard
```bash
streamlit run src/app/streamlit_app.py
```

Dashboard opens at **http://localhost:8501**

## Ì≥ñ Usage

### Dashboard Navigation

**ÌæØ Live Trading Tab**
- Select cryptocurrency and timeframe
- View real-time candlestick charts with MA20
- Monitor prediction trends
- Toggle 60-second auto-refresh

**Ì≥ä Evaluation Tab**
- View comprehensive model metrics
- Analyze confusion matrix and ROC curves
- Explore rolling accuracy trends
- **RL Agent Controls**:
  - `Verify Now`: Manually verify pending predictions
  - `Re-learn Now`: Trigger offline training burst
  - `Make Prediction Now`: Generate new predictions

### Data Exports
- **Prediction logs**: CSV file with all predictions and confidence scores
- **Verification logs**: CSV file with actual outcomes and rewards
- **PDF reports**: Comprehensive evaluation reports

## Ì∑† RL Agent Details

### State Representation
- RSI (Relative Strength Index)
- EMA Difference (fast - slow)
- Momentum (price velocity)
- Volatility (standard deviation)

### Rewards
- **+1**: Correct prediction
- **-1**: Incorrect prediction

### Training
- Experience replay buffer stores verified predictions
- DQN loss minimization using PyTorch
- Configurable learning rate, batch size, training epochs

## Ì¥ß Configuration

### Environment Variables
```bash
export STREAMLIT_PORT=8502             # Custom port
export COINGECKO_RETRY_DELAY=15        # Retry delay seconds
```

### Streamlit Config
Edit `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#0084FF"
backgroundColor = "#0a0e27"
secondaryBackgroundColor = "#1a1f3a"

[server]
port = 8501
headless = false
```

## Ì∑™ Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_eval_utils.py -v

# Run with coverage
pytest --cov=src tests/
```

## Ì∞õ Troubleshooting

**ModuleNotFoundError**
```bash
pip install -e .
```

**Port already in use**
```bash
streamlit run src/app/streamlit_app.py --server.port 8503
```

**CoinGecko rate limits**
- Bot automatically retries 3 times with 15-second delays
- Falls back to mock data if API unavailable
- No manual intervention needed

**Missing PyTorch**
```bash
pip install torch torchvision torchaudio
```

## Ì≥ä Metrics Explained

| Metric | Definition |
|--------|-----------|
| **Accuracy** | Correct predictions / Total predictions |
| **Precision** | True positives / (True positives + False positives) |
| **Recall** | True positives / (True positives + False negatives) |
| **F1-Score** | 2 √ó (Precision √ó Recall) / (Precision + Recall) |
| **Rolling Accuracy** | Accuracy computed over 20-prediction windows |

## Ì¥ó API Integration

### CoinGecko
- **Endpoint**: `/coins/{id}/market_chart`
- **Data**: OHLCV (Open, High, Low, Close, Volume)
- **Update Frequency**: Real-time (as called)
- **Rate Limit**: 10-50 calls/minute (free tier)

## Ì≥ù Requirements

- Python 3.8+
- streamlit >= 1.0.0
- plotly >= 5.0.0
- pandas >= 1.2.0
- numpy >= 1.20.0
- scikit-learn >= 0.24.0
- requests >= 2.26.0
- torch >= 1.9.0 (optional, for DQN training)
- reportlab >= 3.6.0 (PDF export)

See `requirements.txt` for complete list.

## ÔøΩÔøΩ Contact & Support

- **Developer**: Mian M Owais
- **Email**: mianowais980@gmail.com
- **GitHub**: https://github.com/mian-owais
- **Repository**: [Cryptocurrency-Prediction-AI-Bot](https://github.com/mian-owais/Cryptocurrency-Prediction-AI-Bot)

## Ì≥ú License

MIT License - see LICENSE file for details

## ÌæØ Roadmap

- [ ] Historical accuracy tracking dashboards
- [ ] Multi-model ensemble predictions
- [ ] Real-time sentiment analysis
- [ ] Paper trading integration
- [ ] Mobile app interface
- [ ] REST API for external integrations
- [ ] Advanced RL algorithms (PPO, A3C)
- [ ] Real trading with broker integration

---

**Version**: 1.0.0 | **Status**: Active Development | **Last Updated**: November 2025

Built with ‚ù§Ô∏è by Mian M Owais
