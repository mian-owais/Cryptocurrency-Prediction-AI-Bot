# Contributing to Cryptocurrency Prediction AI Bot

Welcome! We're excited that you want to contribute to our AI Trading Bot project. This document explains how to get started.

## 🚀 Quick Start for Contributors

### Step 1: Fork the Repository
1. Go to https://github.com/mian-owais/Cryptocurrency-Prediction-AI-Bot
2. Click the **Fork** button (top right)
3. This creates a copy under your GitHub account

### Step 2: Clone Your Fork
```bash
git clone https://github.com/YOUR-USERNAME/Cryptocurrency-Prediction-AI-Bot.git
cd Cryptocurrency-Prediction-AI-Bot
```

### Step 3: Set Up Your Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate it (Windows Git Bash)
source .venv/Scripts/activate

# Or Windows Command Prompt
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 4: Add Upstream Remote
```bash
# Add the original repository as 'upstream'
git remote add upstream https://github.com/mian-owais/Cryptocurrency-Prediction-AI-Bot.git

# Verify your remotes
git remote -v
# Should show:
# origin    -> your fork
# upstream  -> original repository
```

## 🔧 Making Changes

### Step 1: Create a Feature Branch
```bash
# Always work on a new branch (never on main)
git checkout -b feature/your-feature-name

# Good branch names:
# feature/improve-rl-agent
# bugfix/fix-api-error
# docs/update-readme
# enhancement/add-new-coin-support
```

### Step 2: Make Your Changes
- Edit files as needed
- Test your changes locally:
  ```bash
  streamlit run src/app/streamlit_app.py --server.port 8502
  ```
- Run tests if available:
  ```bash
  python -m pytest tests/
  ```

### Step 3: Commit Your Changes
```bash
# See what changed
git status

# Stage your changes
git add .

# Commit with a clear message
git commit -m "Add feature: description of what you did"

# Good commit messages:
# "Add support for Bitcoin prediction"
# "Fix API rate limiting issue"
# "Improve RL agent training efficiency"
```

### Step 4: Push to Your Fork
```bash
git push origin feature/your-feature-name
```

### Step 5: Create a Pull Request
1. Go to your fork on GitHub: https://github.com/YOUR-USERNAME/Cryptocurrency-Prediction-AI-Bot
2. You'll see a **Compare & pull request** button
3. Click it and fill in:
   - **Title**: Clear description of changes
   - **Description**: Explain what you did and why
4. Click **Create Pull Request**

## 📋 Pull Request Guidelines

Your PR should include:
- ✅ Clear title and description
- ✅ Reference to any related issues
- ✅ Summary of changes made
- ✅ Testing verification
- ✅ No conflicts with the main branch

**Example PR Description:**
```
## Description
Added support for predicting Ethereum trends with improved accuracy

## Changes
- Modified src/model_predict_trend.py to handle Ethereum data
- Updated src/app/streamlit_app.py UI for new coin
- Added tests in tests/test_eth_prediction.py

## Testing
- Tested with historical Ethereum data
- Verified predictions are accurate
- No errors in Streamlit dashboard

## Screenshots
[Optional: Add screenshots if UI changes]
```

## 🔄 Keeping Your Fork Updated

Before starting new work, update your fork with latest changes:

```bash
# Fetch latest changes from upstream
git fetch upstream

# Update your main branch
git checkout main
git merge upstream/main

# Push to your fork
git push origin main
```

## 📦 Project Structure

```
Cryptocurrency-Prediction-AI-Bot/
├── src/
│   ├── app/
│   │   └── streamlit_app.py          # Main Streamlit dashboard
│   ├── data_fetcher.py               # CoinGecko API integration
│   ├── eval_utils.py                 # Evaluation metrics
│   ├── model_predict_trend.py        # Prediction model
│   ├── rl_agent.py                   # DQN Reinforcement Learning
│   └── self_learning_loop.py         # Self-learning mechanism
├── tests/
│   └── check_plot.py                 # Test utilities
├── README.md                          # Project documentation
├── CONTRIBUTING.md                    # This file
├── requirements.txt                   # Python dependencies
└── setup.py                           # Package setup
```

## 🐛 Found a Bug?

1. Check if the issue already exists: https://github.com/mian-owais/Cryptocurrency-Prediction-AI-Bot/issues
2. If not, create a new issue with:
   - Clear description of the bug
   - Steps to reproduce
   - Expected vs actual behavior
   - Your environment (Python version, OS, etc.)

## 💡 Feature Requests

Have an idea? Create an issue with:
- Feature description
- Why it would be useful
- Possible implementation approach

## 📞 Need Help?

- Check the [README.md](README.md) for usage instructions
- Review existing code for examples
- Ask questions in pull requests or issues

## 📜 Code Style

- Follow PEP 8 Python style guide
- Use meaningful variable names
- Add comments for complex logic
- Keep functions focused and small

## 🎯 Areas for Contribution

We're looking for help in:
- **ML/AI**: Improving prediction accuracy, new models
- **UI/UX**: Streamlit dashboard improvements
- **Data**: Adding new coins, data sources
- **Testing**: Writing unit tests
- **Documentation**: Improving guides and comments
- **Performance**: Optimization and caching

## ✨ Recognition

Contributors will be recognized in:
- Commit history
- Pull request comments
- README contributors section (if applicable)

---

**Thank you for contributing! Together we're building amazing AI trading tools! 🚀**

For questions, reach out to: **mianowais980@gmail.com**
