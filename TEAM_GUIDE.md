# 📖 Quick Guide for Group Mates - How to Contribute

## 🎯 Goal
We're collaborating on a **Cryptocurrency Trading AI Bot** using Git and GitHub. Here's how to get involved:

---

## ✅ Step-by-Step Setup (5 Minutes)

### 1️⃣ Fork the Repository
- Go to: https://github.com/mian-owais/Cryptocurrency-Prediction-AI-Bot
- Click **Fork** button (top right)
- This creates YOUR copy of the project

### 2️⃣ Clone Your Fork
```bash
git clone https://github.com/YOUR-USERNAME/Cryptocurrency-Prediction-AI-Bot.git
cd Cryptocurrency-Prediction-AI-Bot
```

### 3️⃣ Install Dependencies
```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows Git Bash)
source .venv/Scripts/activate

# Install packages
pip install -r requirements.txt
```

### 4️⃣ Add Upstream Remote
```bash
git remote add upstream https://github.com/mian-owais/Cryptocurrency-Prediction-AI-Bot.git
```

---

## 🔄 Workflow (Every Time You Work)

### Start Work:
```bash
# Update your main branch
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/what-you-are-doing
```

### Finish Work:
```bash
# Add changes
git add .

# Commit with clear message
git commit -m "Add feature: description"

# Push to your fork
git push origin feature/what-you-are-doing
```

### Create Pull Request:
1. Go to your fork on GitHub
2. Click **"Compare & pull request"** button
3. Fill in title and description
4. Click **"Create Pull Request"**
5. Wait for review and merge! ✨

---

## 📂 What You Can Work On

| Area | Examples |
|------|----------|
| **Features** | Add new coins, improve UI, new indicators |
| **Fixes** | Debug errors, improve speed, fix API issues |
| **Tests** | Write tests for features |
| **Docs** | Improve documentation, add examples |

---

## 📞 Need Help?

- Check: [CONTRIBUTING.md](CONTRIBUTING.md) in the repository
- Ask: Create an issue on GitHub
- Email: mianowais980@gmail.com

---

## ⚡ Pro Tips

✅ **DO:**
- Work on your own branch (never directly on `main`)
- Write clear commit messages
- Test your changes before pushing
- Keep pull requests focused (one feature per PR)

❌ **DON'T:**
- Push to the main branch
- Make huge changes in one PR
- Forget to pull updates from upstream

---

## 🚀 Test Your Setup

After installation, run:
```bash
streamlit run src/app/streamlit_app.py --server.port 8502
```

Visit: http://localhost:8502

You should see the **Cryptocurrency Trading Dashboard**! ✨

---

**Happy Contributing! Questions? Ask in issues or email! 🎉**
