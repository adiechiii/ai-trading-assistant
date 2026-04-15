# 🤖 AI Trading Assistant

AI-powered trading decision system built with **Python, Streamlit, OpenAI (Vision), and Claude (Risk Analysis)**.

---

## 🚀 Overview

This project is designed to act as a **trading decision assistant**, not a signal generator.

It helps traders:

* Reduce emotional trading
* Validate setups before execution
* Avoid late or low-quality entries

The system combines:

* 📊 Chart analysis (TradingView screenshots)
* 🧠 AI reasoning (OpenAI)
* ⚖️ Risk validation (Claude)

---

## ⚙️ Features

* 🖼️ Upload **TradingView chart screenshot (1H)**
* 📝 Upload **analysis screenshot**
* 🤖 Dual AI system:

  * OpenAI → trader perspective
  * Claude → risk manager perspective
* 📉 Detect:

  * Trend & structure
  * Entry quality (early / late / extended)
  * Trade validity
* 🧾 Generate:

  * Bias (LONG / SHORT)
  * Trade plan (Entry / SL / TP)
  * Confidence score
  * Final decision:

    * STRONG
    * WEAK
    * WATCHLIST
    * NOT ACTIONABLE
* 📤 Telegram alerts (optional)
* 💾 History tracking

---

## 🧠 How It Works

```text
Analysis Screenshot + Chart Screenshot
            ↓
   OpenAI (Trader Analysis)
            ↓
 Claude (Risk Management Review)
            ↓
     Decision Engine
            ↓
     Final Trade Decision
```

---

## 🛠️ Tech Stack

* Python
* Streamlit
* OpenAI API (Vision + reasoning)
* Anthropic Claude API
* Requests
* Linux Server (Remote SSH)
* GitHub (version control)

---

## 🖥️ Run on a New Linux Server

### 1. Clone repository

```bash
git clone https://github.com/YOUR_USERNAME/ai-trading-assistant.git
cd ai-trading-assistant
```

---

### 2. Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Create `.env` file

```bash
nano .env
```

Paste your API keys:

```env
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_claude_key
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

⚠️ Never upload `.env` to GitHub

---

### 5. Run the app

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

---

### 6. Open in browser

```text
http://YOUR_SERVER_IP:8501
```

---

## 🔗 Run with VS Code Remote SSH

### 1. Connect to server

* `Ctrl + Shift + P`
* Select `Remote-SSH: Connect to Host`

---

### 2. Open project folder

```text
/root/ai-trading-assistant
```

---

### 3. Open terminal

```text
Ctrl + `
```

---

### 4. Activate environment

```bash
source .venv/bin/activate
```

---

### 5. Run app

```bash
streamlit run app.py
```

---

### 6. Open app

```text
http://localhost:8501
```

---

## 🔄 Daily Workflow (Pipeline)

```bash
cd /root/ai-trading-assistant
source .venv/bin/activate
git pull
streamlit run app.py
```

---

## ▶️ Run in Background (Server)

```bash
nohup streamlit run app.py --server.address 0.0.0.0 --server.port 8501 > streamlit.log 2>&1 &
```

Stop it:

```bash
pkill -f "streamlit run app.py"
```

---

## 📁 Project Structure

```text
ai-trading-assistant/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example
```

---

## 🔐 Security

* `.env` is ignored via `.gitignore`
* API keys are never stored in GitHub
* Always rotate keys if exposed

---

## ⚠️ Disclaimer

This project is for **educational purposes only**.

It is NOT financial advice and should not be used for live trading without proper risk management.

---

## 👨‍💻 Author

Built as a personal AI trading system to improve trading discipline and decision-making.
