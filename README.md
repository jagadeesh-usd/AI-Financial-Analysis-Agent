# 📈 AI-Powered Multi-Agent Financial Analysis System

This repository contains the complete source code and documentation for a **real-world financial analysis system powered by agentic AI**.  
Built with **Python**, **Streamlit**, and the **LangChain framework**, this project demonstrates how multiple specialized AI agents can collaborate to reason, plan, act, and iteratively improve complex financial research tasks.

---

## 🗂️ Table of Contents

- [Project Overview](#project-overview)
- [Data Sources & APIs](#data-sources--apis)
- [Project Structure](#project-structure)
- [Business Understanding](#business-understanding)
- [Methodology](#methodology)
- [How to Run](#how-to-run)
- [Key Results & Features](#key-results--features)
- [Team Contributions](#team-contributions)
- [AI Assistance Disclosure](#ai-assistance-disclosure)

---

## 🧠 Project Overview

The application features a **lead Researcher Agent** that dynamically uses a suite of tools to gather data on a given stock ticker.  
Its findings are passed through a **LangGraph workflow** where a **Critic Agent** evaluates the analysis and a **Refiner Agent** improves it — demonstrating a complete _"reason → critique → improve"_ loop.  

The system also “learns” by saving key insights to a local memory, which it consults in future analyses to provide more **context-aware insights**.

---

## 🌐 Data Sources & APIs

The agent gathers data from multiple real-world sources for comprehensive financial insight:

| Source | Purpose |
|---------|----------|
| **Yahoo Finance (yfinance)** | Stock prices, company information, and latest news |
| **Federal Reserve Economic Data (fredapi)** | Macroeconomic indicators (e.g., GDP, unemployment) |
| **SEC EDGAR (edgar)** | Official company filings (10-K, 10-Q) |

---

## 🧩 Project Structure
```bash
AI-Financial-Analysis-Agent/
│
├── .streamlit/
│   └── secrets.toml        # API Keys and secrets (not committed)
│
├── src/
│   ├── agents.py           # Defines all agent tools and prompts
│   └── chain.py            # Builds the LangGraph workflow
│
├── app.py                  # Main Streamlit UI and application logic
├── requirements.txt        # Python dependencies
├── memory.json             # Agent’s persistent memory (created on first run)
└── README.md               # Project documentation
```
---

## 💼 Business Understanding

Traditional financial analysis pipelines are often rigid and require significant manual intervention.  
**Agentic AI** represents a paradigm shift — enabling systems that can autonomously plan research, integrate diverse data sources in real-time, and even **critique their own outputs** to improve quality.  

This project demonstrates a **practical and scalable application** of this technology for investment research, capable of delivering nuanced insights faster and more efficiently.

---

## 🧮 Methodology

### 🧩 1. Agentic Planning & Research
The **Researcher Agent** receives a stock ticker and autonomously plans its research steps based on a predefined, sequential strategy.

### 🌍 2. Multi-Source Data Gathering
The agent dynamically uses its suite of tools to fetch:
- Real-time stock data and news (Yahoo Finance)
- Macroeconomic data (FRED)
- Official filings (SEC EDGAR)

### 🧠 3. Self-Critique & Refinement
The **Critic Agent** evaluates the initial report for balance, clarity, and completeness.  
The **Refiner Agent** then rewrites the report based on feedback to produce an improved final version.

### 💾 4. Agentic Memory & Learning
The final step saves a **key one-sentence insight** to a persistent `memory.json` file.  
This allows the agent to _“remember”_ past analyses and provide richer context on subsequent runs.

---

## ⚙️ How to Run

Follow these steps to set up and run the project locally:

```bash
1️⃣ Clone the Repository

git clone https://github.com/jagadeesh-usd/AI-Financial-Analysis-Agent.git
cd AI-Financial-Analysis-Agent

2️⃣ Create and Activate a Virtual Environment

macOS / Linux

python3 -m venv venv
source venv/bin/activate

Windows

python -m venv venv
.\venv\Scripts\activate

3️⃣ Install Dependencies

pip install -r requirements.txt

4️⃣ Configure API Keys

Create a .streamlit/secrets.toml file and add your keys:

# .streamlit/secrets.toml
OPENAI_API_KEY = "sk-..."
FRED_API_KEY = "your_fred_api_key_here"

5️⃣ Set Your EDGAR User Agent

The SEC requires a descriptive user agent for API calls.
Edit src/agents.py and update the set_identity call:

set_identity("Your Name your.email@example.com")

6️⃣ Launch the Application

streamlit run app.py
```

---

🚀 Key Results & Features

✅ Fully Autonomous Agent
Plans and executes complex financial research tasks from a single stock ticker input.

✅ Multi-Agent Workflow
Implements a Researcher → Critic → Refiner pipeline using LangGraph for state management and self-improvement.

✅ Persistent Memory
Stores one-line insights for each stock and reuses them for contextual continuity.

✅ Interactive Streamlit UI
Shows real-time status updates and streams the final report word-by-word in an intuitive interface.

---

👥 Team Contributions (AAi-520)


---

🤖 AI Assistance Disclosure

This project follows academic integrity and responsible AI usage principles.
AI tools, such as ChatGPT and GitHub Copilot, were utilized for:
	•	✨ Code formatting and refactoring (PEP8 compliance)
	•	🧾 Commenting and documentation (including this README)
	•	🪲 Debugging guidance and optimization suggestions

All AI-generated content was reviewed, validated, and refined by the team to ensure full understanding and technical accuracy.

---

📚 License

This project is distributed for educational and research purposes.
Please include proper attribution if reusing or extending this work.