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

## Project Overview

The application features a **lead Researcher Agent** that dynamically uses a suite of tools to gather data on a given stock ticker.  
Its findings are passed through a **LangGraph workflow** where a **Critic Agent** evaluates the analysis and a **Refiner Agent** improves it — demonstrating a complete _"reason → critique → improve"_ loop.  

The system also “learns” by saving key insights to a local memory, which it consults in future analyses to provide more **context-aware insights**.

---

## Data Sources & APIs

The system integrates multiple real-world data sources and APIs for comprehensive financial insights. These are accessed via specialized tools in the agents, enabling dynamic data retrieval based on company profile (e.g., market cap, sector). Key sources include:

| Source/API | Library | Purpose & Details |
|------------|---------|-------------------|
| **Yahoo Finance** | `yfinance` | Stock prices (current, historical), company information (e.g., name, market cap, sector), news headlines, financial ratios (e.g., P/E, price-to-book, debt-to-equity, ROE, profit margins), analyst ratings (buy/hold/sell consensus, price targets), and technical indicators (e.g., 52-week range, 50/200-day MAs, 14-day RSI for trend/momentum analysis). Used for core financial and market data. |
| **Federal Reserve Economic Data (FRED)** | `fredapi` | Macroeconomic indicators (e.g., GDP, unemployment rates, CPI, interest rates). Fetches time-series data for contextual economic analysis, especially for mid/large-cap companies. Requires a free API key from fred.stlouisfed.org. |
| **SEC EDGAR** | `edgartools` | Official company filings (e.g., 10-K annual reports, 10-Q quarterly reports). Retrieves and cleans recent filings (last 3 years) for insights into financial health, risks, and operations. Cleans XBRL tags and normalizes text for readability. Requires setting a user agent for compliance. |
| **NewsAPI** | `requests` (via API) | Targeted news searches for sentiment and events, focusing on the last 30 days with keywords (e.g., 'partnerships', 'acquisitions', 'product launch'). Returns top headlines for specific topics, integrated into news analysis. Requires a free API key from newsapi.org. |
| **Google Trends** | `pytrends` | Public interest trends for keywords (e.g., company name over the last 3 months). Provides average interest scores (0-100), peak dates, and related queries. Useful for consumer-facing sectors to gauge brand sentiment and popularity. No API key required. |

These sources enable multi-faceted analysis, from quantitative metrics (e.g., ratios, prices) to qualitative insights (e.g., news sentiment, filings). Tools dynamically select sources based on the research plan.

---

## Project Structure
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

## Methodology

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

## How to Run

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

## Key Results & Features

✅ Fully Autonomous Agent
Plans and executes complex financial research tasks from a single stock ticker input.

✅ Multi-Agent Workflow
Implements a Researcher → Critic → Refiner pipeline using LangGraph for state management and self-improvement.

✅ Persistent Memory
Stores one-line insights for each stock and reuses them for contextual continuity.

✅ Interactive Streamlit UI
Shows real-time status updates and streams the final report word-by-word in an intuitive interface.

---

## Team Contributions (AAI-520)

- **Jagadeesh Kumar Sellappan** 
- **Saurav Kumar Subudhi** 

---

## AI Assistance Disclosure

This project follows academic integrity and responsible AI usage principles.
AI tools, such as ChatGPT and GitHub Copilot, were utilized for:
	•	✨ Code formatting and refactoring (PEP8 compliance)
	•	🧾 Commenting and documentation (including this README)
	•	🪲 Debugging guidance and optimization suggestions

All AI-generated content was reviewed, validated, and refined by the team to ensure full understanding and technical accuracy.

---

## License

This project is distributed for educational and research purposes.
Please include proper attribution if reusing or extending this work.