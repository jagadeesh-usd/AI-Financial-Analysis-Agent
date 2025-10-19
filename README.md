# 📈 AI-Powered Multi-Agent Financial Analysis System

This repository contains the source code and documentation for a **multi-agent financial analysis system** built for the AAI-520 Final Team Project. Implemented in a **Jupyter Notebook** using **Python**, **LangChain**, and **LangGraph**, the system demonstrates how specialized AI agents collaborate to perform complex financial research tasks end-to-end.

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

The system features a **lead Researcher Agent** that dynamically uses tools to gather data for a given stock ticker (e.g., INTC). The analysis is processed through a **LangGraph workflow** involving a **Planner Agent** (creates research plans), **Executor Agent** (collects data), **Critic Agent** (evaluates output), and **Refiner Agent** (improves the report). A **Memory component** saves key insights to provide context-aware analyses in future runs.

---

## Data Sources & APIs

## 🌐 Data Sources & APIs

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
├── project_notebook.ipynb  # Main Jupyter Notebook with code and documentation
├── requirements.txt        # Python dependencies
├── memory.json             # Agent’s persistent memory (created on first run)
└── README.md               # Project documentation
```

---

## Business Understanding

Traditional financial analysis relies on rigid pipelines and manual effort. **Agentic AI** enables autonomous systems to plan research, integrate diverse data, and self-critique for improved accuracy. This project delivers a scalable solution for investment research, producing nuanced, context-aware reports faster than manual methods.

---

## Methodology

### 🧩 1. Agentic Planning
The **Planner Agent** uses `get_company_info` (yfinance) to assess a company’s profile (e.g., market cap, sector) and generates a tailored research plan (JSON array of tools).

### 🌍 2. Multi-Source Data Gathering
The **Executor Agent** follows the plan, calling tools like:
- `get_stock_news` and `search_specific_news` for news sentiment.
- `get_price_summary` for price trends and technicals (yfinance).
- `clean_filing_content` for EDGAR filings.
- `read_notes_from_memory` for historical context.

### 🧠 3. Self-Critique & Refinement
The **Critic Agent** evaluates the initial report for balance and completeness. The **Refiner Agent** revises it based on feedback, producing a polished output.

### 💾 4. Agentic Memory & Learning
The system saves a one-sentence insight to `memory.json` (e.g., "[2025-10-19] - Intel’s AI pivot is promising but profitability remains a concern") for future context, retaining the last three entries per ticker.

---

## How to Run

To run the notebook locally:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/jagadeesh-usd/AI-Financial-Analysis-Agent.git
   cd AI-Financial-Analysis-Agent
   ```

2. **Create and Activate a Virtual Environment**:
   - **macOS/Linux**:
	 ```bash
	 python3 -m venv venv
	 source venv/bin/activate
	 ```
   - **Windows**:
	 ```bash
	 python -m venv venv
	 .\venv\Scripts\activate
	 ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API Keys**:
   Create a `.env` file in the root directory with:
   ```toml
   OPENAI_API_KEY="sk-..."
   FRED_API_KEY="your_fred_api_key_here"
   NEWS_API_KEY="your_newsapi_key_here"
   ```

5. **Set EDGAR User Agent**:
   In `project_notebook.ipynb`, update the `set_identity` call:
   ```python
   set_identity("Your Name your.email@example.com")
   ```

6. **Run the Notebook**:
   Launch Jupyter Notebook and open `project_notebook.ipynb`:
   ```bash
   jupyter notebook
   ```
   Execute cells sequentially to run the INTC demo or test other tickers.

---

## Key Results & Features

- **✅ Autonomous Agent**: Plans and executes financial research from a single ticker input.
- **✅ Multi-Agent Workflow**: Implements Planner → Executor → Critic → Refiner pipeline using LangGraph.
- **✅ Persistent Memory**: Stores insights in `memory.json` for contextual continuity.
- **✅ Comprehensive Analysis**: Integrates stock data, news, filings, and economic indicators into a polished report (e.g., INTC analysis balancing AI growth with risks).

---

## Team Contributions (AAI-520)

- **Jagadeesh Kumar Sellappan** 
- **Saurav Kumar Subudhi** 

---

## AI Assistance Disclosure

This project adheres to academic integrity principles. AI tools (ChatGPT, GitHub Copilot) were used for:
- ✨ Code formatting (PEP 8 compliance).
- 🧾 Commenting and documentation (e.g., notebook markdowns, README).
- 🪲 Debugging guidance and optimization suggestions.

All AI-generated content was reviewed, validated, and refined by the team to ensure technical accuracy and alignment with course goals.

---

## License

This project is for educational purposes. Please include proper attribution if reusing or extending this work.

---
## Streamlit Version

For details on running the Streamlit version, see README_streamlit.md.
