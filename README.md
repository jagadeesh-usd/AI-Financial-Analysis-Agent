📈 AI-Powered Multi-Agent Financial Analysis System
This project is a sophisticated financial analysis system built with a multi-agent AI architecture using Python, Streamlit, and the LangChain framework. The system emulates how real-world investment firms use agentic AI to reason, plan, and act on complex financial data from multiple sources.

The application features a lead Researcher Agent that dynamically uses a suite of tools to gather data. Its findings are then passed through a workflow where a Critic Agent evaluates the analysis and a Refiner Agent improves it, demonstrating a complete "reason-critique-improve" loop. The agent also learns from past analyses by saving key insights to a local memory.

✨ Key Features
This project successfully implements all required agentic functions and workflow patterns:

Agent Functions:

🤖 Plans: The Researcher Agent autonomously plans and executes a multi-step research strategy for any given stock ticker.

🛠️ Uses Tools Dynamically: The agent has access to a suite of tools for real-time data retrieval, including:

Yahoo Finance (yfinance) for stock prices and company news.

Federal Reserve Economic Data (fredapi) for macroeconomic context.

SEC EDGAR (edgar) for official company filings.

🤔 Self-Reflects: The system uses a Critic Agent to evaluate the quality of the initial analysis, providing a mechanism for self-reflection and critique.

🧠 Learns Across Runs: The agent saves key takeaways from each analysis to a local memory.json file, which it consults in future runs to provide more context-aware insights.

Workflow Patterns:

⛓️ Prompt Chaining: Implemented in helper functions, such as the sentiment analysis tool which classifies news headlines.

🔀 Routing: The entire workflow is built on LangGraph, which routes the application's state between the Researcher, Critic, and Refiner agents.

🔄 Evaluator-Optimizer: The Critic -> Refiner loop is a direct implementation of this pattern, where an initial analysis is generated, evaluated, and then refined based on feedback.

🏛️ System Architecture
The application is built on a modular, multi-agent architecture orchestrated by LangGraph.

UI (Streamlit): The user enters a stock ticker.

LangGraph Workflow: The ticker is passed to the LangGraph state machine.

Researcher Agent: This agent is the first node. It plans its research and calls multiple tools (Yahoo Finance, FRED, EDGAR) to gather data and synthesize an initial analysis.

Critic Agent: The analysis is routed to this agent, which evaluates it against predefined criteria (balance, clarity, objectivity).

Refiner Agent: The initial analysis and the critique are routed to this agent, which rewrites and improves the final report.

Memory Agent: The final, refined analysis is used to generate a key insight, which is saved to a local JSON file for future use.

UI (Streamlit): The data, workflow status, and final report are displayed to the user in real-time.

🛠️ Technology Stack
Frontend: Streamlit

Agent Framework: LangChain, LangGraph

LLM: OpenAI GPT-4 Turbo / GPT-4o Mini

Data Tools:

yfinance: Stock prices & news

fredapi: Economic data

edgar: SEC filings

Core Language: Python

🚀 Setup and Installation
Follow these steps to run the project locally.

1. Clone the Repository

git clone [https://github.com/YourUsername/AI-Financial-Analysis-Agent.git](https://github.com/YourUsername/AI-Financial-Analysis-Agent.git)
cd AI-Financial-Analysis-Agent

2. Create and Activate a Virtual Environment

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate

3. Install Dependencies

pip install -r requirements.txt

4. Configure API Keys
This project requires API keys for OpenAI and FRED.

Create a secrets file at .streamlit/secrets.toml and add your keys:

# .streamlit/secrets.toml

OPENAI_API_KEY = "sk-..."
FRED_API_KEY = "your_fred_api_key_here"

5. Set Your EDGAR User Agent
The SEC requires a user agent for API calls. Open src/agents.py and replace the placeholder in the get_latest_filings tool with your own information (e.g., name and email).

▶️ How to Run
Once the setup is complete, run the Streamlit application from your terminal:

streamlit run app.py

The application will open in your web browser. Enter a stock ticker in the sidebar and click the "Generate" button to start the analysis.

📂 Project Structure
The code is organized into a modular structure for clarity and maintainability.

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
├── memory.json             # Agent's memory file (created on first run)
└── README.md               # This file
