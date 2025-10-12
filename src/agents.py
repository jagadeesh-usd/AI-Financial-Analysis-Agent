import yfinance as yf
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
import json
import os
import streamlit as st
from fredapi import Fred
from edgar import Company, set_identity # Updated import for SEC filings

# --- API KEY NOTE ---
# The FRED API requires a free API key.
# 1. Get a key from: https://fred.stlouisfed.org/docs/api/api_key.html
# 2. Add it to your Streamlit secrets file (.streamlit/secrets.toml) as:
# FRED_API_KEY = "your_key_here"

try:
    fred_api_key = st.secrets["FRED_API_KEY"]
    fred = Fred(api_key=fred_api_key)
except Exception:
    st.error("FRED API key not found in Streamlit secrets. The economic data tool will be disabled.")
    fred = None

# --- Memory Component ---
MEMORY_FILE = "memory.json"

@tool
def read_notes_from_memory(ticker: str) -> list[str]:
    """
    Reads past analysis notes for a given stock ticker to provide context for a new analysis.
    Use this tool FIRST before any other tool to get historical context.
    """
    if not os.path.exists(MEMORY_FILE):
        return ["No past notes found for this ticker."]
    
    with open(MEMORY_FILE, 'r') as f:
        try:
            data = json.load(f)
            return data.get(ticker, ["No past notes found for this ticker."])
        except json.JSONDecodeError:
            return ["Memory file is empty or corrupted."]

@tool
def save_note_to_memory(ticker: str, note: str) -> str:
    """
    Saves a single, concise key takeaway from the latest analysis to memory for future reference.
    Use this tool LAST after the analysis is complete. The note should be a single sentence.
    """
    if not isinstance(note, str) or not note.strip():
        return "Error: Note must be a non-empty string."
        
    if os.path.exists(MEMORY_FILE) and os.path.getsize(MEMORY_FILE) > 0:
        with open(MEMORY_FILE, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
    else:
        data = {}

    if ticker not in data:
        data[ticker] = []
    
    data[ticker].append(note.strip())
    data[ticker] = data[ticker][-3:]

    with open(MEMORY_FILE, 'w') as f:
        json.dump(data, f, indent=4)
        
    return f"Successfully saved note for {ticker}."

# --- Agentic Tools Definition ---

@tool
def get_company_info(ticker: str) -> dict:
    """Retrieves general information and key metrics for a given stock ticker."""
    stock = yf.Ticker(ticker)
    info = stock.info
    market_cap = info.get('marketCap')
    return {
        "longName": info.get('longName', 'N/A'),
        "marketCap": f"${market_cap:,}" if isinstance(market_cap, (int, float)) else 'N/A',
        "sector": info.get('sector', 'N/A'),
    }

@tool
def get_stock_news(ticker: str) -> list[str]:
    """Fetches the latest news headlines for a given stock ticker."""
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        if not news:
            return ["No recent news found."]
        return [
            article['content']['title'] 
            for article in news[:5]
            if 'content' in article and 'title' in article.get('content', {})
        ]
    except Exception as e:
        return [f"An error occurred: {e}"]

@tool
def get_economic_data(series_id: str = 'GDP') -> dict:
    """
    Fetches the latest data for a given economic series from FRED.
    Use this to understand the broader economic context. For example, use series_id 'GDP' for GDP data.
    """
    if fred is None:
        return {"error": "FRED API key not configured. Cannot fetch economic data."}
    try:
        data = fred.get_series(series_id)
        latest_value = data.iloc[-1]
        latest_date = data.index[-1].strftime('%Y-%m-%d')
        return {
            "series": series_id,
            "latest_value": f"{latest_value:,.2f}",
            "latest_date": latest_date
        }
    except Exception as e:
        return {"error": f"Failed to fetch data for series {series_id}: {e}"}

@tool
def get_latest_filings(ticker: str) -> list[dict]:
    """
    Fetches the latest SEC filings (10-K, 10-Q) for a given stock ticker using the edgar library.
    """
    try:
        # Step 1: Identify yourself to SEC (required).
        set_identity("jagadeesch1981@gmail.com")

        # Step 2: Create a company object and get filings
        company = Company(ticker.upper())
        filings = company.get_filings().filter(form=["10-K", "10-Q"])

        # Step 3: Iterate and convert each filing to a standard Python dictionary
        results = []
        for filing in filings[:3]:  # Top 3 filings
            filing_data = filing.obj()
            
            results.append({
                "form_type": filing_data.get("form"),
                "filed_at": filing_data.get("filingDate"),
                "accession_number": str(filing_data.get("accessionNumber", "")).replace("-", ""),
                "url": filing.url
            })

        if not results:
            return [{"error": f"No recent 10-K or 10-Q filings found for {ticker}."}]

        return results
    except Exception as e:
        return [{"error": f"Failed to fetch SEC filings for ticker {ticker}: {e}"}]

# --- Agent Creation Factory ---
def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    agent = create_openai_functions_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False, return_intermediate_steps=True)

# --- Agent Definitions ---

def get_researcher_agent(llm: ChatOpenAI):
    researcher_tools = [read_notes_from_memory, get_company_info, get_stock_news, get_economic_data, get_latest_filings]
    researcher_system_prompt = (
        "You are an expert financial researcher. Your goal is to produce a detailed analysis paragraph.\n\n"
        "Follow this exact sequence:\n"
        "1. **Consult Memory:** Use `read_notes_from_memory` for historical context.\n"
        "2. **Gather Company Data:** Use `get_company_info` and `get_stock_news` for the latest information.\n"
        "3. **Analyze Macro-Economic Context:** Use `get_economic_data` with a relevant series ID (e.g., 'GDP', 'UNRATE') to understand the broader economic environment.\n"
        "4. **Review Company Filings:** Use `get_latest_filings` to understand the company's official financial health from SEC documents.\n"
        "5. **Synthesize Analysis:** Combine all gathered information into a single, detailed analysis paragraph. This paragraph MUST be your final output.\n\n"
        "**Formatting instructions:** Ensure your final output is a well-formatted, readable paragraph with proper spacing and punctuation. Do not output markdown or any other special formatting."
    )
    return create_agent(llm, researcher_tools, researcher_system_prompt)

def get_critic_agent(llm: ChatOpenAI):
    critic_prompt = ChatPromptTemplate.from_template(
        "You are a meticulous financial 'Critic' agent. Evaluate an analysis based on:\n"
        "1. **Balance:** Does it present both risks and opportunities?\n"
        "2. **Clarity:** Is the language clear and concise?\n"
        "3. **Objectivity:** Is the analysis data-driven?\n\n"
        "Provide short, bulleted feedback. If excellent, state 'No major changes needed'.\n\n"
        "Initial Analysis to Critique:\n{initial_analysis}"
    )
    return critic_prompt | llm

def get_refiner_agent(llm: ChatOpenAI):
    refiner_prompt = ChatPromptTemplate.from_template(
        "You are a 'Refiner' agent. Your task is to rewrite and improve an initial financial analysis based on a critique.\n\n"
        "Initial Analysis:\n{initial_analysis}\n\n"
        "Critique:\n{critique}\n\n"
        "Your Final, Rewritten Analysis (as a single, polished paragraph):\n"
        "**Formatting instructions:** Ensure your final output is a single, well-formatted, and readable paragraph. Do not use markdown. The text should be suitable for direct display in a user interface."
    )
    return refiner_prompt | llm

