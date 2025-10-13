import yfinance as yf
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
import json
import os
import streamlit as st
from fredapi import Fred
from edgar import Company, set_identity
import requests
import re
from bs4 import BeautifulSoup
from datetime import datetime
from pytrends.request import TrendReq


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

def clean_filing_content(text):
    """
    Clean the extracted filing content by removing XBRL/inline tags and normalizing whitespace.
    """
    # Remove all XBRL/inline tags (e.g., xbrli:shares, iso4217:USD, us-gaap:*, etc.)
    text = re.sub(r'\b(xbrli|us-gaap|dei|srt|aapl|intc):[^\s>]+', '', text)

    # Remove standalone tags like "P1Y", "c-1", "f-46", etc.
    text = re.sub(r'\b([A-Za-z]+-[0-9]+|[A-Za-z]+[0-9]+|[A-Z]{2,}[A-Za-z]*\d*)\b', '', text)

    # Remove dates in the format "YYYY-MM-DD" if they are standalone
    text = re.sub(r'(?<!\d)\d{4}-\d{2}-\d{2}(?!\d)', '', text)

    # Remove numbers like "0000050863" if they are standalone
    text = re.sub(r'(?<!\d)\d{8,}(?!\d)', '', text)

    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text).strip()

    return text
    
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
    
    note_with_date = f"[**{datetime.now().strftime('%Y-%m-%d')}**] - {note.strip()}"
    data[ticker].append(note_with_date)
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
            for article in news[:10]
            if 'content' in article and 'title' in article.get('content', {})
        ]
    except Exception as e:
        return [f"An error occurred: {e}"]

@tool
def get_price_summary(ticker: str) -> dict:
    """
    Retrieves a summary of the stock's price movements over the last 30 days.
    Provides the recent high, low, and percentage change.
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="30d")
        if hist.empty:
            return {"error": "Could not retrieve 30-day price history."}

        thirty_day_high = hist['High'].max()
        thirty_day_low = hist['Low'].min()
        thirty_day_close = hist['Close'][-1]
        thirty_day_open = hist['Open'][0]
        thirty_day_change_pct = ((thirty_day_close - thirty_day_open) / thirty_day_open) * 100

        return {
            "30-day high": f"${thirty_day_high:.2f}",
            "30-day low": f"${thirty_day_low:.2f}",
            "30-day change": f"{thirty_day_change_pct:.2f}%"
        }
    except Exception as e:
        return {"error": f"Failed to calculate price summary: {e}"}

@tool
def get_financial_ratios(ticker: str) -> dict:
    """
    Calculates and returns key financial ratios for a given stock ticker.
    This provides insights into the company's valuation, profitability, and financial health.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        ratios = {
            "trailing_pe": info.get("trailingPE", "N/A"),
            "forward_pe": info.get("forwardPE", "N/A"),
            "price_to_book": info.get("priceToBook", "N/A"),
            "price_to_sales": info.get("priceToSalesTrailing12Months", "N/A"),
            "debt_to_equity": info.get("debtToEquity", "N/A"),
            "return_on_equity": info.get("returnOnEquity", "N/A"),
            "profit_margins": info.get("profitMargins", "N/A"),
        }
        
        return ratios
    except Exception as e:
        return {"error": f"Could not retrieve financial ratios: {e}"}

@tool
def get_analyst_ratings(ticker: str) -> dict:
    """
    Fetches the latest analyst ratings and price targets for a stock.
    This helps understand the consensus view from market professionals.
    """
    try:
        stock = yf.Ticker(ticker)
        recommendations = stock.recommendations
        
        if recommendations.empty:
            return {"message": "No analyst ratings found for this period."}
            
        # Get the most recent ratings
        latest_ratings = recommendations.tail(5)
        
        # Summarize ratings
        rating_counts = recommendations['strongBuy'].count() + recommendations['buy'].count()
        hold_counts = recommendations['hold'].count()
        sell_counts = recommendations['sell'].count() + recommendations['strongSell'].count()

        summary = {
            "period": recommendations.index.max().strftime('%Y-%m'),
            "buy_ratings": int(rating_counts),
            "hold_ratings": int(hold_counts),
            "sell_ratings": int(sell_counts),
            "latest_recommendations": latest_ratings[['firm', 'toGrade']].to_dict('records')
        }
        return summary
    except Exception as e:
        return {"error": f"Could not retrieve analyst ratings: {e}"}

@tool
def get_google_trends(keyword: str, timeframe: str = 'today 3-m') -> dict:
    """
    Fetches Google Trends data for a specific keyword over a given timeframe (e.g., 'today 3-m').
    This is useful for gauging public interest in a company or its products, especially for consumer brands.
    The keyword should ideally be the company's name (e.g., 'NVIDIA').
    """
    try:
        pytrends = TrendReq(hl='en-US', tz=360)
        pytrends.build_payload([keyword], cat=0, timeframe=timeframe, geo='', gprop='')
        interest_over_time_df = pytrends.interest_over_time()

        if interest_over_time_df.empty:
            return {"message": "No Google Trends data found for this keyword."}
        
        avg_interest = interest_over_time_df[keyword].mean()
        peak_interest_date = interest_over_time_df[keyword].idxmax().strftime('%Y-%m-%d')
        
        return {
            "keyword": keyword,
            "average_interest_score": round(avg_interest, 2),
            "peak_interest_date": peak_interest_date,
            "comment": f"The average interest score is {round(avg_interest, 2)} out of 100 over the last 3 months."
        }
    except Exception as e:
        return {"error": f"Could not retrieve Google Trends data: {e}"}
    
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

def clean_filing_content(text):
    """
    Clean the extracted filing content by removing XBRL/inline tags and normalizing whitespace.
    """
    # Remove all XBRL/inline tags (e.g., xbrli:shares, iso4217:USD, us-gaap:*, etc.)
    text = re.sub(r'\b(xbrli|us-gaap|dei|srt|aapl|intc):[^\s>]+', '', text)

    # Remove standalone tags like "P1Y", "c-1", "f-46", etc.
    text = re.sub(r'\b([A-Za-z]+-[0-9]+|[A-Za-z]+[0-9]+|[A-Z]{2,}[A-Za-z]*\d*)\b', '', text)

    # Remove dates in the format "YYYY-MM-DD" if they are standalone
    text = re.sub(r'(?<!\d)\d{4}-\d{2}-\d{2}(?!\d)', '', text)

    # Remove numbers like "0000050863" if they are standalone
    text = re.sub(r'(?<!\d)\d{8,}(?!\d)', '', text)

    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text).strip()

    return text

@tool
def get_latest_filings(ticker: str, top_n: int = 1) -> list[dict]:
    """
    Fetch latest 10-K / 10-Q filings for the ticker.
    Returns a list of dicts with metadata and cleaned content.
    """
    try:
        # Identify to SEC
        set_identity("jagadeesch1981@gmail.com")
        company = Company(ticker.upper())
        filings = company.get_filings().filter(form=["10-K", "10-Q"])
        filings_df = filings.to_pandas()
        results = []
        for _, filing_row in filings_df.head(top_n).iterrows():
            filing_dict = filing_row.to_dict()
            # Prefer the filing's cik; fallback to company.cik if not present
            cik_raw = filing_row.get("cik") or getattr(company, "cik", None) or filing_row.get("company_info", {}).get("cik")
            if not cik_raw:
                results.append({"error": "No CIK available for this filing.", "filing_meta": filing_dict})
                continue
            cik = str(cik_raw).replace('-', '').zfill(10)  # zero-pad to 10 digits
            accession = str(filing_row.get("accession_number") or filing_row.get("accessionNo") or "").replace('-', '')
            primary_document = filing_row.get("primaryDocument") or filing_row.get("primary_doc") or ""
            # Construct canonical document URL
            document_url = None
            if accession and primary_document:
                document_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{primary_document}"
            # Fallback URLs
            fallback_urls = []
            if filing_row.get("filing_href"):
                fallback_urls.append(filing_row.get("filing_href"))
            if filing_row.get("linkToFilingDetails"):
                fallback_urls.append(filing_row.get("linkToFilingDetails"))
            if filing_row.get("filing_url"):
                fallback_urls.append(filing_row.get("filing_url"))
            # Try primary constructed URL first, then fallbacks
            tried_urls = []
            headers = {"User-Agent": "jagadeesch1981@gmail.com"}
            response_text = None
            for url in ([document_url] if document_url else []) + fallback_urls:
                if not url:
                    continue
                tried_urls.append(url)
                try:
                    resp = requests.get(url, headers=headers, timeout=15)
                    resp.raise_for_status()
                    response_text = resp.text
                    used_url = url
                    break
                except requests.exceptions.HTTPError:
                    continue
                except requests.exceptions.RequestException as e:
                    results.append({
                        "error": f"Network error fetching filing: {e}",
                        "filing_meta": filing_dict,
                        "tried_urls": tried_urls
                    })
                    response_text = None
                    break
            if not response_text:
                results.append({
                    "error": f"Could not fetch filing document (404 or unavailable).",
                    "filing_meta": filing_dict,
                    "tried_urls": tried_urls
                })
                continue
            # Clean the text
            soup = BeautifulSoup(response_text, 'html.parser')
            # Remove all XBRL and XML tags
            for tag in soup.find_all(True, {"name": True}):
                tag.decompose()
            for tag in soup.find_all(True, {"xlink": True}):
                tag.decompose()
            for tag in soup.find_all(True, {"xbrli": True}):
                tag.decompose()
            for tag in soup.find_all(True, {"us-gaap": True}):
                tag.decompose()
            for tag in soup.find_all(True, {"dei": True}):
                tag.decompose()
            for tag in soup.find_all(True, {"srt": True}):
                tag.decompose()
            for tag in soup.find_all(True, {"aapl": True}):
                tag.decompose()
            for tag in soup.find_all(True, {"intc": True}):
                tag.decompose()
            for tag in soup.find_all(True, {"ixt": True}):
                tag.decompose()
            for tag in soup.find_all(True, {"ixt-sec": True}):
                tag.decompose()
            for tag in soup.find_all(True, {"xbrldi": True}):
                tag.decompose()
            for tag in soup.find_all(True, {"xsi": True}):
                tag.decompose()
            for tag in soup.find_all(True, {"iso4217": True}):
                tag.decompose()
            for tag in soup.find_all(True, {"ecd": True}):
                tag.decompose()
            for tag in soup.find_all(True, {"xml": True}):
                tag.decompose()
            # Remove all <head> tags
            if soup.head:
                soup.head.decompose()
            # Get the cleaned text
            cleaned_text = soup.get_text(separator=' ', strip=True)
            # Further clean the text using regex
            cleaned_text = clean_filing_content(cleaned_text)
            results.append({
                "form_type": filing_dict.get("form"),
                "filed_at": str(filing_dict.get("filing_date") or filing_dict.get("filingDate")),
                "accession_number": accession,
                "url": used_url,
                "content": cleaned_text,
                "meta": filing_dict
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

def get_news_analyst_agent(llm: ChatOpenAI):
    """
    Defines a specialist agent that performs a multi-step news analysis chain.
    """
    # This agent only needs one tool: the ability to get raw news headlines.
    news_analyst_tools = [get_stock_news]
    
    # This detailed prompt IS the "Prompt Chain". It instructs the agent on the exact sequence of steps.
    news_analyst_prompt = (
        "You are an expert financial news analyst. Your goal is to produce a concise summary of the latest news sentiment.\n\n"
        "Follow this exact multi-step process:\n"
        "1. **Ingest:** First, use the `get_stock_news` tool to fetch the latest raw news headlines for the given stock ticker.\n"
        "2. **Classify & Analyze:** Internally, classify the sentiment of each headline (Positive, Negative, Neutral) and identify the key topics being discussed (e.g., earnings, partnerships, market trends).\n"
        "3. **Summarize:** Finally, synthesize your findings into a concise, 2-3 sentence summary that captures the overall sentiment and the most important news points.\n\n"
        "Your final output MUST be only the summary paragraph. Do not output the list of headlines or your classification details."
    )
    return create_agent(llm, news_analyst_tools, news_analyst_prompt)

def get_researcher_agent(llm: ChatOpenAI):
    # 1. CREATE the specialist agent and wrap it in a Tool
    news_analyst_agent = get_news_analyst_agent(llm)
    
    def run_news_agent(ticker: str):
        return news_analyst_agent.invoke({"input": ticker})
    
    news_analysis_tool = Tool(
        name="Financial_News_Analyst",
        # The specialist agent's invoke method becomes the tool's function
        func=run_news_agent,
        description="Provides a comprehensive summary of the latest news sentiment for a given stock ticker. The input should be a dictionary with a single key 'input' and the value as the ticker symbol (e.g., {'input': 'AAPL'})."
    )
   
    researcher_tools = [
        read_notes_from_memory, 
        get_company_info, 
        get_price_summary, 
        news_analysis_tool, 
        get_economic_data, 
        get_latest_filings,
        get_financial_ratios,
        get_analyst_ratings,
        get_google_trends
    ]

    # 2. UPDATE the system prompt with a new step for the agent's plan
    researcher_system_prompt = (
        "You are an expert financial researcher. Your primary goal is to produce a detailed analysis paragraph by dynamically adapting your research strategy based on the company's Market Cap and Sector.\n\n"
        "**Execution Plan:**\n"
        "1.  **Consult Memory (MANDATORY FIRST STEP):** Always begin by using the `read_notes_from_memory` tool to gather historical context.\n"
        "2.  **Fetch Company Info & Classify:** Use the `get_company_info` tool to get the company's market cap (`marketCapRaw`) and `sector`.\n"
        "3.  **Dynamic Tool Selection (Two-Factor Approach):** Based on the info gathered, select and prioritize tools as follows:\n\n"
        "    **A. By Market Cap (Primary Filter):**\n"
        "    - **Penny Stock (<$50M):** Your scope is limited. Prioritize `get_price_summary` and `Financial_News_Analyst`. Generally, skip filings and economic data.\n"
        "    - **Mid-Cap ($2B–$10B):** Perform a balanced analysis. Use `get_price_summary`, `Financial_News_Analyst`, and `get_latest_filings`. `get_economic_data` is optional.\n"
        "    - **Large-Cap (>$10B):** Conduct a comprehensive analysis. Use the full suite of tools.\n\n"
        "    **B. By Sector (Secondary Prioritization):** Use the sector to refine your focus and prioritize what to look for with the tools you've selected.\n"
        "    - **Technology or Healthcare:** Pay extremely close attention to `Financial_News_Analyst` for news on innovation, competition, clinical trials, or regulatory changes. In `get_latest_filings`, look for R&D spending.\n"
        "    - **Financials or Industrials:** `get_economic_data` is crucial (e.g., interest rates, GDP). In `get_latest_filings`, focus on balance sheet health and debt.\n"
        "    - **Consumer Cyclical or Consumer Defensive:** `get_economic_data` is very important (e.g., consumer sentiment). Also monitor `Financial_News_Analyst` for supply chain and demand trends.\n"
        "    - **Utilities, Energy, or Real Estate:** Focus on `get_latest_filings` for dividend sustainability and debt. `get_economic_data` is important for interest rate sensitivity.\n\n"
        "    **C. Add Deeper Insight with Specialist Tools:** After the primary analysis, use these tools to add more color:\n"
        "    - **`get_financial_ratios`:** Use this to assess the company's valuation and health. Are they profitable (return_on_equity)? Are they expensive (trailing_pe)?\n"
        "    - **`get_analyst_ratings`:** Check this to see if Wall Street agrees with your assessment. Is there a strong consensus?\n"
        "    - **For consumer-facing companies (e.g., Technology, Consumer Cyclical):** Use `get_google_trends` with the company's name as the keyword to check for public interest trends. Is their brand gaining or losing momentum?\n\n"
        "4.  **Synthesize Final Analysis:** After executing your dynamic research plan, combine all gathered information into a single, detailed analysis paragraph. This paragraph MUST be your final output.\n\n"
        "**Formatting Instructions:** Ensure your final output is a well-formatted, readable paragraph with proper spacing and punctuation. Do not output markdown."
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
        "Your Final, Rewritten Analysis :\n"
        "**Formatting instructions:** Ensure your final output is a well-formatted, readable paragraph with proper header, spacing and punctuation. Do not output markdown."
    )
    return refiner_prompt | llm

