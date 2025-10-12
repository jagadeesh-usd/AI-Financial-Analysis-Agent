import streamlit as st
import yfinance as yf
import pandas as pd
import json
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.chain import build_agentic_workflow

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Financial Analysis Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Page Styling: center + sizing for titles & headers ---
st.markdown(
    """
    <style>
    /* General font family for a cleaner look */
    .css-1d391kg { font-family: 'Helvetica Neue', Arial, sans-serif; }

    /* Center and style primary page title (st.title) */
    h1 {
        text-align: center !important;
        color: #0b57a4 !important;
        font-size: 2.4rem !important;
        font-weight: 800 !important;
        margin-top: 0.6rem !important;
        margin-bottom: 0.2rem !important;
    }

    /* Center and style subheaders (st.header / st.subheader) */
    h2, h3 {
        text-align: center !important;
        color: #123e6a !important;
        font-weight: 700 !important;
        margin-top: 0.6rem !important;
        margin-bottom: 0.4rem !important;
    }

    /* Markdown headings (#### -> h4, etc.) used in your layout */
    [data-testid="stMarkdownContainer"] h4,
    h4 {
        text-align: center !important;
        color: #1f6fb2 !important;
        font-size: 1.25rem !important;
        font-weight: 700 !important;
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }

    /* Slightly smaller centered headings (##### -> h5) */
    [data-testid="stMarkdownContainer"] h5,
    h5 {
        text-align: center !important;
        color: #255a8a !important;
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        margin-top: 0.35rem !important;
        margin-bottom: 0.35rem !important;
    }

    /* Center normal paragraph markdown where used intentionally */
    .centered-paragraph {
        text-align: center;
        font-size: 16px;
        color: #1a2938;
    }

    /* Center alerts / info boxes content */
    .stAlert > div {
        text-align: center;
    }

    /* Small tweak for containers to visually balance centered headers */
    .block-container {
        padding-top: 1rem;
    }

    /* Keep sidebar title left-aligned (default) but slightly larger */
    .css-1v3fvcr h1, .css-1v3fvcr h2 {
        font-size: 1.1rem !important;
    }

    /* Avoid changing layout on mobile too drastically */
    @media (max-width: 600px) {
        h1 { font-size: 1.8rem !important; }
        [data-testid="stMarkdownContainer"] h4 { font-size: 1.05rem !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- App Title and Description ---
st.title("📈 AI Financial Analysis Agent")
st.markdown(
    "<p class='centered-paragraph'>Welcome to an advanced financial research assistant.</p>",
    unsafe_allow_html=True,
)

# --- OpenAI Client Initialization ---
try:
    # Ensure the model is robust enough for agentic work
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=st.secrets["OPENAI_API_KEY"], temperature=0)
except Exception:
    st.error("OpenAI API key not found. Please add it to your Streamlit secrets.", icon="🚨")
    st.stop()

# --- Helper Functions ---
def analyze_news_sentiment(news_headlines: list, llm: ChatOpenAI):
    """Uses an LLM to classify news headlines as Positive, Negative, or Neutral."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert sentiment analysis AI. Classify financial news headlines."),
        ("human", """
        Please classify the sentiment of each headline below as 'Positive', 'Negative', or 'Neutral'.
        Return the result as a JSON object with a single key "sentiments", which is a list of objects.
        Each object should have 'headline' and 'sentiment' keys.

        Headlines:
        {headlines}
        """)
    ])
    sentiment_chain = prompt | llm
    try:
        response_content = sentiment_chain.invoke({"headlines": json.dumps(news_headlines, indent=2)}).content
        cleaned_response = response_content.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned_response).get("sentiments", [])
    except (json.JSONDecodeError, AttributeError):
        return []

def display_memory(ticker: str):
    """Reads and displays the agent's memory for a given ticker."""
    MEMORY_FILE = "memory.json"
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, 'r') as f:
            try:
                memory = json.load(f)
            except json.JSONDecodeError:
                memory = {} # Handle empty or corrupt file
        
        if ticker in memory and memory[ticker]:
            # with st.expander(f"**{ticker}**", expanded=True):
            for note in memory[ticker]:
                st.info(note)

# --- Sidebar controls ---
with st.sidebar.form(key="controls"):
    st.title("AI Financial Analysis")
    ticker_symbol = st.text_input(
        "Ticker symbol",
        value="NVDA",
        max_chars=8,
        help="Enter a valid stock ticker, e.g. AAPL, MSFT, GOOGL, NVDA",
    ).upper()

    history_period = st.selectbox(
        "Historical range",
        options=["6mo", "1y", "2y", "5y"],
        index=2,
        help="Select the lookback window for the price chart",
    )

    run_button = st.form_submit_button("Generate analysis")
    st.caption("Pro tip: use a ticker from the suggestions to avoid lookup errors.")

if not run_button:
    st.info("Enter a ticker and click 'Generate analysis' to begin.")
    st.stop()

if not ticker_symbol:
    st.warning("Please provide a ticker symbol.")
    st.stop()

# --- Main Streamlit Application Logic ---
def main():
    
    static_info_col, agent_col, past_col = st.columns([0.5, 1, 0.5])

    with static_info_col:
        st.markdown("#### Historical Price Chart")
        with st.container(height=500, border=True, gap="medium"):
            try:
                st.write("")  # spacer to keep layout consistent
                stock_data = yf.Ticker(ticker_symbol).history(period=history_period)
                if stock_data.empty:
                    st.warning("Could not retrieve stock data. Enter the correct ticker symbol.")
                    st.stop()
                else:
                    st.line_chart(stock_data['Close'], use_container_width=True)
            except Exception as e:
                st.error(f"Error fetching stock data: {e}")

    with past_col:
        st.markdown(f"#### Past Insights")
        with st.container(height=500, border=True, gap="medium"):
            display_memory(ticker_symbol)

    with agent_col:
        st.markdown(f"#### Analysis for {ticker_symbol}")
        with st.container(height=500, border=True, gap="medium"):
            with st.spinner('🤖 AI Agentic Workflow is running...'):
                research_placeholder = st.empty()
                critic_placeholder = st.empty()
                final_report_placeholder = st.empty()
                
                st.sidebar.write('Agent Workflow Status:')
                status_placeholder = st.sidebar.empty()

                status_placeholder.info("Workflow started!")
                
                agent_workflow = build_agentic_workflow()
                inputs = {"ticker": ticker_symbol}
                
                final_analysis = "Analysis could not be generated."
                memory_confirmation = "Key insights will be saved upon completion."
                
                
                research_status_placeholder = st.sidebar.empty()
                critic_status_placeholder = st.sidebar.empty()
                refine_status_placeholder = st.sidebar.empty()
                memory_status_placeholder = st.sidebar.empty()

                for event in agent_workflow.stream(inputs):
                    for key, value in event.items():
                        if key == "researcher":
                            research_status_placeholder.info("🕵️‍♂️ **Researcher Agent:** Executing research...")
                            
                            with research_placeholder.expander("Research", expanded=False):
                                
                                intermediate_steps = value.get('research_steps', [])
                                company_info = next((obs for act, obs in intermediate_steps if act.tool == "get_company_info"), None)
                                price_summary = next((obs for act, obs in intermediate_steps if act.tool == "get_price_summary"), None)
                                news_headlines = next((obs for act, obs in intermediate_steps if act.tool == "get_stock_news"), [])
                                # New logic to extract filings data
                                filings_data = next((obs for act, obs in intermediate_steps if act.tool == "get_latest_filings"), [])

                            
                                if company_info:
                                    # with info_container:
                                    with st.container():
                                        st.markdown("##### Company Information")
                                        st.json(company_info)

                                # ADD a new block to display the price summary
                                if price_summary:
                                    with st.container():
                                        st.markdown("##### 30-Day Price Summary")
                                        st.json(price_summary)
                                
                                if news_headlines and isinstance(news_headlines, list):
                                    #with news_container:
                                    with st.container():
                                        st.markdown("##### Recent News & AI Sentiment")
                                        sentiments = analyze_news_sentiment(news_headlines, llm)
                                        if sentiments:
                                            st.dataframe(pd.DataFrame(sentiments), use_container_width=True)
                                        else:
                                            st.write("Could not generate sentiment analysis.")
                            
                                # New logic to display filings data
                                if filings_data and isinstance(filings_data, list):
                                    # with filings_container:
                                    with st.container():
                                        st.markdown("##### Latest SEC Filings (10-K & 10-Q)")
                                        st.dataframe(pd.DataFrame(filings_data), use_container_width=True)

                                if "initial_analysis" in value:
                                    st.markdown("##### Initial Analysis")
                                    st.text(value["initial_analysis"])
                                

                        elif key == "critic":
                            research_status_placeholder.success("🕵️‍♂️ **Researcher Agent:** Research complete.")
                            critic_status_placeholder.warning("🧐 **Critic Agent:** Evaluating the initial analysis...")
                            with critic_placeholder.expander('critique'):
                                if "critique" in value:
                                    st.markdown(value["critique"])

                        elif key == "refiner":
                            critic_status_placeholder.success("🧐 **Critic Agent:** Evaluation complete.")
                            refine_status_placeholder.info("✍️ **Refiner Agent:** Rewriting analysis...")
                            if "refined_analysis" in value:
                                final_analysis = value["refined_analysis"]
                        
                        elif key == "save_memory":
                            refine_status_placeholder.success("✍️ **Refiner Agent:** Analysis rewritten.")
                            # memory_status_placeholder.info("💾 **Memory Agent:** Saving key insights...")
                        
                            if "memory_confirmation" in value:
                                memory_confirmation = value["memory_confirmation"]
                                

                status_placeholder.success("Workflow Complete!")

            with final_report_placeholder.container():
                    st.markdown("#### Final Report")
                    with st.container(border=True, height=300):
                        st.text(final_analysis)
                # st.info(f"**Learning Update:** {memory_confirmation}")

if __name__ == "__main__":
    main()