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
    layout="centered",
)

# --- App Title and Description ---
st.sidebar.title("📈 AI Financial Analysis Agent")
st.markdown("""
Welcome to an advanced financial research assistant. Enter a stock ticker and the agent will generate a report.
""")

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
            with st.expander(f"**Past Insights for {ticker}**"):
                for note in memory[ticker]:
                    st.info(note)

# --- Main Streamlit Application Logic ---
def main():
    st.sidebar.divider()
    ticker_symbol = st.sidebar.text_input(
        "Enter a stock ticker symbol:",
        "NVDA",
        help="Try tickers like AAPL, GOOGL, MSFT, NVDA, TSLA."
    ).upper()

    if st.sidebar.button(f"Generate AI Analysis for {ticker_symbol}"):
        if not ticker_symbol:
            st.warning("Please enter a ticker symbol.")
            st.stop()
        
        st.subheader(f"Analysis for {ticker_symbol}")
        display_memory(ticker_symbol)

        st.markdown("#### Historical Price Chart")
        try:
            stock_data = yf.Ticker(ticker_symbol).history(period="2y")
            if stock_data.empty:
                st.warning("Could not retrieve stock data.")
            else:
                st.line_chart(stock_data['Close'], use_container_width=True)
        except Exception as e:
            st.error(f"Error fetching stock data: {e}")
        
        research_placeholder = st.empty()
        critic_placeholder = st.empty()
        info_container = st.container()
        news_container = st.container()
        filings_container = st.container() # New container for filings
        final_report_placeholder = st.empty()
        
        st.sidebar.divider()
        st.sidebar.write('Agent Workflow Status:')
        status_placeholder = st.sidebar.empty()
        
        with st.spinner('🤖 AI Agentic Workflow is running...'):
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
                        
                        with research_placeholder.expander("Research"):
                            
                            intermediate_steps = value.get('research_steps', [])
                            company_info = next((obs for act, obs in intermediate_steps if act.tool == "get_company_info"), None)
                            news_headlines = next((obs for act, obs in intermediate_steps if act.tool == "get_stock_news"), [])
                            # New logic to extract filings data
                            filings_data = next((obs for act, obs in intermediate_steps if act.tool == "get_latest_filings"), [])

                        
                            if company_info:
                                # with info_container:
                                with st.container():
                                    st.markdown("##### Company Information")
                                    st.json(company_info)
                            
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
             st.markdown("#### Final AI-Generated Report")
             st.text(final_analysis)
             # st.info(f"**Learning Update:** {memory_confirmation}")

if __name__ == "__main__":
    main()

