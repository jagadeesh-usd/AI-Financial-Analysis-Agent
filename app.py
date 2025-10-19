import streamlit as st
import yfinance as yf
import pandas as pd
import json
import os

from langchain_core.callbacks import BaseCallbackHandler
from src.chain import build_agentic_workflow
from typing import Any, Dict, List


# --- Page Configuration ---
st.write("")
st.set_page_config(
    page_title="AI Financial Analysis Agent",
    page_icon="📈",
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

# --- Maps tool names to user-friendly text ---
TOOL_DESCRIPTIONS = {
    "read_notes_from_memory": "Consulting Past Insights",
    "get_company_info": "Fetching Company Info",
    "get_price_summary": "Analyzing Price Trends",
    "Financial_News_Analyst": "Analyzing News Sentiment",
    "get_financial_ratios": "Evaluating Financial Ratios",
    "get_latest_filings": "Reviewing SEC Filings",
    "get_analyst_ratings": "Checking Analyst Ratings",
    "get_economic_data": "Assessing Economic Context",
    "get_google_trends": "Analyzing Public Interest",
    "search_specific_news": "Industry Specific News",
    "get_stock_news": "Analyzing Stock Indicators"
}


class StreamlitCallbackHandler(BaseCallbackHandler):
    """A stateful callback handler that correctly attributes events to the active agent,
    supporting the new planner/executor workflow."""

    def __init__(self, progress_bar, status_text_placeholder):
        self.progress_bar = progress_bar
        self.status_text = status_text_placeholder
        
        # Redefined stages for the new 5-step workflow
        self.stages = {
            "planner": 20,
            "executor": 40,
            "critic": 60,
            "refiner": 80,
            "save_memory": 100
        }
        self.completed_stages = set()
        self.active_agent = None

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Called when a main agent node starts. Sets the active agent."""
        if not serialized:
            return

        # This logic for digging into the prompt remains the same
        messages = serialized.get("kwargs", {}).get("messages", [])
        if not messages:
            return

        first_message = messages[0]
        prompt_kwargs = first_message.get("kwargs", {})
        prompt_template = prompt_kwargs.get("prompt", {}).get("kwargs", {}).get("template", "")

        agent_name = None
        # MODIFIED: Updated the keyword matching for the new agents
        if "expert financial planning agent" in prompt_template.lower():
            agent_name = "planner"
        elif "expert financial research execution agent" in prompt_template.lower():
            agent_name = "executor"
        elif "meticulous financial 'critic'" in prompt_template.lower():
            agent_name = "critic"
        elif "'refiner' agent" in prompt_template.lower():
            agent_name = "refiner"
        elif "note-taking assistant" in prompt_template.lower():
            agent_name = "save_memory"

        # This logic for updating the UI remains the same
        if agent_name and agent_name in self.stages and agent_name not in self.completed_stages:
            self.active_agent = agent_name
            progress = self.stages[agent_name]
            self.progress_bar.progress(progress)
            self.status_text.info(f"🧠 **Agent:** {agent_name.capitalize()} started...")
            self.completed_stages.add(agent_name)

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> Any:
        """Called when a tool starts. Only provides updates if the planner or executor is active."""
        # MODIFIED: Allow tool reports for both planner and executor
        if self.active_agent not in ["planner", "executor"]:
            return
            
        tool_name = serialized.get("name", "Unknown tool")
        tool_desc = TOOL_DESCRIPTIONS.get(tool_name, "a specific tool")
        self.status_text.info(f"🛠️ **Tool:** Using {tool_desc}...")

# --- Helper Functions ---
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
                st.info(note.replace('$', '\\$'))


def display_stock_snapshot(company_info, price_summary):
    """
    Renders a visually appealing stock snapshot dashboard using st.metric and columns.
    """
    if not company_info or not price_summary:
        st.warning("Stock data is not available to display snapshot.")
        return
    st.markdown("##### Stock Snapshot")
    # --- Header: Company Name ---
    # st.subheader(f"{company_info.get('longName', 'N/A')}")
    st.metric(label="Latest Price", value=price_summary.get('latest_price', 'N/A'))
    
    # st.markdown("---") # Visual separator

    # --- Key Metrics in Columns ---
    col1, col2 = st.columns(2)
    with col1:
        # Use the raw numerical value for calculations
        market_cap_str = company_info.get('marketCap')

        cleaned_str = market_cap_str.replace('$', '').replace(',', '')
        market_cap_val = float(cleaned_str)
        
        if market_cap_val and market_cap_val > 0:
            if market_cap_val >= 1_000_000_000:
                display_cap = f"${market_cap_val / 1_000_000_000:.2f} B"
            else:
                display_cap = f"${market_cap_val / 1_000_000:.2f} M"
            st.metric(label="Market Cap", value=display_cap)
        else:
            st.metric(label="Market Cap", value="N/A")
            
    with col2:
        low = price_summary.get('52_week_low', 'N/A')
        high = price_summary.get('52_week_high', 'N/A')
        st.metric(label="52-Week Range", value=f"{low} - {high}")

    # st.markdown("---")

    # --- Trend & Momentum Analysis with Context ---
    trend_conclusion = price_summary.get('trend_analysis', {}).get('trend_conclusion', 'N/A')
    rsi_condition = price_summary.get('momentum_analysis', {}).get('condition', 'N/A')
    
    st.markdown(f"**Trend:** {trend_conclusion} | **Momentum:** {rsi_condition}")


def display_research_details(
        reasoning, 
        plan_list,
        company_info, 
        price_summary, 
        news_summary, 
        filings_data, 
        initial_analysis,
        financial_ratios, 
        analyst_ratings, 
        google_trends, 
        economic_data, 
        specific_news
        ):
    """
    Renders the detailed research findings, starting with a visual snapshot
    and followed by deeper insights.
    """
    
    # --- Display Researcher's Plan ---
    if reasoning and plan_list:
        with st.container():
            st.markdown("##### 🕵️‍♂️ Researcher's Plan")
            st.markdown(reasoning)
            # Format the plan with arrows for better visualization
            formatted_plan = [f"➡️ {tool.strip()}" for tool in plan_list]
            st.markdown("\n".join(formatted_plan))

    # --- Display the Visual Stock Snapshot first ---
    display_stock_snapshot(company_info, price_summary)

    # --- Display Deeper Insights ---
    if news_summary:
        with st.container():
            st.markdown("##### News Sentiment Summary")
            summary_text = news_summary.get('output', 'No summary provided.')
            st.info(summary_text.replace('$', '\\$'))
            
    # --- Display Financial Ratios ---
    if financial_ratios and not financial_ratios.get("error"):
        with st.container():
            st.markdown("##### Financial Ratios")
            for key, value in financial_ratios.items():
                st.write(f"**{key.replace('_', ' ').title()}:** {value}")

    # --- Display Analyst Ratings ---
    if analyst_ratings and not analyst_ratings.get("error"):
        with st.container():
            st.markdown("##### Analyst Ratings")
            st.metric(label="Buy Ratings", value=analyst_ratings.get('buy_ratings', 0))
            st.metric(label="Hold Ratings", value=analyst_ratings.get('hold_ratings', 0))
            st.metric(label="Sell Ratings", value=analyst_ratings.get('sell_ratings', 0))

    # --- Display Google Trends ---
    if google_trends and not google_trends.get("error"):
        with st.container():
            st.markdown("##### Public Interest (Google Trends)")
            st.write(f"**Keyword:** {google_trends.get('keyword')}")
            st.write(f"**Average Interest Score:** {google_trends.get('average_interest_score')}/100")
            st.write(f"**Peak Interest Date:** {google_trends.get('peak_interest_date')}")

    # --- Display Economic Data ---
    if economic_data and not economic_data.get("error"):
        with st.container():
            st.markdown("##### Economic Context")
            st.write(f"**Series:** {economic_data.get('series')}")
            st.write(f"**Latest Value:** {economic_data.get('latest_value').replace('$', '\\$')}")
            st.write(f"**Date:** {economic_data.get('latest_date')}")
            
    # --- Display Specific News Search Results ---
    if specific_news and isinstance(specific_news, list):
         with st.container():
            st.markdown("##### Targeted News Search")
            for headline in specific_news:
                frmt_headline = headline.replace('$', '\\$')
                st.write(f"- {frmt_headline}")

    if filings_data and isinstance(filings_data, list) and not pd.DataFrame(filings_data).empty:
        with st.container():
            st.markdown("##### Latest SEC Filings")
            st.dataframe(pd.DataFrame(filings_data))

    if initial_analysis:
        with st.container():
            st.markdown("##### Initial Analysis")
            st.text(initial_analysis)

# --- Sidebar controls ---
st.sidebar.header("AI Financial Analysis")
with st.sidebar.form(key="controls"):
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
        # --- Company Information section ---
        st.markdown("#### Company Information")
        with st.container(height=120, border=True):
            try:
                company_info = yf.Ticker(ticker_symbol).info
                st.write(f"**Name:** {company_info.get('longName', 'N/A')}")
                st.write(f"**Sector:** {company_info.get('sector', 'N/A')}")
                st.write(f"**Industry:** {company_info.get('industry', 'N/A')}")
                st.write(f"**Website:** {company_info.get('website', 'N/A')}")
            except Exception as e:
                st.error(f"Error fetching company info: {e}")

        # --- Historical Price Chart section ---
        st.markdown("#### Historical Price Chart")
        with st.container(height=400, border=True, gap="medium"):
            try:
                st.write("")  # spacer to keep layout consistent
                stock_data = yf.Ticker(ticker_symbol).history(period=history_period)
                if stock_data.empty:
                    st.warning("Could not retrieve stock data. Enter the correct ticker symbol.")
                    st.stop()
                else:
                    st.line_chart(stock_data['Close'])
            except Exception as e:
                st.error(f"Error fetching stock data: {e}")

    with past_col:
        st.markdown(f"#### Past Insights")
        with st.container(height=600, border=True, gap="medium"):
            display_memory(ticker_symbol)

    with agent_col:
        st.markdown(f"#### Analysis for {ticker_symbol}")
        with st.spinner(f'🤖 Analyzing {ticker_symbol} for you...', show_time=True):
            # --- NEW: Setup progress bar and status text for the callback ---
            progress_bar = st.empty()
            status_text_placeholder = st.empty()
            
            with st.container(height=600, border=False, gap="small", width="stretch"):
                final_flex_placeholder = st.empty()
                final_report_placeholder = st.empty()
                research_placeholder = st.empty()
                critic_placeholder = st.empty()
                
                st.sidebar.subheader('Agent Status')

                with st.sidebar.container(border=True):
                    status_placeholder = st.empty()
                    status_placeholder.info("Workflow started!")
                    
                    progress_bar.progress(0)
                    # --- Instantiate and use the callback handler ---
                    streamlit_handler = StreamlitCallbackHandler(progress_bar, status_text_placeholder)
                    config = {"callbacks": [streamlit_handler]}

                    st.latex(r'''\cdots''')

                    with st.expander("##### Researcher's Plan", expanded=True):
                        reason_placeholder = st.empty()
                        plan_placeholder = st.empty() # A single placeholder for the entire plan
                    chosen_tools = [] # A list to track the tools the agent decides to use
                    
                    with st.expander("##### Workflow Agents", expanded=True):
                        plan_status_placeholder = st.empty()
                        research_status_placeholder = st.empty()
                        critic_status_placeholder = st.empty()
                        refine_status_placeholder = st.empty()

                        memory_status_placeholder = st.empty()

                    st.latex(r'''\cdots''')

                agent_workflow = build_agentic_workflow()
                inputs = {"ticker": ticker_symbol}
                
                final_analysis = "Analysis could not be generated."
                memory_confirmation = "Key insights will be saved upon completion."

                for event in agent_workflow.stream(inputs, config=config):
                    for key, value in event.items():
                        # --- NEW: Handle the 'planner' node ---
                        if key == "planner":
                            plan_status_placeholder.info("🕵️‍♂️ **Planner Agent:** Creating research plan...")
                            st.session_state['reasoning'] = value.get('reasoning', 'No reasoning generated.')
                            st.session_state['plan_list']  = value.get('plan', 'No plan generated.')
                            st.session_state['company_info'] = value.get('company_info', 'No company info generated.')
                            # Format the plan for display in the sidebar
                            reason_placeholder.markdown(st.session_state['reasoning'])
                            plan_list = [f"{i+1}. {tool.strip()}" for i, tool in enumerate(st.session_state['plan_list'])]
                            plan_placeholder.markdown("\n".join(plan_list))
                            plan_status_placeholder.success("🕵️‍♂️ **Planner Agent:** Creating research plan complete.")
                            research_status_placeholder.info("🛠️ **Executor Agent:** Executing research plan...")

                        elif key == "executor":                            
                            intermediate_steps = value.get('research_steps', [])
                            
                            # Parse tool outputs and save to session state
                            st.session_state['price_summary']  = next((obs for act, obs in intermediate_steps if act.tool == "get_price_summary"), None)
                            st.session_state['news_summary_output'] = next((obs for act, obs in intermediate_steps if act.tool == "Financial_News_Analyst"), None)
                            st.session_state['filings_data'] = next((obs for act, obs in intermediate_steps if act.tool == "get_latest_filings"), [])
                            st.session_state['financial_ratios'] = next((obs for act, obs in intermediate_steps if act.tool == "get_financial_ratios"), None)
                            st.session_state['analyst_ratings'] = next((obs for act, obs in intermediate_steps if act.tool == "get_analyst_ratings"), None)
                            st.session_state['google_trends'] = next((obs for act, obs in intermediate_steps if act.tool == "get_google_trends"), None)
                            st.session_state['economic_data'] = next((obs for act, obs in intermediate_steps if act.tool == "get_economic_data"), None)
                            st.session_state['specific_news'] = next((obs for act, obs in intermediate_steps if act.tool == "search_specific_news"), None)
                            
                            if "initial_analysis" in value:
                                st.session_state['initial_analysis'] = value["initial_analysis"]

                            # This temporary expander can still be used for debugging during the run
                            with research_placeholder.expander("🔬 Research Trail", expanded=False):
                                display_research_details(
                                    st.session_state.get('reasoning'),
                                    st.session_state.get('plan_list'),
                                    st.session_state.get('company_info'),
                                    st.session_state.get('price_summary'),
                                    st.session_state.get('news_summary_output'),
                                    st.session_state.get('filings_data'),
                                    st.session_state.get('initial_analysis'),
                                    st.session_state.get('financial_ratios'),
                                    st.session_state.get('analyst_ratings'),
                                    st.session_state.get('google_trends'),
                                    st.session_state.get('economic_data'),
                                    st.session_state.get('specific_news')
                                )
                            # Signal completion of the execution phase
                            research_status_placeholder.success("🛠️ **Execution Agent:** Research complete.")

                        # --- Handle 'critic' node after executor ---
                        elif key == "critic":
                            critic_status_placeholder.warning("🧐 **Critic Agent:** Evaluating the initial analysis...")
                            if "critique" in value:
                                st.session_state['critique'] = value["critique"]
                                with critic_placeholder.expander('🧐 critique', expanded=False):
                                    st.markdown(st.session_state['critique'])

                        # --- Refiner and Save Memory (no changes here) ---
                        elif key == "refiner":
                            critic_status_placeholder.success("🧐 **Critic Agent:** Evaluation complete.")
                            refine_status_placeholder.info("✍️ **Refiner Agent:** Rewriting analysis...")
                            if "refined_analysis" in value:
                                final_analysis = value["refined_analysis"]
                        
                        elif key == "save_memory":
                            refine_status_placeholder.success("✍️ **Refiner Agent:** Analysis rewritten.")
                            if "memory_confirmation" in value:
                                memory_confirmation = value["memory_confirmation"]

                status_placeholder.success("Workflow Complete!")
                progress_bar.progress(100) 

            with final_flex_placeholder.container(horizontal=True, height="content", horizontal_alignment="right"):
                with st.popover('🔬 Research Trail'):
                    display_research_details(
                                    st.session_state.get('reasoning'),
                                    st.session_state.get('plan_list'),
                                    st.session_state.get('company_info'),
                                    st.session_state.get('price_summary'),
                                    st.session_state.get('news_summary_output'),
                                    st.session_state.get('filings_data'),
                                    st.session_state.get('initial_analysis'),
                                    st.session_state.get('financial_ratios'),
                                    st.session_state.get('analyst_ratings'),
                                    st.session_state.get('google_trends'),
                                    st.session_state.get('economic_data'),
                                    st.session_state.get('specific_news')
                                    )
                if st.session_state['critique']:
                    with st.popover('🧐 Critique'):
                        st.markdown(st.session_state['critique'])
            
            with final_report_placeholder.container(height=600,width="stretch", border=False):
                st.markdown(final_analysis.replace('$', '\\$'))
                st.latex(r'''\cdots''')
                research_placeholder.empty()
                critic_placeholder.empty()
                progress_bar.empty()
                status_text_placeholder.empty()
            
                # st.info(f"**Learning Update:** {memory_confirmation}")

if __name__ == "__main__":
    main()