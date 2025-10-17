import streamlit as st
import yfinance as yf
import pandas as pd
import json
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
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

# --- Streamlit Callback Handler ---
# class StreamlitCallbackHandler(BaseCallbackHandler):
#     """A custom callback handler that updates the Streamlit UI in real-time."""

#     def __init__(self, progress_bar, status_text_placeholder):
#         self.progress_bar = progress_bar
#         self.status_text = status_text_placeholder
#         self.progress = 0
#         self.total_steps = 1 + len(TOOL_DESCRIPTIONS) + 3 
#         self.main_agent_nodes = ["researcher", "critic", "refiner", "save_memory"]

#     def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> Any:
#         """Called when a chain (agent) is about to run."""
        
#         # --- FIX: Add a guard clause to handle NoneType ---
#         if not serialized:
#             return # Ignore events that don't have a serialized dictionary

#         agent_name = serialized.get("id", ["Unknown agent"])[-1]
        
#         if agent_name not in self.main_agent_nodes:
#             return # Ignore internal chains like ChatPromptTemplate

#         self.progress += 1
#         progress_percent = min(100, int((self.progress / self.total_steps) * 100))
#         self.progress_bar.progress(progress_percent)
        
#         self.status_text.info(f"🧠 **Agent:** {agent_name.capitalize()} started...")

#     def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> Any:
#         """Called when a tool is about to be run."""
#         self.progress += 1
#         progress_percent = min(100, int((self.progress / self.total_steps) * 100))
#         self.progress_bar.progress(progress_percent)

#         tool_name = serialized.get("name", "Unknown tool")
#         tool_desc = TOOL_DESCRIPTIONS.get(tool_name, tool_name)
#         self.status_text.info(f"🛠️ **Tool:** Using {tool_desc}...")

# In app.py

# In app.py, replace your existing StreamlitCallbackHandler


# class StreamlitCallbackHandler(BaseCallbackHandler): 
#     """A custom callback handler that reliably tracks and updates the Streamlit UI
#     for major agentic workflow stages."""

#     def __init__(self, progress_bar, status_text_placeholder):
#         self.progress_bar = progress_bar
#         self.status_text = status_text_placeholder
        
#         # Define the stages and their corresponding progress percentages
#         self.stages = {
#             "researcher": 15,
#             "researcher": 25,
#             "critic": 50,
#             "refiner": 75,
#             "save_memory": 100
#         }
#         # Keep track of which main stages we have already logged to avoid duplicates
#         self.completed_stages = set()

#     def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any):
#         for stage in self.stages:
#             if stage in prompts[0].lower() and stage not in self.completed_stages:
#                 print(stage)
#                 progress = self.stages[stage]
#                 self.progress_bar.progress(progress)
#                 self.status_text.info(f"🧠 **Agent:** {stage.capitalize()} started...")
#                 self.completed_stages.add(stage)


#     def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> Any:
#         """Called when a tool is about to be run. Provides granular updates for the status text only."""
#         tool_name = serialized.get("name", "Unknown tool")
#         print(tool_name)
#         tool_desc = TOOL_DESCRIPTIONS.get(tool_name, "a specific tool")
        
#         # This provides the real-time feedback on which tool is being used
#         # It does NOT affect the main progress bar.
#         self.status_text.info(f"🛠️ **Tool:** Using {tool_desc}...")



# class StreamlitCallbackHandler(BaseCallbackHandler):
#     """
#     Stateful Streamlit callback handler that:
#       - reliably detects the active agent/role from serialized chain objects or LLM prompts
#       - sets a current active_agent so on_tool_start messages are attributed correctly
#       - updates a progress bar once per major stage (researcher/critic/refiner/save_memory)
#       - shows friendly "Using ..." messages when prompts include that pattern
#       - provides optional debug printing when debug=True
#     """

#     def __init__(self, progress_bar, status_text_placeholder, debug: bool = False):
#         self.progress_bar = progress_bar
#         self.status_text = status_text_placeholder

#         # main stages mapping -> percent
#         self.stages = {
#             "researcher": 25,
#             "critic": 50,
#             "refiner": 75,
#             "save_memory": 100,
#         }

#         # track which stages already reported
#         self.completed_stages = set()

#         # transient: currently active role key (e.g., "researcher")
#         self.active_agent: Optional[str] = None
#         self.active_agent_display: Optional[str] = None

#         # debug toggle
#         self.debug = debug

#     # -------------------------
#     # Helpers for detecting role
#     # -------------------------
#     def _collect_template_strings(self, obj: Any, found: List[str]) -> None:
#         """
#         Recursively walk the serialized `obj` (dict/list structure) and collect
#         any string values that look like prompt templates (keys named 'template', or strings nested inside).
#         """
#         if isinstance(obj, dict):
#             for k, v in obj.items():
#                 # common keys that contain template/prompt text
#                 if isinstance(k, str) and "template" in k.lower() and isinstance(v, str):
#                     found.append(v)
#                 # some structures store prompt text in nested kwargs.prompt.kwargs.template
#                 if k == "template" and isinstance(v, str):
#                     found.append(v)
#                 else:
#                     self._collect_template_strings(v, found)
#         elif isinstance(obj, (list, tuple)):
#             for item in obj:
#                 self._collect_template_strings(item, found)
#         elif isinstance(obj, str):
#             # occasionally top-level strings are present
#             found.append(obj)
#         # ignore other scalar types

#     def _detect_role_from_templates(self, templates: List[str]) -> Optional[str]:
#         """
#         Look through collected template strings for known role indicators and return stage key.
#         """
#         if not templates:
#             return None
#         combined = " ".join(templates).lower()

#         # patterns that indicate each role. Add or tune these to match your exact prompts.
#         patterns = {
#             "researcher": [
#                 r"expert financial researcher",
#                 r"you are an expert financial researcher",
#                 r"consult memory \(mandatory first step\)",
#                 r"dynamic tool selection",
#                 r"market cap based plans",
#             ],
#             "critic": [
#                 r"critic agent",
#                 r"meticulous financial 'critic'",
#                 r"evaluate an analysis",
#                 r"initial analysis to critique",
#             ],
#             "refiner": [
#                 r"refiner agent",
#                 r"rewrite and improve an initial financial analysis",
#                 r"final, rewritten analysis",
#             ],
#             "save_memory": [
#                 r"note-taking assistant",
#                 r"single-sentence takeaway",
#                 r"saved for future reference",
#             ],
#         }

#         for role_key, pats in patterns.items():
#             for pat in pats:
#                 if re.search(pat, combined, flags=re.IGNORECASE):
#                     if self.debug:
#                         print(f"[debug] detected role '{role_key}' via pattern '{pat}'")
#                     return role_key
#         return None

#     def _extract_using_action(self, templates: List[str]) -> Optional[str]:
#         """
#         Attempt to find a 'Using ...' action line in templates. Returns the short action text or None.
#         """
#         if not templates:
#             return None
#         combined = " ".join(templates)
#         m = re.search(r"(?mi)\b(using\s+[^.\n]+)", combined)
#         if m:
#             action = m.group(1).strip().rstrip(".")
#             return action
#         # Also try to find explicit "### Role: <RoleName>" followed by a first line action
#         # e.g. "### Role: Researcher\nUsing Reviewing SEC Filings..."
#         for t in templates:
#             m2 = re.search(r"(?mi)###\s*Role:\s*(.+)\n([^\n]+)", t)
#             if m2:
#                 candidate = m2.group(2).strip()
#                 if candidate.lower().startswith("using"):
#                     return candidate.rstrip(".")
#         return None

#     # -------------------------
#     # Callback handlers
#     # -------------------------
#     def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
#         """
#         Called when a chain/agent node starts. We try to detect which role is starting by:
#           1. scanning serialized for prompt template strings
#           2. matching those templates to known role patterns
#           3. falling back to serialized['id'] last element or serialized.get('name')
#         If detected, set self.active_agent and update progress/status once for that role.
#         """
#         if not serialized:
#             return

#         if self.debug:
#             print("[debug] on_chain_start serialized:", serialized)

#         # 1) collect templates/strings from serialized
#         templates: List[str] = []
#         try:
#             self._collect_template_strings(serialized, templates)
#         except Exception as e:
#             if self.debug:
#                 print("[debug] _collect_template_strings error:", e)

#         # 2) try detect role from templates
#         detected_role = self._detect_role_from_templates(templates)

#         # 3) fallback: check serialized['id'] last element and name fields
#         if not detected_role:
#             try:
#                 sid = serialized.get("id")
#                 if isinstance(sid, (list, tuple)) and sid:
#                     cand = str(sid[-1]).lower()
#                     if cand in self.stages:
#                         detected_role = cand
#                 if not detected_role:
#                     name = serialized.get("name") or serialized.get("__name__")
#                     if isinstance(name, str) and name.lower() in self.stages:
#                         detected_role = name.lower()
#             except Exception:
#                 detected_role = None

#         # If still not detected, do not change active_agent (avoid false attribution)
#         if not detected_role:
#             if self.debug:
#                 print("[debug] on_chain_start: could not detect role. templates:", templates, "serialized id:", serialized.get("id"))
#             return

#         # Set active agent state
#         self.active_agent = detected_role
#         self.active_agent_display = detected_role.capitalize()

#         # Update progress once per stage
#         if detected_role in self.stages and detected_role not in self.completed_stages:
#             try:
#                 self.progress_bar.progress(self.stages[detected_role])
#             except Exception:
#                 # streamlit widget might have been removed; ignore
#                 pass
#             self.completed_stages.add(detected_role)

#         # Compose a friendly message using any "Using ..." action if present
#         action_text = self._extract_using_action(templates)
#         if action_text:
#             msg = f"🧠 **{self.active_agent_display}:** {action_text}..."
#         else:
#             msg = f"🧠 **{self.active_agent_display}:** started..."

#         try:
#             self.status_text.info(msg)
#         except Exception:
#             # ignore UI errors
#             if self.debug:
#                 print("[debug] Failed to write status_text.info")

#     def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
#         """
#         Robust fallback: if on_chain_start wasn't emitted or didn't detect role, try parsing the LLM prompts.
#         This supports prompts that use the '### Role:' header pattern or contain 'Using ...'.
#         """
#         if not prompts:
#             return

#         first_prompt = prompts[0]
#         if self.debug:
#             print("[debug] on_llm_start prompt:", first_prompt)

#         # attempt to parse explicit "### Role:" header
#         header_match = re.search(r"###\s*Role:\s*(.+)", first_prompt)
#         role_name = header_match.group(1).strip() if header_match else None

#         # attempt to find "Using ..." action
#         action_match = re.search(r"(?mi)^\s*(using\s+.+?)(?:[.\n]|$)", first_prompt)
#         action_text = action_match.group(1).strip().rstrip(".") if action_match else None

#         # If a role header exists, set active_agent/context accordingly
#         if role_name:
#             role_key = role_name.lower()
#             self.active_agent = role_key
#             self.active_agent_display = role_name

#             if role_key in self.stages and role_key not in self.completed_stages:
#                 try:
#                     self.progress_bar.progress(self.stages[role_key])
#                 except Exception:
#                     pass
#                 self.completed_stages.add(role_key)

#             if action_text:
#                 msg = f"🧠 **{self.active_agent_display}:** {action_text}..."
#             else:
#                 msg = f"🧠 **{self.active_agent_display}:** working on..."

#             try:
#                 self.status_text.info(msg)
#             except Exception:
#                 if self.debug:
#                     print("[debug] Could not write status_text in on_llm_start")

#             return

#         # If no explicit header, we can still try to detect using patterns from the prompt body
#         # Reuse template detection by treating the single prompt as a template
#         detected = self._detect_role_from_templates([first_prompt])
#         if detected:
#             self.active_agent = detected
#             self.active_agent_display = detected.capitalize()
#             if detected in self.stages and detected not in self.completed_stages:
#                 try:
#                     self.progress_bar.progress(self.stages[detected])
#                 except Exception:
#                     pass
#                 self.completed_stages.add(detected)

#             if action_text:
#                 msg = f"🧠 **{self.active_agent_display}:** {action_text}..."
#             else:
#                 msg = f"🧠 **{self.active_agent_display}:** working on..."
#             try:
#                 self.status_text.info(msg)
#             except Exception:
#                 if self.debug:
#                     print("[debug] Could not write status_text in on_llm_start (detected)")

#     def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
#         """
#         When a tool starts, attribute the tool to the currently active_agent (if any).
#         If no active_agent, still show the tool but without the role suffix.
#         """
#         tool_name = serialized.get("name", "Unknown tool") if isinstance(serialized, dict) else str(serialized)
#         tool_desc = TOOL_DESCRIPTIONS.get(tool_name, tool_name.replace("_", " ").capitalize())

#         if self.active_agent_display:
#             msg = f"🛠️ **Tool:** Using {tool_desc}... ({self.active_agent_display})"
#         else:
#             msg = f"🛠️ **Tool:** Using {tool_desc}..."

#         try:
#             self.status_text.info(msg)
#         except Exception:
#             if self.debug:
#                 print("[debug] Could not write status_text in on_tool_start")

#     def on_chain_end(self, *args, **kwargs) -> None:
#         """
#         Robust handler for chain end events.

#         Accepts flexible argument signatures because different LangChain components
#         call this hook with different parameters. We try to extract `serialized`
#         and `outputs` when available, log a completion message for the active agent,
#         and clear transient state so subsequent tool calls are not misattributed.
#         """
#         serialized = None
#         outputs = None

#         # 1) positional args: common shapes:
#         #    (serialized, outputs), (outputs), (serialized)
#         try:
#             if len(args) == 2:
#                 serialized, outputs = args[0], args[1]
#             elif len(args) == 1:
#                 # Ambiguous: could be serialized OR outputs. Try to heuristically decide.
#                 single = args[0]
#                 # If it looks like an outputs dict (has typical LLM keys), treat as outputs
#                 if isinstance(single, dict) and any(k in single for k in ("output", "outputs", "generations", "text", "result")):
#                     outputs = single
#                 else:
#                     serialized = single
#             # else: no positional args
#         except Exception:
#             serialized = None
#             outputs = None

#         # 2) keyword args - prefer explicit
#         if "outputs" in kwargs:
#             outputs = kwargs.get("outputs")
#         if "serialized" in kwargs:
#             serialized = kwargs.get("serialized")

#         # (optional) debug printing
#         if getattr(self, "debug", False):
#             try:
#                 print("[debug] on_chain_end called. serialized:", type(serialized), "outputs keys:", (list(outputs.keys()) if isinstance(outputs, dict) else type(outputs)))
#             except Exception:
#                 pass

#         # Compose and show completion message if we have an active agent
#         try:
#             if getattr(self, "active_agent_display", None):
#                 # include a tiny summary if outputs looks like it contains text (avoid big dumps)
#                 summary_suffix = ""
#                 if isinstance(outputs, dict):
#                     # try to find a short textual key to include (safe, small)
#                     for key in ("output", "text", "result", "final_answer"):
#                         if key in outputs and isinstance(outputs[key], str):
#                             short = outputs[key].strip().split("\n")[0][:200]
#                             summary_suffix = f" — {short}..." if short else ""
#                             break
#                     # some AgentExecutors return nested keys
#                     if not summary_suffix:
#                         for k, v in outputs.items():
#                             if isinstance(v, str) and len(v) < 300:
#                                 summary_suffix = f" — {v.strip().splitlines()[0][:200]}..."
#                                 break

#                 try:
#                     self.status_text.info(f"✅ **{self.active_agent_display}:** finished{summary_suffix}")
#                 except Exception:
#                     # UI may be unavailable; ignore but optionally debug
#                     if getattr(self, "debug", False):
#                         print("[debug] Failed to write completion message to status_text")
#         except Exception:
#             # swallow any UI exceptions to avoid crashing the app
#             if getattr(self, "debug", False):
#                 import traceback
#                 traceback.print_exc()

#         # Always clear transient active agent state so next tool calls won't be misattributed
#         try:
#             self.active_agent = None
#             self.active_agent_display = None
#         except Exception:
#             pass

#     # Optionally implement on_llm_end if you want to mark LLM-completion specifically
#     def on_llm_end(self, response: Dict[str, Any], **kwargs: Any) -> None:
#         # this is a lightweight hook - we won't change progress here but we can clear active_agent_display if desired
#         # keep active_agent to be cleared by on_chain_end to ensure chain-level end is what resets attribution
#         if self.debug:
#             print("[debug] on_llm_end called")

class StreamlitCallbackHandler(BaseCallbackHandler):
    """A stateful callback handler that correctly attributes events to the active agent."""
    def __init__(self, progress_bar, status_text_placeholder):
        self.progress_bar = progress_bar
        self.status_text = status_text_placeholder
        self.stages = {
            "researcher": 25,
            "critic": 50,
            "refiner": 75,
            "save_memory": 100
        }
        self.completed_stages = set()
        self.active_agent = None

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Called when a main agent node starts. Sets the active agent."""
        if not serialized:
            return

        messages = serialized.get("kwargs", {}).get("messages", [])
        if not messages:
            return

        first_message = messages[0]
        prompt_kwargs = first_message.get("kwargs", {})
        prompt_template = prompt_kwargs.get("prompt", {}).get("kwargs", {}).get("template", "")

        agent_name = None
        if "expert financial researcher" in prompt_template.lower():
            agent_name = "researcher"
        elif "meticulous financial 'critic'" in prompt_template.lower():
            agent_name = "critic"
        elif "'refiner' agent" in prompt_template.lower():
            agent_name = "refiner"
        elif "note-taking assistant" in prompt_template.lower():
            agent_name = "save_memory"

        if agent_name and agent_name in self.stages and agent_name not in self.completed_stages:
            self.active_agent = agent_name
            progress = self.stages[agent_name]
            self.progress_bar.progress(progress)
            self.status_text.info(f"🧠 **Agent:** {agent_name.capitalize()} started...")
            self.completed_stages.add(agent_name)

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> Any:
        """Called when a tool starts. Only provides updates if the researcher is active."""
        if self.active_agent != "researcher":
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


def display_research_details(company_info, price_summary, news_summary, filings_data, initial_analysis):
    """
    Renders the detailed research findings, starting with a visual snapshot
    and followed by deeper insights.
    """
    
    # --- 1. Display the Visual Stock Snapshot first ---
    display_stock_snapshot(company_info, price_summary)
        
    # --- 2. Display Deeper Insights ---
    if news_summary:
        with st.container():
            st.markdown("##### News Sentiment Summary")
            summary_text = news_summary.get('output', 'No summary provided.')
            st.info(summary_text.replace('$', '\\$'))

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
                    # --- NEW: Instantiate and use the callback handler ---
                    streamlit_handler = StreamlitCallbackHandler(progress_bar, status_text_placeholder)
                    config = {"callbacks": [streamlit_handler]}

                    st.latex(r'''\cdots''')

                    with st.expander("##### Researcher's Plan", expanded=True):
                        plan_placeholder = st.empty() # A single placeholder for the entire plan
                    chosen_tools = [] # A list to track the tools the agent decides to use
                    
                    with st.expander("##### Workflow Agents", expanded=True):
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
                        if key == "researcher":
                            intermediate_steps = value.get('research_steps', [])

                            chosen_tools = [] 
                            if intermediate_steps:
                                # Loop through ALL tool calls in the event, not just the last one
                                for step in intermediate_steps:
                                    action = step[0] # The first element is the AgentAction
                                    tool_name = action.tool
                                    if tool_name in TOOL_DESCRIPTIONS:
                                        tool_desc = TOOL_DESCRIPTIONS[tool_name]
                                        if tool_desc not in chosen_tools:
                                            chosen_tools.append(tool_desc)
                            
                            # Update the sidebar placeholder with the complete, ordered plan
                            if chosen_tools:
                                plan_markdown = ""
                                for i, tool in enumerate(chosen_tools):
                                    plan_markdown += f"{i+1}. {tool}\n"
                                plan_placeholder.markdown(plan_markdown)

                            research_status_placeholder.info("🕵️‍♂️ **Researcher Agent:** Executing research...")
                            
                            with research_placeholder.expander("🔬 Research Trail", expanded=False):
                                
                                intermediate_steps = value.get('research_steps', [])
                                
                                st.session_state['company_info'] = next((obs for act, obs in intermediate_steps if act.tool == "get_company_info"), None)
                                st.session_state['price_summary']  = next((obs for act, obs in intermediate_steps if act.tool == "get_price_summary"), None)
                                st.session_state['news_summary_output'] = next((obs for act, obs in intermediate_steps if act.tool == "Financial_News_Analyst"), None)
                                st.session_state['filings_data'] = next((obs for act, obs in intermediate_steps if act.tool == "get_latest_filings"), [])
                                
                                if "initial_analysis" in value:
                                    st.session_state['initial_analysis'] = value["initial_analysis"]

                                display_research_details(
                                    st.session_state['company_info'],
                                    st.session_state['price_summary'],
                                    st.session_state['news_summary_output'],
                                    st.session_state['filings_data'],
                                    st.session_state['initial_analysis']
                                    )

                        elif key == "critic":
                            research_status_placeholder.success("🕵️‍♂️ **Researcher Agent:** Research complete.")
                            critic_status_placeholder.warning("🧐 **Critic Agent:** Evaluating the initial analysis...")
                            if "critique" in value:
                                st.session_state['critique'] = value["critique"]
                                with critic_placeholder.expander('🧐 critique'):
                                    st.markdown(st.session_state['critique'])

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

                progress_bar.progress(100)
                status_placeholder.success("Workflow Complete!")

            with final_flex_placeholder.container(horizontal=True, height="content", horizontal_alignment="right"):
                with st.popover('🔬 Research Trail'):
                    display_research_details(
                                    st.session_state['company_info'],
                                    st.session_state['price_summary'],
                                    st.session_state['news_summary_output'],
                                    st.session_state['filings_data'],
                                    st.session_state['initial_analysis']
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