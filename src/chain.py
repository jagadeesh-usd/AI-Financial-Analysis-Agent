from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator
from src.agents import get_researcher_agent, get_critic_agent, get_refiner_agent, save_note_to_memory
import streamlit as st

# --- Agent State ---
class AgentState(TypedDict):
    ticker: str
    research_steps: Annotated[List[dict], operator.add]
    initial_analysis: str
    critique: str
    refined_analysis: str
    memory_confirmation: str

# --- Agent Nodes ---
def researcher_node(state):
    """Researches and provides the initial analysis."""
    researcher_agent = get_researcher_agent(ChatOpenAI(model="gpt-4.1-mini", temperature=0))
    result = researcher_agent.invoke({"input": f"Analyze the stock {state['ticker']}"})
    return {"research_steps": result['intermediate_steps'], "initial_analysis": result['output']}

def critic_node(state):
    """Critiques the initial analysis."""
    critic_agent = get_critic_agent(ChatOpenAI(model="gpt-4.1-mini", temperature=0))
    critique_text = critic_agent.invoke({"initial_analysis": state["initial_analysis"]}).content
    # st.write(critique_text)
    return {"critique": critique_text}

def refiner_node(state):
    """Refines the analysis based on the critique."""
    refiner_agent = get_refiner_agent(ChatOpenAI(model="gpt-4.1-mini", temperature=0))
    refined_text = refiner_agent.invoke({
        "initial_analysis": state["initial_analysis"],
        "critique": state["critique"]
    }).content
    # st.write(refined_text)
    return {"refined_analysis": refined_text}

def save_memory_node(state):
    """Generates a key insight and saves it to memory."""
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
    
    # Use an LLM to generate a concise note from the refined analysis
    note_generation_prompt = ChatPromptTemplate.from_template(
        "You are a note-taking assistant. Based on the following financial analysis, generate a single, concise sentence that captures the most important takeaway. This will be saved for future reference.\n\n"
        "Analysis:\n{analysis}\n\n"
        "Your single-sentence takeaway:"
    )
    
    note_chain = note_generation_prompt | llm
    key_insight = note_chain.invoke({"analysis": state["refined_analysis"]}).content
    
    # Save the generated note to memory
    confirmation = save_note_to_memory.invoke({
        "ticker": state["ticker"],
        "note": key_insight
    })
    
    return {"memory_confirmation": confirmation}

# --- Graph Definition ---
def build_agentic_workflow():
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("refiner", refiner_node)
    workflow.add_node("save_memory", save_memory_node)

    # Define edges
    workflow.set_entry_point("researcher")
    workflow.add_edge("researcher", "critic")
    workflow.add_edge("critic", "refiner")
    workflow.add_edge("refiner", "save_memory")
    workflow.add_edge("save_memory", END)

    return workflow.compile()

