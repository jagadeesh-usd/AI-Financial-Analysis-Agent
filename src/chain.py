from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator
from src.agents import get_planner_agent, get_executor_agent, get_critic_agent, get_refiner_agent, save_note_to_memory
import streamlit as st
import json

# --- Agent State ---
class AgentState(TypedDict):
    ticker: str
    plan: List[str] 
    reasoning: str
    company_info: dict
    research_steps: Annotated[List[dict], operator.add]
    initial_analysis: str
    critique: str
    refined_analysis: str
    memory_confirmation: str

# --- Agent Nodes ---
def planner_node(state):
    """Generates the research plan and reasoning."""
    planner_agent = get_planner_agent(ChatOpenAI(model="gpt-4.1-mini", temperature=0))
    # The agent's invocation now returns intermediate steps
    result = planner_agent.invoke({"input": f"Create a plan for {state['ticker']}"})
    
    # Parse the JSON string from the main output
    result_json_string = result['output']
    plan_data = json.loads(result_json_string)
    
    # Extract company info from the planner's intermediate steps
    company_info_data = {}
    if 'intermediate_steps' in result and result['intermediate_steps']:
        for action, observation in result['intermediate_steps']:
            if action.tool == 'get_company_info':
                company_info_data = observation
                break
    
    return {
        "reasoning": plan_data["reasoning"],
        "plan": plan_data["plan"],
        "company_info": company_info_data   
    }

def executor_node(state):
    """Executes the research plan."""
    executor_agent = get_executor_agent(ChatOpenAI(model="gpt-4.1-mini", temperature=0))
    # Pass both the plan and the ticker to the executor
    result = executor_agent.invoke({
        "input": f"Execute the following plan for the ticker {state['ticker']}:\n\nPLAN: {state['plan']}"
    })
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

    # Add the new nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("refiner", refiner_node)
    workflow.add_node("save_memory", save_memory_node)

    # Define the new sequence of edges
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "critic")
    workflow.add_edge("critic", "refiner")
    workflow.add_edge("refiner", "save_memory")
    workflow.add_edge("save_memory", END)

    return workflow.compile()

