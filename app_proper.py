from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain.schema import BaseMessage
 
# Define the state
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]
    next_agent: str
 
# Initialize the LLM
from perplexity_api import fake_LLM
llm = ## Replace with your local LLM object (server or api, etc)
 
# Define the agents
 
def analyzer_agent(state):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert grid analyzer. Analyze the given data and identify potential risks to grid stability."),
        ("human", "Current Grid Status:\n"
                  "- Total load: 10,000 MW\n"
                  "- Conventional generation: 6,000 MW\n"
                  "- Wind generation: 3,000 MW\n"
                  "- Solar generation: 1,000 MW\n"
                  "- Grid frequency: 49.95 Hz (Normal range: 49.8 - 50.2 Hz)\n\n"
                  "Weather Forecast (next 6 hours):\n"
                  "- Wind speed decreasing from 15 m/s to 5 m/s\n"
                  "- Cloud cover increasing from 10% to 80%\n\n"
                  "Demand Forecast (next 6 hours):\n"
                  "- Peak demand expected to reach 11,000 MW in 4 hours\n\n"
                  "Energy Storage Status:\n"
                  "- Available capacity: 500 MWh\n"
                  "- Current charge level: 60%\n\n"
                  "Recent Incidents:\n"
                  "- Minor voltage fluctuations reported in the northern region\n"
                  "- One transmission line operating at 90% capacity\n\n"
                  "Analyze this data and identify potential risks to grid stability.")
    ])
    print(prompt.format_messages())
    response = llm.invoke(prompt.format_messages())
    state["messages"].append(response)
    state["next_agent"] = "strategy"
    return state
 
def strategy_agent(state):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert grid strategist. Propose a strategy to maintain grid stability based on the analysis."),
        ("human", "Based on the analysis, propose a strategy to maintain grid stability over the next 6 hours."),
        ("human", "{analysis}")
    ])
    analysis = state["messages"][-1].content
    response = llm.invoke(prompt.format_messages(analysis=analysis))
    state["messages"].append(response)
    state["next_agent"] = "action"
    return state
 
def action_agent(state):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert in grid operations. Suggest specific actions for managing the grid based on the proposed strategy."),
        ("human", "Based on the proposed strategy, suggest specific actions for:\n"
                  "a. Adjusting conventional generation\n"
                  "b. Managing energy storage\n"
                  "c. Implementing demand response measures"),
        ("human", "{strategy}")
    ])
    strategy = state["messages"][-1].content
    response = llm.invoke(prompt.format_messages(strategy=strategy))
    state["messages"].append(response)
    state["next_agent"] = "summary"
    return state
 
def summary_agent(state):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a communication expert. Provide a clear summary of the situation and recommendations for control room operators."),
        ("human", "Summarize the situation and recommendations based on the analysis, strategy, and actions proposed."),
        ("human", "{full_conversation}")
    ])
    full_conversation = "\n".join([msg.content for msg in state["messages"]])
    response = llm.invoke(prompt.format_messages(full_conversation=full_conversation))
    state["messages"].append(response)
    return state
 
# Define the workflow
workflow = StateGraph(State)
 
# Add nodes
workflow.add_node("analyzer", analyzer_agent)
workflow.add_node("strategy", strategy_agent)
workflow.add_node("action", action_agent)
workflow.add_node("summary", summary_agent)
 
# Add edges
workflow.add_edge("analyzer", "strategy")
workflow.add_edge("strategy", "action")
workflow.add_edge("action", "summary")
workflow.add_edge("summary", END)
 
# Set the entry point
workflow.set_entry_point("analyzer")
 
# Compile the workflow
app = workflow.compile()
 
# Run the workflow
result = app.invoke({
    "messages": [],
    "next_agent": "analyzer"
})
 
# Print the final summary
print(result["messages"][-1].content)