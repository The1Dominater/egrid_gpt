### ReWOO Upgraded ###

# Default package imports
import re, json, os, operator
from langgraph.graph import  StateGraph, MessageGraph, START, END
from typing import Annotated, List, Sequence, TypedDict, Optional, Literal
from langchain_core.messages import BaseMessage, HumanMessage
from api_rewoo import ReWOO

#########################################################################################################
########################################### LLM Setup ###################################################
#########################################################################################################
# --- OpenAI Model Hosted by RJ --- #
# from langchain_openai.chat_models import ChatOpenAI

# llm = ChatOpenAI(temperature=0.0, base_url="http://rjain-40832s.nrel.gov:1234/v1", api_key="not-needed")

# --- Phi 3 Model pulled from HF --- #
#from local_phi_3_api import ChatMicroGridLLM

#llm = ChatMicroGridLLM()

# --- Perplexity Lab LLM from webscrapper --- #
from api_perplexity import fake_LLM

llm = fake_LLM(model="llama-3-sonar-large-32k-online", sleep_time=60)
#llm = fake_LLM(model="mixtral-8x7b-instruct", sleep_time=60)

#########################################################################################################
#################################### Helper Functions ###################################################
#########################################################################################################
def extract_python_list(text):
    # Regex pattern to find list inside brackets
    pattern = r'\[([^\[\]]*)\]'
    match = re.search(pattern, text)
    
    if match:
        list_str = match.group(1)
        # Split the string into list items
        items = [item.strip() for item in list_str.split(',')]
        return items
    else:
        return []

def display_msgs(sender: str, receiver: str, query: str, response: str):
    # Description: Simple function to display a QnA interation in color
    # Input: sender - actor(human or agent) who sent the query
    #        receiver - actor(human or agent) who received the query
    #        query - input from sender displayed in green
    #        response - output from receiver displated in purple
    # Output: python list with each entry as a dict following the format {"Description" : "", "Tasks" : ""}
    print(f"\033[1m\033[32m{sender}\n\033[0m \033[32m{query}\033[0m")
    print(f"\033[1m\033[35m{receiver}\n\033[0m \033[35m{response}\033[0m")

    return

#########################################################################################################
################################### Agent Definitions ###################################################
#########################################################################################################
def prompt_builder_agent(state):
    # Description: Prompt builder agent recontrusts query in a prompt or set of prompts to ask the ReWOO agent(s)
    # Input: state - current state object of the graph, especially:
    #              * messages - current message change
    # Output: state - updated state object of graph including:
    #               * messages - appends a zero-shot message response
    query = (state["messages"][0]).content
    i = state["iteration"]
    
    if i == 0:
        prompt = f"If this query can be broken into parts to make it easier to answer, please rewrite it into a series of sub-questions:{query}. Pretend as if you are asking each question to the same person sequentially to get the answer to the original question. Assume they know the context and you can be concise, using pronouns and references to previous questions. Do NOT attempt to answer the sub-questions. Please return the sub-questions as a python list, with a format following [\"Question 1?\", \"Question 2?\",...]"
        response = llm.invoke(prompt.format(query=query)).content
    else:
        conversation = state["conversation"]
        feedback = state["feedback"]
        prompt = f"Here is the user's original query:{query}\nHere is how an AI responded:{conversation}\nHere is feedback on the AI response:{feedback}\nUsing this context please rewrite the user query into an improved set of sub-questions to re-ask to the AI. Pretend as if you are asking each question to the same person sequentially to get the answer to the original question. Assume they know the context and you can be concise, using pronouns and references to previous questions. Do NOT attempt to answer the sub-questions. Please return the sub-questions as a python list, with a format following [\"Question 1?\", \"Question 2?\",...]"
        response = llm.invoke(prompt.format(query=query,conversation=conversation,feedback=feedback)).content
    
    all_questions = extract_python_list(response)
    if VERBOSE == True:
        display_msgs("Human", "Question Agent", prompt, all_questions)

    return {"messages": [HumanMessage(content="Question has been rewritten!")],
            "all_questions": all_questions}

def rewoo_agent(state):
    # Description: ReWOO agent which plans a series of tasks then executes each tasks step-by-step
    # Input: state - current state object of the graph, especially:
    #              * messages - current message change
    # Output: state - updated state object of graph including:
    #               * messages - appends a zero-shot message response

    global VERBOSE
    i = state["iteration"]
    query = (state["messages"][0]).content
    all_questions = state["all_questions"]
    rewoo = ReWOO()

    conversation = "Original Problem Statement:" + query + "\n"
    for question in all_questions:
        response = rewoo.invoke(f"Context:{conversation}\nUsing the context please respond to this question:{question}")
        # Update context
        conversation = conversation + "Question:" + question + "\nAnswer:" + response + "\n"
        # Print the converstation
        if VERBOSE == True:
            display_msgs("Human", "ReWOO Agent", question, response)
    
    return {"messages": [HumanMessage(content="ReWOO has solved each question!")],
            "conversation": conversation,
            "iteration": (i + 1)}

def reflection_agent(state):
    global VERBOSE
    conversation = state["conversation"]
    query = (state["messages"][0]).content
    prompt = f"The user originally asked:{query}\nAn AI solver broke the query down into a series of questions and responded to each one like so:{conversation}\nPlease determine if the AI solver has fully responded to the user's original query. Does the AI solver's response fully answer the user's original query? When responding if you feel the original query was answered please included ANSWERED at the very end of your response. If you feel the response did not answer the original question say IMPROVE."
    response = llm.invoke(prompt.format(query=query, conversation=conversation)).content

    if VERBOSE == True:
        display_msgs("Human", "Reflection Agent", prompt, response)

    return {"messages": [HumanMessage(content="Reflection agent provided feedback!")],
            "feedback":response}

def continuation_agent(state) -> Literal["End", "Continue"]:
    i = state["iteration"]
    max_i = state["max_iterations"]
    if i == max_i:
        return "End"
    else:
        feedback = state["feedback"]
        if "ANSWERED" in feedback:
            return "End"
        else:
            return "Continue"

#########################################################################################################
####################################### State Graph #####################################################
#########################################################################################################
# This is essentially a blackboard object for passing info between agents
class AgentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage], operator.add]
    conversation: Optional[str]
    feedback: Optional[str]
    all_questions : List
    iteration : Optional[int] = None

def build_graph():
    # Description: Builds a graph using the established agent functions
    # Input: none
    # Output: CompiledGraph object

    # Initialize graph
    graph_builder = StateGraph(AgentState)
    # Add agents as nodes
    graph_builder.add_node("Question Agent", prompt_builder_agent)
    graph_builder.add_node("ReWOO Agent", rewoo_agent)
    graph_builder.add_node("Reflection Agent", reflection_agent)

    # # Add edges to establish flow
    graph_builder.add_edge(START, "Question Agent")
    graph_builder.add_edge("Question Agent", "ReWOO Agent")
    graph_builder.add_edge("ReWOO Agent", "Reflection Agent")
    graph_builder.add_conditional_edges("Reflection Agent", continuation_agent, {
        "Continue": "Question Agent",
        "End": END})

    graph = graph_builder.compile()
    # Display graph
    graph.get_graph().print_ascii()

    return graph

def main(user_input: str = ""):
    # Description: Main function which executes all necessary functions
    # Input: none
    # Output: none

    global VERBOSE # If true, prints all queries to LLM(s) and responses
    VERBOSE = False
    MAX_ITERATIONS = 3 # Set this to the number of times you want the agents to loop throught the cycle

    # Build graph
    graph = build_graph()
    # Load in initial message(Can pass manually if desired)
    if user_input == "":
        user_input = "The US government is offering a $3,000 tax deductible for the first 10,000 solar PV installed this year, and your utility company is offering some amount discounts on home battery system.\nKnowing the average household uses 30kWh per day.\nIf you are the head of planning department at the utility, what is the percentage discount for battery system to increase the battery adoption so net load will not be fluctuate when cloud is covered.\nPlease give a plan with specific numbers for overall cost of implementation, which minimizes company expenses."
    # Invoke with CompliledGraph with default variables
    conversation = graph.invoke({"messages" : [HumanMessage(content=user_input)],
                                "max_iterations" : MAX_ITERATIONS})

    # Print out info about solution
    print("Framework: ReWOO")
    # Print out summary
    summary = conversation["conversation"]

    return summary

# Run main function
if __name__ == "__main__":
    main()