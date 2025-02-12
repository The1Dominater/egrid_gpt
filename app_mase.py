### Multi-Agent Solver Expanded ###

# Default package imports
import re, json, os, operator, asyncio, datetime, time
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

#########################################################################################################
#################################### Helper Functions ###################################################
#########################################################################################################
def load_prompts():
    # Description: Read prompt texts from external files and load them as strings
    # Input: none
    # Output: json file a python list with entries as dicts{"name": "", "text" : "" }
    with open('./prompts/prompts.json') as f:
        prompts = json.load(f)
    for prompt in prompts:
        with open(prompt["text"], 'r') as prompt_file:
            prompt["text"] = prompt_file.read()
    
    return prompts

def display_msgs(sender: str, receiver: str, query: str, response: str):
    # Description: Simple function to display a QnA interation in color
    # Input: sender - actor(human or agent) who sent the query
    #        receiver - actor(human or agent) who received the query
    #        query - input from sender displayed in green
    #        response - output from receiver displated in purple
    # Output: python list with each entry as a dict following the format {"Description" : "", "Tasks" : ""}
    print(f"\033[1m\033[32m{sender}\n\033[0m\033[32m{query}\033[0m")
    print(f"\033[1m\033[35m{receiver}\n\033[0m\033[35m{response}\033[0m")

    return

def extract_python_dict(response):
    # Description: Takes in the response from the primary agent and extracts a python dict from response
    # Input: response - chunck of text containing the python dict
    # Output: agent_profiles - the python dict as a string
    
    pattern = r'{\s*("[^"]*"\s*:\s*"[^"]*"\s*(?:,\s*"[^"]*"\s*:\s*"[^"]*"\s*)*)}'
    
    # Use re.search to find the pattern in the text
    match = re.search(pattern, response, re.DOTALL)
    
    # Cycle through each expert agent to make their profile
    if match:
        dict_str = match.group(0)
        return dict_str
    else:
        print("Could not determine agent names")
        return None  # Return None if pattern is not found in the text

def create_agent_profiles(response):
    # Description: Takes in the response and generates a python list of agents from response
    # Input: response - chunck of text containing the agent names and descriptions
    # Output: agent_profiles - a python list of dicts containing the expert agent profiles
    
    try:
        # Parse the matched string as JSON
        agent_list = json.loads(response)

        agent_profiles = []
        for agent_name, agent_description in agent_list.items():
            agent_profiles.append({
                "Name": agent_name,
                "Description": agent_description,
                "Question": "",
                "Response" : "",
                "Feedback": "None"
            })
        return agent_profiles
    except json.JSONDecodeError:
        print("Json was improperly formatted")
        return None  # Return None if JSON parsing fails

def store_feedback(response, agent_profiles):
    # Description: Takes in the response and add feedback to the python list of agents based on the response
    # Input: response - chunck of text containing the agent names and feedback
    # Output: agent_profiles - a python list of dicts containing the expert agent profiles; adds feedback to the profiles
    try:
        # Parse the matched string as JSON
        feedback = json.loads(response)

        for agent_profile in agent_profiles:
            for agent_name, agent_feedback in feedback.items():
                if agent_name == agent_profile["Name"]:
                    agent_profile["Feedback"] = agent_feedback
                    break
            
    except json.JSONDecodeError:
        print("Json was improperly formatted! No feedback available!")

    return agent_profiles

#########################################################################################################
################################### Agent Definitions ###################################################
#########################################################################################################
def idea_agent(state):
    # Description: Queries an LLM, who carefully reviews the user's problem statement and provides
    #              a thorough and critical analysis of the provided problem statement
    # Input: state - current state object of the graph
    #               * prompts - load in engineered prompts from text files
    # Output: state - updated state object of graph including:
    #               * messages - append ideas on how to solve problem and ideas on the panel of experts to create
    #               * iterations - set initial iterations to 0

    global VERBOSE
    query = (state["messages"][0]).content # Load the initial user input as the query
    prompts = state["prompts"]
    # Dummy proof setting the max iteration value
    if state["max_iterations"] is None:
        max_i = 1
    else:
        #Don't let the maximum number of iterations exceed 5 for resource purposes
        max_i = min(5,int(state["max_iterations"]))

    # Allows for use of pretext prompts like "scratchpad tool" and "God mode"
    pretext = next(prompt["text"] for prompt in prompts if "pretext_prompt" in prompt["name"])
    
    # Give the Idea agent a prompt about its primary function of analyzing problems
    problem_analysis_prompt = next(prompt["text"] for prompt in prompts if "problem_analysis_prompt" in prompt["name"])
    prompt = problem_analysis_prompt.format(pretext=pretext, query=query)
    problem_analysis_idea = llm.invoke(prompt)
    if VERBOSE == True:
        display_msgs("Human", "Idea Agent", prompt, problem_analysis_idea)

    # Create a panel of experts based on its analysis of the problem statement
    create_panel_of_experts_prompt = next(prompt["text"] for prompt in prompts if "create_panel_of_experts_prompt" in prompt["name"])
    prompt = create_panel_of_experts_prompt.format(pretext=pretext, problem_analysis=problem_analysis_idea)
    create_panel_of_experts_idea = llm.invoke(prompt)
    if VERBOSE == True:
        display_msgs("Human", "Idea Agent", prompt, create_panel_of_experts_idea)

    # Extract panel of experts as a python list
    poe_no_fluff = extract_python_dict(create_panel_of_experts_idea)
    agent_profiles = create_agent_profiles(poe_no_fluff)
    
    return {"messages": [HumanMessage(content=f"Problem analysis:{problem_analysis_idea}"), HumanMessage(content=poe_no_fluff)],
            "iteration": 0,
            "poe_description": poe_no_fluff,
            "agent_profiles": agent_profiles,
            "max_iterations": max_i}

def delegation_agent(state):
    # Description: Updates the descriptions and tasks of each agent contained in the agents_list based on feedback from reflection agent
    # Input: state - current state object of the graph, especially:
    #              * prompts - list of prompts to pass to LLM
    #              * agent_list - list of expert agents to update
    #              * agent_profiles - list of agent profiles containing dict w/ agent description and task
    #              * problem_analysis - initial analysis conducted by the idea agent
    #              * poe_decription - initial description of panel of exeperts
    # Output: state - updated state object of graph including:
    #               * agent_profiles - generates/regenerates description and task for each expert agent
    #               * iteration - increments iteration value

    global VERBOSE
    # Grab state variables
    prompts = state["prompts"]
    agent_profiles = state["agent_profiles"]
    poe_description = state["poe_description"]
    i = state["iteration"]

    # Setup local variables
    query = (state["messages"][0]).content
    problem_analysis = (state["messages"][1]).content
    
    conversation = []
    pretext = next(prompt["text"] for prompt in prompts if "pretext_prompt" in prompt["name"])

    generate_question_prompt = next(prompt["text"] for prompt in prompts if "generate_question_prompt" in prompt["name"])

    # For each agent in the panel of experts
    for agent_profile in agent_profiles:
        # Create a description
        agent_name = agent_profile["Name"]
        agent_description = agent_profile["Description"]
        agent_feedback = agent_profile["Feedback"]
        # Create a list of tasks
        q_prompt = generate_question_prompt.format(pretext=pretext,
                                                query=query, 
                                                problem_analysis=problem_analysis,
                                                poe_description=poe_description,
                                                agent_name=agent_name, 
                                                agent_description=agent_description,
                                                agent_feedback=agent_feedback)
        agent_question = llm.invoke(q_prompt)
        agent_profile["Question"] = agent_question

        # Append delegation agents responses to the message chain
        conversation.append(HumanMessage(content=agent_question))
        
        if VERBOSE == True:
            display_msgs("Human", "Delegation Agent", q_prompt, agent_question)
    
    return {"messages": conversation, 
            "agent_profiles": agent_profiles,
            "iteration":(i + 1)}

def query_agent(state):
    # Description: Iterates over the list of agents, querying each one in order to collect its expert response
    # Input: state - current state object of the graph, especially:
    #              * agent_profiles - list of agent profiles containing dict w/ agent description and task
    # Output: state - updated state object of graph including:
    #               * agent_profiles - generates/regenerates description and task for each expert agent

    global VERBOSE
    agent_profiles=state["agent_profiles"]
    conversation = []
    rewoo = ReWOO() # Initiate a ReWOO solving agent via API
    
    for agent_profile in agent_profiles:
        # Ask ReWOO to handle agent's subproblem
        agent_question=agent_profile["Question"]
        agent_response = rewoo.invoke(agent_question)
        # Store response
        agent_profile["Response"] = agent_response

        # Output to converstation
        conversation.append(HumanMessage(content=agent_response))
        if VERBOSE == True:
            agent_name = agent_profile["Name"]
            display_msgs("Query Agent", agent_name, agent_question, agent_response)     

    return {"messages": conversation, 
            "agent_profiles": agent_profiles}

def continuation_agent(state)-> Literal["End", "Continue"]:
    i = state["iteration"]
    max_i = state["max_iterations"]
    if i == max_i:
        return "End"
    else:
        #return "Continue"
        query = (state["messages"][0]).content
        prompts = state["prompts"]
        agent_profiles = state["agent_profiles"]
        
        pretext = next(prompt["text"] for prompt in prompts if "pretext_prompt" in prompt["name"])
        continuation_prompt = next(prompt["text"] for prompt in prompts if "continuation_prompt" in prompt["name"])

        responses = ""
        for agent_profile in agent_profiles:
            agent_response = agent_profile["Response"]
            responses = responses + agent_response
        
        prompt = continuation_prompt.format(pretext=pretext,
                                            query=query,
                                            responses=responses)
        is_answered = llm.invoke(prompt)

        display_msgs("Human Agent", "Continuation Agent", prompt, is_answered)
        
        if "ANSWERED" in is_answered:
            return "End"
        else:
            return "Continue"

def reflection_agent(state):
    # Description: Creates a primary reflection agent to examine the responses of other agents and provide
    #              feedback to help them improve their responses
    # Input: state - current state object of the graph, especially:
    #              * prompts - list of prompts to pass to LLM
    #              * agent_profiles - list of agent profiles containing dict w/ agent description and task
    # Output: state - updated state object of graph including:
    #               * agent_profiles - generates/regenerates description and task for each expert agent

    global VERBOSE
    # Grab state variables
    prompts = state["prompts"]
    agent_profiles=state["agent_profiles"]
    # Setup local variables
    query = (state["messages"][0]).content
    poe_description = state["poe_description"]
    pretext = next(prompt["text"] for prompt in prompts if "pretext_prompt" in prompt["name"])
    feedback_prompt = next(prompt["text"] for prompt in prompts if "feedback_prompt" in prompt["name"])

    q_and_r = ""
    for agent_profile in agent_profiles:
        agent_name = agent_profile["Name"]
        agent_question = agent_profile["Question"]
        agent_response = agent_profile["Response"]

        # Compile all tasks and responses in order to allow the feedback agent to analyze all at the same time
        q_and_r = q_and_r + "Expert:" + agent_name + "\nQuestion:" + agent_question + "\nResponse:" + agent_response + "\n"

    # Query the feedback LLM with the tasks and responses
    prompt = feedback_prompt.format(pretext=pretext,
                                    query=query,
                                    poe_description=poe_description,
                                    q_and_r=q_and_r)
    all_feedback = llm.invoke(prompt)

    # Extract the feedback and assign it to each agent
    feedback_dict = extract_python_dict(all_feedback)
    agent_profiles = store_feedback(feedback_dict, agent_profiles)

    if VERBOSE == True:
        display_msgs("Human", "Feedback Agent", prompt, all_feedback)
        
    return {"messages": [HumanMessage(content=all_feedback)], 
            "agent_profiles": agent_profiles}

def summarization_agent(state):
    # Description: Creates a summary of the responses from the expert agents
    # Input: state - current state object of the graph, especially:
    #              * prompts - list of prompts to pass to LLM
    #              * agent_profiles - list of agent profiles containing dict w/ agent description and task
    # Output: state - updated state object of graph including:
    #               * agent_profiles - generates/regenerates description and task for each expert agent

    global VERBOSE
    prompts = state["prompts"]
    query = (state["messages"][0]).content
    agent_profiles = state["agent_profiles"]
    poe_description = state["poe_description"]
    pretext = next(prompt["text"] for prompt in prompts if "pretext_prompt" in prompt["name"]) 
    summarize_prompt = next(prompt["text"] for prompt in prompts if "summarize_prompt" in prompt["name"])
    
    compiled_responses  = ""
    for agent_profile in agent_profiles:
        compiled_responses = compiled_responses + agent_profile["Name"] + ":" + agent_profile["Response"] + "\n"

    prompt = summarize_prompt.format(pretext=pretext,
                                     query=query,
                                     poe_description=poe_description,
                                     compiled_responses=compiled_responses)
    
    summary = llm.invoke(prompt)

    if VERBOSE == True:
        display_msgs("Human", "Summarization Agent", prompt, summary)  

    return {"messages": [HumanMessage(content=summary)]}

#########################################################################################################
####################################### State Graph #####################################################
#########################################################################################################
# This is essentially a blackboard object for passing info between agents
class AgentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage], operator.add]
    prompts : List
    agent_profiles : Optional[List] = None
    poe_description : Optional[str] = None
    iteration : Optional[int] = None
    max_iterations : int

def build_graph():
    # Description: Builds a graph using the established agent functions
    # Input: none
    # Output: CompiledGraph object

   # Initialize graph
    graph_builder = StateGraph(AgentState)
    # Add agents as nodes
    graph_builder.add_node("Idea Agent", idea_agent)
    graph_builder.add_node("Delegation Agent", delegation_agent)
    graph_builder.add_node("Query Agent", query_agent)
    graph_builder.add_node("Reflection Agent", reflection_agent)
    graph_builder.add_node("Summarization Agent", summarization_agent)

    # # Add edges to establish flow
    graph_builder.add_edge(START, "Idea Agent")
    graph_builder.add_edge("Idea Agent", "Delegation Agent")
    graph_builder.add_edge("Delegation Agent", "Query Agent")
    graph_builder.add_conditional_edges("Query Agent", continuation_agent, {
        "Continue": "Reflection Agent",
        "End": "Summarization Agent"})
    graph_builder.add_edge("Reflection Agent", "Delegation Agent")
    graph_builder.add_edge("Summarization Agent", END)

    graph = graph_builder.compile()
    # Display graph
    #graph.get_graph().print_ascii()

    return graph

def main(user_input:str = ""):
    # Description: Main function which executes all necessary functions
    # Input: none
    # Output: none

    global VERBOSE # If true, prints all queries to LLM(s) and responses
    VERBOSE = False
    MAX_ITERATIONS = 3 # Set this to the number of times you want the agents to loop throught the cycle

    # Build graph
    graph = build_graph()
    # Load agent prompts
    prompts = load_prompts()
    # Load in initial message(Can pass manually if desired)
    if user_input == "":
        user_input = next(prompt["text"] for prompt in prompts if "user_input_prompt" in prompt["name"])
    # Invoke with CompliledGraph with default variables
    conversation = graph.invoke({"messages" : [HumanMessage(content=user_input)],
                                "prompts" : prompts,
                                "max_iterations" : MAX_ITERATIONS,
                                "sender" : "User" })
    
    # Print out info about solution
    print("Framework: MASE")
    summary = (conversation["messages"][-1]).content
    # Print out summary
    #print(f"Final Response: {summary}")

    return summary

# Run main function
if __name__ == "__main__":
    main()