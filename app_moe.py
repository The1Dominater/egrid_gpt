### Mixture of Experts(Simple) ###

# Default imports
import re, json, os, operator
from langgraph.graph import StateGraph, START, END, add_messages
from typing import Annotated, List, Sequence, TypedDict, Optional, Literal
from langchain_core.messages import BaseMessage, HumanMessage
from api_agent_builder import ExpertAgent

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

main_llm = fake_LLM(model="mixtral-8x7b-instruct", sleep_time=60)
#main_llm = fake_LLM(model="llama-3-sonar-large-32k-online", sleep_time=60)

#########################################################################################################
#################################### Helper Functions ###################################################
#########################################################################################################
def grab_prompt(prompt_name: str):
    # Description: Reads prompt text from external files and load it as a string
    # Input: name of prompt
    # Output: text in prompt file as a string

    with open('./moe_prompts/prompts.json') as f:
        prompts = json.load(f)
    
    txt_prompt = "\nNo prompt found...\n" # Default prompt to return if prompt is not found
    for prompt in prompts:
        name = prompt["name"]
        if name == prompt_name:
            with open(prompt["text"], 'r') as prompt_file:
                txt_prompt = prompt_file.read()
    
    #print(f"Prompt Name:{prompt_name} returned prompt:{txt_prompt}")
    return txt_prompt

def extract_python_dict(text):
    # Description: Takes in the response from the primary agent and extracts a python dict from response
    # Input: response - chunck of text containing the python dict
    # Output: agent_profiles - the python dict as a string

    # Use re.search to find the pattern in the text
    pattern = r'{\s*("[^"]*"\s*:\s*"[^"]*"\s*(?:,\s*"[^"]*"\s*:\s*"[^"]*"\s*)*)}'
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        dict_str = match.group(0)
        return dict_str
    else:
        print("Could not extract dict from text!")
        return None  # Return None if pattern is not found in the text

def create_expert_agents(response):
    # Description: Takes in the response and generates a python list of agents from response
    # Input: response - chunck of text containing the agent names and descriptions
    # Output: agent_profiles - a python list of dicts containing the expert agent profiles
    
    try:
        # Parse the matched string as JSON
        agents = json.loads(response)

        agent_profiles = []
        for agent_name, agent_description in agents.items():
            new_agent = ExpertAgent(name=agent_name,role=agent_description)
            agent_profiles.append(new_agent)
        return agent_profiles
    except json.JSONDecodeError:
        print("Json was improperly formatted")
        return None  # Return None if JSON parsing fails

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
################################### Graph Functions #####################################################
#########################################################################################################
def brainstorm(state):
    # Description: Review the user's input and rewrite it as an imporved prompt, then creata a mixture of experts who can provide
    #              a thorough, critical analysis of the problem the problem statement
    # Input: state - current state object of the graph
    #               * messages - to grab original user input
    # Output: state - updated state object of graph including:
    #               * messages - append ideas on how to solve problem and ideas on the panel of experts to create
    #               * prompts - load in engineered prompts from text files
    #               * iterations - set initial iterations to 0

    # Initialize necesary variables
    global VERBOSE
    user_input = (state["messages"][0]).content
    
    # Given a user prompt pull out the query
    pretext=grab_prompt("pretext")
    brainstorm_prompt = grab_prompt("brainstorm")
    brainstorm_prompt = brainstorm_prompt.format(pretext="",user_input=user_input)
    improved_prompt = main_llm.invoke(brainstorm_prompt)
    if VERBOSE == True:
        display_msgs("Human", "Prompt Expert", brainstorm_prompt, improved_prompt)
    return {"messages": [HumanMessage(content=improved_prompt)],
            "question": improved_prompt,
            "iteration": 0}


def update_moe(state):
    # Initialize necesary variables
    global VERBOSE
    improved_prompt = state["question"]
    i = state["iteration"]
    # Create descriptions for a mixture of experts based on the new user prompt
    pretext=grab_prompt("pretext")
    moe_prompt = grab_prompt("moe")
    if i == 0:
        moe_prompt = moe_prompt.format(pretext="",improved_prompt=improved_prompt)
    else:
        evaluation = state["evaluation"]
        previous_solution = state["solution"]
        moe_prompt = moe_prompt.format(pretext="",improved_prompt=improved_prompt,previous_solution=previous_solution,evaluation=evaluation)

    moe_idea = main_llm.invoke(moe_prompt)
    if VERBOSE == True:
        display_msgs("Human", "Idea Expert", moe_prompt, moe_idea)

    # Extract panel of experts as a python list
    moe_dict = extract_python_dict(moe_idea)
    moe = create_expert_agents(moe_dict)

    for expert in moe:
        other_experts = ""
        for other_expert in moe:
            if other_expert != expert:
                other_experts = "Other Expert Name:" + other_expert.get_name() + "\nOther Expert Role:" + other_expert.get_role() + "\n"
        expert.set_context(f"All the experts working on the problem:{other_experts}\nFocus on your specific role and avoid carrying out analysis other experts would be better suited for.")

    return {"messages": [HumanMessage(content=moe_dict)],
            "moe": moe}

def interview_experts(state):
    # Description: "Interviews" each domain expert by quering them with the improved prompt; gathers unqiue perspective on problem from each expert
    # Input: state - current state object of the graph, especially:
    #              * question - improved prompt based on original user query
    #              * agent_profiles - list of agent profiles containing dict w/ agent description and task
    # Output: state - updated state object of graph including:
    #               * feedback - generates feedback and concerns about conversation
    global VERBOSE
    improved_prompt = state["question"]
    moe = state["moe"]

    all_responses = ""
    for expert in moe:
        # Query Expert
        response = expert.query(improved_prompt)
        # Append response
        expert_name = expert.get_name()
        all_responses = all_responses + expert_name + " said:" + response + "\n"
        if VERBOSE == True:
            display_msgs("Human", expert_name, improved_prompt, response)

    return {"messages": [HumanMessage(content="Panel interviewed!")],
            "solution": all_responses}

def evaluate_responses(state):
    # Description: Evaluate the responses of other experts and provide
    #              feedback to help them improve their responses
    # Input: state - current state object of the graph, especially:
    #              * messages - allows for grabing the initial user input to ensure the conversation still aligns with it
    #              * question - improved prompt based on original user input
    #              * agent_profiles - list of agent profiles containing dict w/ agent description and task
    # Output: state - updated state object of graph including:
    #               * feedback - generates feedback and concerns about conversation

    global VERBOSE
    user_input = (state["messages"][0]).content
    moe=state["moe"]
    solution = state["solution"]
    i = state["iteration"]

    # Ask LLM to evaluate solution so far
    pretext = grab_prompt("pretext")
    evaluate_solution_prompt = grab_prompt("evaluate_solution")
    evaluate_solution_prompt = evaluate_solution_prompt.format(pretext="",user_input=user_input,solution=solution)
    evaluation = main_llm.invoke(evaluate_solution_prompt)

    if VERBOSE == True:
        display_msgs("Human", "Critical Analysis Expert" , evaluate_solution_prompt, evaluation)   
        
    return {"messages": [HumanMessage(content=evaluation)],
            "iteration":(i + 1), 
            "evaluation": evaluation}

def router(state)-> Literal["End", "Continue"]:
    # Decription: If the max number of cycles has been reached force experts to respond, else continue if the question has not been answered
    # Input: state - current state object of the graph, especially:
    #              * iteration - current iteration of the graph
    # Output: string Literal - whether or not to end or continue
    i = state["iteration"]
    max_i = state["max_iterations"]
    if i == max_i:
        return "End"
    else:
        evaluation = state["evaluation"]
        if "ANSWERED" in evaluation[-10:-1]:
            return "End"
        else:
            return "Continue"

def revise_moe(state):
    # Description: Reviews the evaluation of the solution input and rewrites the agent descriptions
    # Input: state - current state object of the graph
    #               * messages - to grab original user input
    # Output: state - updated state object of graph including:
    #               * messages - append ideas on how to solve problem and ideas on the panel of experts to create
    #               * prompts - load in engineered prompts from text files
    #               * iterations - set initial iterations to 0

    # Initialize necesary variables
    global VERBOSE
  
    # Create descriptions for a mixture of experts based on the new user prompt
    moe_prompt = grab_prompt("revise_moe")
    moe_prompt = moe_prompt.format(pretext="",previous_solution=previous_solution,evaluation=evaluation)
    moe_idea = main_llm.invoke(moe_prompt)
    if VERBOSE == True:
        display_msgs("Human", "Idea Expert", moe_prompt, moe_idea)

    # Extract panel of experts as a python list
    moe_dict = extract_python_dict(moe_idea)
    moe = create_expert_agents(moe_dict)

    return {"messages": [HumanMessage(content=improved_prompt), HumanMessage(content=moe_dict)],
            "question": improved_prompt,
            "moe": moe,
            "iteration": 0}

#########################################################################################################
####################################### State Graph #####################################################
#########################################################################################################
# This is essentially a blackboard object for passing info between agents
class AgentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage], operator.add]
    moe : Optional[List] = None
    question : str
    solution : str
    evaluation: str
    iteration : Optional[int] = None
    max_iterations : int

def build_graph():
    # Description: Builds a graph using the established agent functions
    # Input: none
    # Output: CompiledGraph object

    # Initialize graph
    graph_builder = StateGraph(AgentState)

    # Add functions as nodes
    graph_builder.add_node("Brainstorm", brainstorm)
    graph_builder.add_node("Update MoE", update_moe)    
    graph_builder.add_node("Interview Experts", interview_experts)
    graph_builder.add_node("Evaluate Responses", evaluate_responses)

    # Add edges to establish flow
    graph_builder.add_edge(START, "Brainstorm")
    graph_builder.add_edge("Brainstorm", "Update MoE")
    graph_builder.add_edge("Update MoE", "Interview Experts")
    graph_builder.add_edge("Interview Experts", "Evaluate Responses")
    graph_builder.add_conditional_edges("Evaluate Responses", router, {
        "Continue": "Update MoE",
        "End": END
    })

    # Compile the graph
    graph = graph_builder.compile()
    # Display graph
    #graph.get_graph().print_ascii()

    return graph

def main(user_input: str = ""):
    # Description: Main function which executes all necessary functions
    # Input: none
    # Output: summary as a str

    global VERBOSE # If true, prints all queries to LLM(s) and responses
    VERBOSE = False
    MAX_ITERATIONS = 3 # Set this to the number of times you want the agents to loop throught the cycle

    # Build graph
    graph = build_graph()
    # Load in initial message(Can pass manually if desired)
    if user_input == "":
        user_input = grab_prompt("user_input")
    # Invoke with CompliledGraph obj with default variables
    conversation = graph.invoke({"messages" : [HumanMessage(content=user_input)],
                                 "max_iterations" : MAX_ITERATIONS})

    # Print out info about solution
    print("Framework: MoE")
    # Print out summary
    summary = (conversation["messages"][-1]).content
    #print(f"Final Response: {summary}")

    return summary

# Run main function
if __name__ == "__main__":
    main()