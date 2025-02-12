### Mixture of Experts(Advanced, follows Nova prompt structure) ###

# Default imports
import re, json, os, operator
from langgraph.graph import StateGraph, START, END, add_messages
from typing import Annotated, List, Sequence, TypedDict, Optional, Literal
from langchain_core.messages import BaseMessage, HumanMessage
from api_agent_builder import ExpertAgent
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

#main_llm = fake_LLM(model="mixtral-8x7b-instruct", sleep_time=60)
main_llm = fake_LLM(model="llama-3-sonar-large-32k-online", sleep_time=60)
#main_llm = fake_LLM(model="llama-3-8b-instruct", sleep_time=60)

#########################################################################################################
#################################### Helper Functions ###################################################
#########################################################################################################
def grab_prompt(prompt_name: str):
    # Description: Reads prompt text from external files and load it as a string
    # Input: name of prompt
    # Output: text in prompt file as a string

    with open('./moe_prompts/prompts.json') as f:
        prompts = json.load(f)
    
    txt_prompt = f"Could not find prompt:{prompt_name}\n" # Default prompt to return if prompt is not found
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

def create_experts(response):
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
    except:
        print("Json was improperly formatted")
        return "ERROR" # Return None if JSON parsing fails

def update_experts(response, moe):
    # Description: Takes in the response and generates a python list of agents from response
    # Input: response - chunck of text containing the agent names and descriptions
    # Output: agent_profiles - a python list of dicts containing the expert agent profiles
    
    updated_moe = moe

    try:
        # Parse the matched string as JSON
        updated_experts = json.loads(response)

        if len(update_experts) == len(moe):
            for expert in updated_moe:
                expert_name = expert.get_name()
                for updated_name, updated_description in updated_experts:
                    if expert_name == updated_name:
                        expert.set_role(updated_description)
                        del update_experts[updated_name]
                        break
    except:
        print("Json was improperly formatted")
    
    return updated_moe # Return None if JSON parsing fails

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

#########################################################################################################
################################### Graph Functions #####################################################
#########################################################################################################
def classify_problem(state):
    global VERBOSE
    user_input = (state["messages"][0]).content
    
    # Given a user prompt pull out the query
    classify_prompt=grab_prompt("classify_problem")
    classify_prompt=classify_prompt.format(pretext="",user_input=user_input)
    response = main_llm.invoke(classify_prompt)
    if VERBOSE == True:
        display_msgs("Human", "Classification Expert", classify_prompt, response)

    # Extract complexity rating from response
    complexity = extract_python_dict(response)

    return {"messages": [HumanMessage(content="Rated problem complexity...")],
            "complexity": complexity}

def complexity_router(state) -> Literal["EASY","DETERMINISTIC","OPEN"]:
    complexity = state["complexity"]

    if complexity is not None:
        if "EASY" in complexity:
            return "EASY"
        elif "DETERMINISTIC" in complexity:
            return "DETERMINISTIC"
        else:
            return "OPEN"
    else:
        print("Failed to determine question type")
        return "EASY"

def respond_immediately(state):
    global VERBOSE
    user_input = (state["messages"][0]).content
    
    # Given a user prompt pull out the query
    pretext=grab_prompt("pretext")
    #prompt = f"{pretext}\n\nUser Input:\n{user_input}"
    prompt = f"{user_input}"
    response = main_llm.invoke(prompt)
    if VERBOSE == True:
        display_msgs("Human", "Base Model", prompt, response)
    return {"messages": [HumanMessage(content="Basic model responded")],
            "summary": response}

def reason_deterministically(state):
    global VERBOSE
    user_input = (state["messages"][0]).content
    
    # Given a user prompt pull out the query
    rewoo = ReWOO()
    response = rewoo.invoke(user_input)
    if VERBOSE == True:
        display_msgs("Human", "ReWOO Agent", prompt, response)
    return {"messages": [HumanMessage(content="ReWOO agent responded")],
            "summary": response}

def improve_prompt(state):
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

def analyze_problem(state):
    # Initialize necesary variables
    global VERBOSE
    improved_prompt = state["question"]
    
    # Given a user prompt pull out the query
    pretext=grab_prompt("pretext")
    problem_analysis_prompt = grab_prompt("problem_analysis")
    problem_analysis_prompt = problem_analysis_prompt.format(pretext="",improved_prompt=improved_prompt)
    problem_analysis = main_llm.invoke(problem_analysis_prompt)
    if VERBOSE == True:
        display_msgs("Human", "Problem Analysis Expert", problem_analysis_prompt, problem_analysis)
    return {"messages": [HumanMessage(content=problem_analysis)],
            "problem_analysis": problem_analysis}

def update_moe(state):
    # Initialize necesary variables
    global VERBOSE
    improved_prompt = state["question"]
    i = state["iteration"]
    # Create descriptions for a mixture of experts based on the new user prompt
    pretext=grab_prompt("pretext")
    
    if i == 0:
        moe_prompt = grab_prompt("moe")
        moe_prompt = moe_prompt.format(pretext="",improved_prompt=improved_prompt)
    else:
        evaluation = state["evaluation"]
        previous_solution = state["solution"]
        moe = state["moe"]
        moe_prompt = grab_prompt("moe_upgrade")
        all_experts = ""
        for current_expert in moe:
            all_experts = all_experts + "Expert Name:" + current_expert.get_name() + "\nRole:" + current_expert.get_role() + "\n"
        moe_prompt = moe_prompt.format(pretext="",improved_prompt=improved_prompt,experts=all_experts,previous_solution=previous_solution,evaluation=evaluation)

    for attempt in range(3):
        moe_idea = main_llm.invoke(moe_prompt)
        if VERBOSE == True:
            display_msgs("Human", "Idea Expert", moe_prompt, moe_idea)

        # Extract panel of experts as a python list
        moe_dict = extract_python_dict(moe_idea)
        if moe_dict is not None:
            break
    
    if i == 0:
        moe = create_experts(moe_dict)
    else:
        previous_moe = state["moe"]
        moe = update_experts(moe_dict,previous_moe)

    for expert in moe:
        other_experts = ""
        for other_expert in moe:
            if other_expert != expert:
                other_experts = "Peer Name:" + other_expert.get_name() + "\nPeer Role:" + other_expert.get_role() + "\n"
        expert.set_context(f"All the experts working on the problem - {other_experts}\nFocus on your specific role and avoid carrying out analysis other experts would be better suited for.")

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
    problem_analysis = state["problem_analysis"]
    moe = state["moe"]

    all_responses = ""
    for expert in moe:
        # Query Expert
        response = expert.query(f"The user asked: {improved_prompt}\nAn AI analyst responded:{problem_analysis}\n\nBased on your unique role, please explain which part of the response is most important and why. Please use your unique insight to recommend how the user can best should address this part of the response.")
        # Append response
        expert_name = expert.get_name()
        all_responses = all_responses + expert_name + " said:" + response + "\n"
        if VERBOSE == True:
            display_msgs("Human", expert_name, improved_prompt, response)

    return {"messages": [HumanMessage(content="Panel interviewed!")],
            "solution": all_responses}

def summarize_responses(state):
    # Description: Reviews the evaluation of the solution input and rewrites the agent descriptions
    # Input: state - current state object of the graph
    #               * messages - to grab original user input
    # Output: state - updated state object of graph including:
    #               * messages - append ideas on how to solve problem and ideas on the panel of experts to create
    #               * prompts - load in engineered prompts from text files
    #               * iterations - set initial iterations to 0

    # Initialize necesary variables
    global VERBOSE
    improved_prompt = state["question"]
    solution = state["solution"]
    problem_analysis = state["problem_analysis"]
  
    # Create descriptions for a mixture of experts based on the new user prompt
    summary_prompt = grab_prompt("summary")
    summary_prompt = summary_prompt.format(pretext="",improved_prompt=improved_prompt,problem_analysis=problem_analysis,solution=solution)
    summary = main_llm.invoke(summary_prompt)
    if VERBOSE == True:
        display_msgs("Human", "Solution Expert", summary_prompt, summary)

    return {"messages": [HumanMessage(content="Summary completed!")],
            "summary": summary}

def evaluate_solution(state):
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

def evaluation_router(state)-> Literal["End", "Continue"]:
    # Decription: If the max number of cycles has been reached force experts to respond, else continue if the question has not been answered
    # Input: state - current state object of the graph, especially:
    #              * iteration - current iteration of the graph
    # Output: string Literal - whether or not to end or continue
    i = state["iteration"]
    max_i = state["max_iterations"]
    if i == max_i:
        return "End"
    elif i == 1: # Always loop back after the first iteration to improve solution
        return "Continue"
    else: # It the evaluator feels the solution is adequate stop, else continue
        evaluation = state["evaluation"]
        if "ANSWERED" in evaluation:
            return "End"
        else:
            return "Continue"

#########################################################################################################
####################################### State Graph #####################################################
#########################################################################################################
# This is essentially a blackboard object for passing info between agents
class AgentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage], operator.add]
    moe : Optional[List] = None
    question : str
    problem_analysis : str
    solution : str
    summary : str
    summary : str
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
    graph_builder.add_node("Classify Problem", classify_problem)
    graph_builder.add_node("Respond Immediately", respond_immediately)
    graph_builder.add_node("Reason with CoT", reason_deterministically)
    graph_builder.add_node("Improve Prompt", improve_prompt)
    graph_builder.add_node("Update MoE", update_moe)
    graph_builder.add_node("Analyze Problem", analyze_problem)
    graph_builder.add_node("Interview Experts", interview_experts)
    graph_builder.add_node("Summarize Responses", summarize_responses)
    graph_builder.add_node("Evaluate Solution", evaluate_solution)

    # Add edges to establish flow
    graph_builder.add_edge(START, "Classify Problem")
    graph_builder.add_conditional_edges("Classify Problem", complexity_router, {
        "EASY": "Respond Immediately",
        "DETERMINISTIC": "Reason with CoT",
        "OPEN": "Improve Prompt"
    })
    graph_builder.add_edge("Respond Immediately", END)
    graph_builder.add_edge("Reason with CoT", END)
    graph_builder.add_edge("Improve Prompt", "Analyze Problem")
    graph_builder.add_edge("Analyze Problem", "Update MoE")
    graph_builder.add_edge("Update MoE", "Interview Experts")
    graph_builder.add_edge("Interview Experts", "Summarize Responses")
    graph_builder.add_edge("Summarize Responses", "Evaluate Solution")
    graph_builder.add_conditional_edges("Evaluate Solution", evaluation_router, {
        "Continue": "Analyze Problem",
        "End": END
    })

    # Compile the graph
    graph = graph_builder.compile()
    # Display graph
    #graph.get_graph().print_ascii()

    return graph

def main(user_input: str = None):
    # Description: Main function which executes all necessary functions
    # Input: none
    # Output: none

    global VERBOSE # If true, prints all queries to LLM(s) and responses
    VERBOSE = False
    MAX_ITERATIONS = 3 # Set this to the number of times you want the agents to loop throught the cycle

    # Build graph
    graph = build_graph()
    # Load in initial message(Can pass manually if desired)
    if user_input is None:
        user_input = grab_prompt("user_input")
    # Invoke with CompliledGraph obj with default variables
    conversation = graph.invoke({"messages" : [HumanMessage(content=user_input)],
                                 "max_iterations" : MAX_ITERATIONS})

    # Print out info about solution
    #print("Framework: Nova 2")
    #all_responses = conversation["solution"]
    #print(f"#################################################################\nRaw Expert Solutions:\n{all_responses}")
    summary = conversation["summary"]
    #print(f"#################################################################\nFinalized Solution Summary:\n{summary}")

    return summary

# Run main function
if __name__ == "__main__":
    main()