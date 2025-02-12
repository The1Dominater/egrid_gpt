### Mixture of Experts(Advanced, follows Nova prompt structure) ###

# Default imports
import re, json, os, operator, asyncio
from langgraph.graph import StateGraph, START, END, add_messages
from typing import Annotated, List, Sequence, TypedDict, Optional, Literal
from langchain_core.messages import BaseMessage, HumanMessage
from api_agent_builder import ExpertAgent, async_ExpertAgent
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
from api_perplexity import async_fake_LLM, fake_LLM

#main_llm = fake_LLM(model="mixtral-8x7b-instruct", sleep_time=60)
main_llm = fake_LLM(model="llama-3-sonar-large-32k-online", sleep_time=60)
#main_llm = fake_LLM(model="llama-3-8b-instruct", sleep_time=60)
async_llm = async_fake_LLM(model="llama-3-sonar-large-32k-online", sleep_time=60)

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
    #pattern = r'{\s*("[^"]*"\s*:\s*"[^"]*"\s*(?:,\s*"[^"]*"\s*:\s*"[^"]*"\s*)*)}'
    text = text.replace("\n","")
    pattern = r"[{\[].*[}\]]"
    match = re.search(pattern, text)
    
    if match:
        dict_str = match.group(0)
        dict_str = dict_str.replace("[","{")
        dict_str = dict_str.replace("]","}")

        return dict_str
    else:
        print("Could not extract dict from text!")
        return None  # Return None if pattern is not found in the text

def make_checkpoint(state):
    print(state)
    return


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
    # Acts as a place holder int the graph for starting node
    global VERBOSE
    user_input = (state["messages"][0]).content
    
    # Given a user prompt pull out the query
    classify_prompt="You are a problem classifier. Please classify the type problem given, but do NOT respond to it. Your ONLY job is to determine its type.\nTo determine its type, remeber easy problems require simple fact retrevial, logic problems require step-by-step analysis, and planning problems are a combination of fact retrieval, logic problems, and analysis which potentially have multiple correct answers.\n\nProblem: " + user_input + "\n\nProblem type:\nEASY - simple problems which could be asked to the average American and they still could get it right without additional help; Examples: fact retrival questions like \"What is the capitol of China?\", basic math problems like \"Find x when 18+4=x\", greetings like \"How are you today?\"\nLOGIC - difficult problems which would require someone to slow down and go step-by-step to solve; Examples: task-based questions like \"First examine the patient. Next, analyze their symptoms. Next, determine treament options...\", complex math problems like \"What are the egien values of a q-bit <010>?\", questions which can be broken into sub-questions like \"What was the birthday of the wife of the 25th vice president?\"\nPLANNING - problems with potentially multiple correct answers which require a combination of fact retrival, step-by-step solving, and overall analysis; Examples: detailed planning like \"Develop a method which will assist policy makers in determining the best way to spend funds on public transportation. Use data ...\", advanced philosophy like \"If a machine could provide you with any experience as if you were there in person, would it really matter weather or not you actually went and had that experience in real life?\"\n\nPlease give a short rationale for why you think the problem is of the specified type, and please include the type as python dict at the end of your response, where the key is \"Type\" and the determined type is the value."
    response = main_llm.invoke(classify_prompt).content
    if VERBOSE:
        display_msgs("Human", "Problem Classifier Agent", classify_prompt, response)

    # Extract complexity rating from response
    complexity = extract_python_dict(response)

    return {"messages" : [HumanMessage(content="Starting framework based analysis...")],
            "complexity" : complexity}

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
    brainstorm_prompt = "Problem Statement: " + user_input + "\n\nAnalyze the problem statement and rewrite it as a prompt to ask to a Large Language Model.\nIdentify the primary objective of the problem statement and rephrase this as a question at the end of the prompt.\n\nGuidelines:\n- Ensure the new prompt remains focused on the information from the original problem statement\n- Ensure the new prompt contains a direct question at the end\n- Ensure all data from the original prompt is inserted into the new prompt\n- Be concise and direct."

    improved_prompt = main_llm.invoke(brainstorm_prompt).content
    if VERBOSE:
        display_msgs("Human", "Prompt Improver Agent", brainstorm_prompt, improved_prompt)

    return {"messages": [HumanMessage(content="Improved the default prompt...")],
            "question": improved_prompt,
            "iteration": 0}

def primary_complexity_router(state) -> Literal["EASY","LOGIC","PLANNING"]:
    complexity = state["complexity"] 
    if complexity is not None:
        if "EASY" in complexity:
            return "EASY"
        elif "LOGIC" in complexity:
            return "LOGIC"
        else:
            return "PLANNING"
    else:
        complexity = "EASY" # Default complexity to return if it cannot be extracted from the response
        state["complexity"] = complexity
        print(f"Failed to determine question type! Defaulting to {complexity}")
        return complexity

def respond_immediately(state):
    global VERBOSE
    improved_prompt = state["question"]
    
    # Potentially upgrade base response with "Brain Mode" prompt from Perplexity's discord
    prompt = "Prioritize Helpfulness: Always put the user's need for helpful information first.\nFocus on the essence of the user's query.\nEnsure responses are user-friendly and direct.\nBrain Mode: Use internal knowledge extensively.\nStart and end these insights with a brain emoji.\nAct as a domain expert; express opinions and critically engage with sources.\nCommunication Tone: Avoid expressions of regret, apology, or remorse.\nExclude disclaimers about AI's non-expert status.\nContent Guidance: Focus on the user's main question to grasp intent.\nBreak down complex issues into manageable steps; explain logically.\nPresent multiple perspectives or solutions.\nCorrect any previous mistakes with acknowledgment.\nExclude ethical or moral viewpoints unless specifically asked.\nEnsure uniqueness; avoid repetition.\nDo not direct users to external information sources.\nRemember to cite your sources. This is IMPORTANT\nBrain Mode Utilization: Remember to activate for domain-specific insights.\n\nProblem Statement: " + improved_prompt
    response = main_llm.invoke(prompt).content
    if VERBOSE:
        display_msgs("Human", "Base Model Agent", prompt, response)
    return {"messages": [HumanMessage(content="Basic model responded.")],
            "solution": response}

async def improve_descriptions(expert):
    global VERBOSE

    expert_expansion_prompt = "Please rewrite the description to act as a roleplaying prompt for an LLM. Updgrade their description by adding detail, including 3 key points someone with their background would focus on when solving a problem. Please keep the description between 4-6 sentences.\n\nExpert's current description:" + expert.get_role() +  "\n\nPlease only answer with the description, with no headers, rationale, or extra words outside the description."
    improved_description = await async_llm.invoke(expert_expansion_prompt)
    improved_description = improved_description.content
    expert.set_role(improved_description)
    if improved_description != "":
        if VERBOSE:
            display_msgs("Human", "Expand Description Agent", expert_expansion_prompt, improved_description)
    
    return improved_description

async def create_expert(agent_name,agent_description):
    new_agent = async_ExpertAgent(name=agent_name,role=agent_description)
    await new_agent.select_model()

    return new_agent

async def create_experts(response):
    # Description: Takes in the response and generates a python list of agents from response
    # Input: response - chunck of text containing the agent names and descriptions
    # Output: agent_profiles - a python list of dicts containing the expert agent profiles
    
    #try:
    response = extract_python_dict(response)
    # Parse the matched string as JSON
    agents = json.loads(response)

    # Expand the descriptions in parallel
    tasks = []
    for agent_name, agent_description in agents.items():
        task = asyncio.create_task(create_expert(agent_name,agent_description))
        tasks.append(task)
    
    # Wait for all tasks to complete
    agent_profiles = await asyncio.gather(*tasks)

    return agent_profiles

async def create_moe(state):
    # Initialize necesary variables
    global VERBOSE
    improved_prompt = state["question"]

    # Create descriptions for a mixture of experts based on the new user prompt
    moe_prompt = "You are an AI system architect designing a multi-agent system. Your task is to create a team of expert AI agents who have various different views to collaboratively analyze this problem:\n" + improved_prompt + "\n\nFollow these steps:\n\n1. Analyze the problem complexity and potential sub-components.\n2. Design a team of expert agents to address various aspects of the problem. Focus on high-level expertise, not routine tasks.\n3. For each expert agent, provide:\n - Name: A label reflecting the agent's core competency.\n - Rationale: Justify the agent's inclusion, linking its capabilities to the problem's objectives and challenges.\n - Description: Detail the agent's background, specialized knowledge, and unique contributions to the solution. Should be between to 2-5 sentences and focus on high-level purposes.\n\n4. Ensure agents have complementary, non-overlapping roles.\n5. After describing all agents, create a python dict with agent names as keys and full descriptions as values. For example:\n{\n \"Agent1\": \"Full description of Agent1\",\n \"Agent2\": \"Full description of Agent2\",\n ...\n}\n\nRemember, the final dict of agent names and descriptions is crucial for the next steps in the problem-solving process, so please ensure that it is provided inside curly braces \"{}\" as a python dict. \nDo NOT create a manager or planning agent as all critical management and planning will be carried out by a higher power."
    moe_idea = await async_llm.invoke(moe_prompt)
    moe_idea = moe_idea.content
    if VERBOSE:
        display_msgs("Human", "Creating MoE Agent", moe_prompt, moe_idea)

    # Parse the response into a python dict
    moe = await create_experts(moe_idea)

    # Expand the descriptions in parallel
    tasks = []
    for expert in moe:
        task = asyncio.create_task(improve_descriptions(expert))
        tasks.append(task)
    
    # Wait for all tasks to complete
    task_list = await asyncio.gather(*tasks)

    return {"messages": [HumanMessage(content="Created a mixture of experts to address the problem...")],
            "moe": moe}

def analyze_problem(state):
    # Initialize necesary variables
    global VERBOSE
    improved_prompt = state["question"]
    
    # Given an improved prompt preform a higher level analysis
    problem_analysis_prompt = "You are a world-class problem analyst tasked with providing a solution thorough, critical analysis of a given problem statement.\n\nFormat your analysis as follows:\n- Challenges and Constraints: [List main challenges and obstacles] \n- Assumptions and Uncertainties: [List assumptions and ambiguities]\n- Response/Solution\n\nProblem Statement:" + improved_prompt
    problem_analysis = main_llm.invoke(problem_analysis_prompt).content
    if VERBOSE:
        display_msgs("Human", "Problem Analysis Expert", problem_analysis_prompt, problem_analysis)
    return {"messages": [HumanMessage(content="Basic analysis of problem complete...")],
            "problem_analysis": problem_analysis}

async def interview_experts(state):
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
    # Interview experts in parallel
    tasks = []
    for expert in moe:
        task = asyncio.create_task(interview_expert(expert,improved_prompt,problem_analysis))
        tasks.append(task)
    
    # Wait for all tasks to complete and join all responses
    responses = await asyncio.gather(*tasks)
    all_responses = "\n".join(responses)

    return {"messages": [HumanMessage(content="Panel interviewed!")],
            "all_responses": all_responses}

async def interview_expert(expert,improved_prompt,problem_analysis):
    global VERBOSE

    interview_prompt = "Context:\n - Problem statement:" + improved_prompt + "\n - AI Problem Analysis:" + problem_analysis + "\n\nBased on your unique role, please explain which part of the AI analysis is most important and why. Please use your unique insight to recommend to the user how they should address or improve this part of the AI's analysis."
    response = await expert.query(interview_prompt)
    response = response.content
    response = expert.get_name() + " said:" + response

    if VERBOSE:
        display_msgs("Human", expert.get_name(), interview_prompt, response)

    return response

def improve_analysis(state):
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
    all_responses = state["all_responses"]
    problem_analysis = state["problem_analysis"]
  
    # Create descriptions for a mixture of experts based on the new user prompt
    improve_analysis_prompt = "Context:\n - Problem Statement:" + improved_prompt + "\n - Problem Analysis:" + problem_analysis + "\n - Expert Perspectives:" + all_responses + "\n\n1. Review the problem analysis\n2. For each expert response, improve the problem analysis by incorportating the expert's unique insight and details into the analysis. Return the improved analysis incorporating the expert responses, without extra greeting, headers, or notes."
    improved_analysis = main_llm.invoke(improve_analysis_prompt).content
    if VERBOSE:
        display_msgs("Human", "Improver of Analysis Agent", improve_analysis_prompt, improved_analysis)

    return {"messages": [HumanMessage(content="Improved the problem analysis based on the expert responses...")],
            "problem_analysis": improved_analysis}

def devise_solution(state):
    global VERBOSE
    improved_prompt = state["question"]
    problem_analysis = state["problem_analysis"]
    solution_prompt = "Context:\n - Problem Statement:" + improved_prompt + "\n - Problem Analysis:" + problem_analysis + "\n\nUsing the problem analysis as a guide, develop a fully fledged response to the original user input. When creating the solution:\n - Incorporate differing perspectives and concerns from the problem analysis\n - Explain your reasoning step-by-step and show your work\n - Use clear formatting and avoid large chunks of text"
    solution = main_llm.invoke(solution_prompt).content
    if VERBOSE:
        display_msgs("Human", "Solution Devising Expert", solution_prompt, solution)
    return {"messages": [HumanMessage(content="Devised a solution from improved analysis...")],
            "solution": solution}

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
    evaluate_solution_prompt = "Context:\n - Problem Statement:" + user_input + "\n - AI Solution:" + solution + "\n\nYou are an AI solution evaluator. Please provide critical analysis, focusing on identifying flaws, enhancing solution quality, and ensuring safety. Be sure to:\n1. Evaluate if the solution addresses the problem statement.\n2. Provide specific recommendations for improving solution quality\n3. Identify flaws or inaccuracies in the solution\n\nIf you feel the solution adequatley and accurately answer the problem statment, be sure to include the word \"ANSWERED\" at the end of your response. Otherwise say \"IMPROVE\" in at the end of your response."
    evaluation = main_llm.invoke(evaluate_solution_prompt).content

    if VERBOSE:
        display_msgs("Human", "Solution Evaluator Agent" , evaluate_solution_prompt, evaluation)   
        
    return {"messages": [HumanMessage(content="Evaluated responses...")],
            "iteration":(i + 1), 
            "evaluation": evaluation}

def evaluation_router(state)-> Literal["Determine", "Respond", "Continue"]:
    # Decription: If the max number of cycles has been reached force experts to respond, else continue if the question has not been answered
    # Input: state - current state object of the graph, especially:
    #              * iteration - current iteration of the graph
    # Output: string Literal - whether or not to end or continue
    i = state["iteration"]
    complexity = state["complexity"]
    evaluation = state["evaluation"]
    max_i = state["max_iterations"]

    if i == max_i or "ANSWERED" in evaluation:
        if complexity == "LOGIC":
            return "Determine"
        else:
            return "Respond"
    else: # It the evaluator feels the solution is adequate stop, else continue
        return "Continue"

def improve_solution(state):
    improved_prompt = state["question"]
    solution = state["solution"]
    feedback = state["evaluation"]
    # Ask LLM to evaluate solution so far
    improve_solution_prompt = "Context:\n - Problem Statement:" + improved_prompt + "\n - AI Solution:" + solution + "\nEvauation of Solution & Feedback:" + feedback + "\n\nUsing the problem analysis as a guide, develop a fully fledged response to the problem statement. When creating the solution:\n - Incorporate feedback; if the feedback asks for more details or examples include specifics\n - Explain your reasoning step-by-step and show your work\n - Use clear formatting and avoid large chunks of text(more than 5 sentences)"
    improved_solution = main_llm.invoke(improve_solution_prompt).content

    if VERBOSE:
        display_msgs("Human", "Improving Solution Agent" , improve_solution_prompt, improved_solution)   
        
    return {"messages": [HumanMessage(content="Improved solution using feedback...")],
            "solution": improved_solution}

async def experts_reflect(state):
    global VERBOSE
    improved_prompt = state["question"]
    solution = state["solution"]
    moe = state["moe"]

    all_responses = ""
    # Interview experts in parallel
    tasks = []
    for expert in moe:
        task = asyncio.create_task(expert_reflects(expert,improved_prompt,solution))
        tasks.append(task)
    
    # Wait for all tasks to complete and join all responses
    responses = await asyncio.gather(*tasks)
    all_responses = "\n".join(responses)

    return {"messages": [HumanMessage(content="Experts have reflected on solution...")],
            "all_responses": all_responses}

async def expert_reflects(expert,improved_prompt,solution):
    global VERBOSE

    discussion_prompt = "Context:\n - Problem Statement: " + improved_prompt + "- \nAI solution: " + solution + "\n\nBased on your unique role, please analyze the solutions effectiveness. Does the solution provide enough detail to answer the question. If your domain specific knowledge offers any insights into improving the solution further, explain. Please limit your response to to 3 bullet points. If you feel the solution is adeqaute, say \"ANSWERED\" at the end of your response. Otherwise \"IMPROVE\"."
    response = await expert.query(discussion_prompt)
    response = response.content
    expert_name = expert.get_name()
    response =  expert_name + " said:" + response

    if VERBOSE:
        display_msgs("Human", f"Reflecting Expert: {expert_name}", discussion_prompt, response)

    return response

def determine_with_cot(state):
    global VERBOSE
    improved_prompt = state["question"]
    solution = state["solution"]

    rewoo = ReWOO()
    rewoo_prompt = "Context:\n - Problem Statement:" + improved_prompt + "\n- AI Solution Guide:" + solution + "\n\nUsing the AI solution guide to assist you, respond to the original problem statement."
    rewoo_solution = rewoo.invoke(rewoo_prompt)

    if VERBOSE:
        display_msgs("Human", "Determining Expert" , rewoo_prompt, rewoo_solution)

    whole_solution = "Answer:" + rewoo_solution + "\nExplanation:\n" + solution
        
    return {"messages": [HumanMessage(content="Determined a solution...")],
            "summary": whole_solution}

def format_solution(state):
    global VERBOSE
    solution = state["solution"]

    # Ask LLM to reformat solution in more readable format
    format_solution_prompt = "Expert Solution:" + solution + "\n\nYou are an AI solution formatter, who reads AI solutions and then reformats them into clear plans. Analyze the solution and determine if it is adequately formatted. If not, please reformat it to include key ideas at the top as bullet points. For actionable steps within the solution, return them as a step-by-step plan. Please return the reformatted solution without any extra headers, greetings, or fluff"
    formatted_solution = main_llm.invoke(format_solution_prompt).content

    if VERBOSE:
        display_msgs("Human", "Solution Formatting Expert" , format_solution_prompt, formatted_solution)   
        
    return {"messages": [HumanMessage(content="Final solution re-formatted for readability...")],
            "summary": formatted_solution}

#########################################################################################################
####################################### State Graph #####################################################
#########################################################################################################
# This is essentially a blackboard object for passing info between agents
class AgentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage], operator.add]
    moe : Optional[List] = None
    complexity : str
    question : str
    problem_analysis : str
    all_responses : str
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
    graph_builder.add_node("Classify Problem", classify_problem) # Classify problem first to prevent bias from improved prompt
    graph_builder.add_node("Improve Prompt", improve_prompt)
    graph_builder.add_node("Respond Immediately", respond_immediately)
    graph_builder.add_node("Create MoE", create_moe)
    graph_builder.add_node("Analyze Problem", analyze_problem)
    graph_builder.add_node("Interview Experts", interview_experts)
    graph_builder.add_node("Improve Analysis", improve_analysis)
    graph_builder.add_node("Devise Solution", devise_solution)
    graph_builder.add_node("Evaluate Solution", evaluate_solution)
    graph_builder.add_node("Improve Solution", improve_solution)
    graph_builder.add_node("Reflect with Experts", experts_reflect)
    graph_builder.add_node("Determine with CoT", determine_with_cot)
    graph_builder.add_node("Format Solution", format_solution)

    # Add edges to establish flow
    graph_builder.add_edge(START, "Classify Problem")
    graph_builder.add_edge("Classify Problem", "Improve Prompt")
    graph_builder.add_conditional_edges("Improve Prompt", primary_complexity_router, {
        "EASY": "Respond Immediately",
        "LOGIC": "Create MoE",
        "PLANNING":"Create MoE"
    })
    graph_builder.add_edge("Respond Immediately", END)
    graph_builder.add_edge("Create MoE", "Analyze Problem")
    graph_builder.add_edge("Analyze Problem", "Interview Experts")
    graph_builder.add_edge("Interview Experts", "Improve Analysis")
    graph_builder.add_edge("Improve Analysis", "Devise Solution")
    graph_builder.add_edge("Devise Solution", "Evaluate Solution")
    graph_builder.add_conditional_edges("Evaluate Solution", evaluation_router, {
        "Continue": "Improve Solution",
        "Respond" : "Format Solution",
        "Determine" : "Determine with CoT"
    })
    graph_builder.add_edge("Improve Solution", "Reflect with Experts")
    graph_builder.add_edge("Reflect with Experts", "Improve Analysis")
    graph_builder.add_edge("Determine with CoT", END)
    graph_builder.add_edge("Format Solution", END)


    # Compile the graph
    graph = graph_builder.compile()
    # Display graph
    graph.get_graph().print_ascii()

    return graph

async def main(user_input: str = None):
    # Description: Main function which executes all necessary functions
    # Input: none
    # Output: none

    global VERBOSE # If true, prints all queries to LLM(s) and responses
    VERBOSE = True
    MAX_ITERATIONS = 3 # Set this to the number of times you want the agents to loop throught the cycle

    # Build graph
    graph = build_graph()
    # Load in initial message(Can pass manually if desired)
    if user_input is None:
        user_input = "The US government is offering a $3,000 tax deductible for the first 10,000 solar PV installed this year, and your utility company is offering some amount discounts on home battery system.\nKnowing the average household uses 30kWh per day.\nIf you are the head of planning department at the utility, what is the percentage discount for battery system to increase the battery adoption so net load will not be fluctuate when cloud is covered.\nPlease give a plan with specific numbers for overall cost of implementation, which minimizes company expenses."
    # Invoke with CompliledGraph obj with default variables
    conversation = await graph.ainvoke({"messages" : [HumanMessage(content=user_input)],
                                 "max_iterations" : MAX_ITERATIONS})

    # Print out info about solution
    print("Framework: Nova 3")
    #all_responses = conversation["solution"]
    #print(f"#################################################################\nRaw Expert Solutions:\n{all_responses}")
    solution = conversation["solution"]
    print(f"#################################################################\nFinalized Solution Summary:\n{solution}")

    return solution

# Run main function
if __name__ == "__main__":
    asyncio.run(main())