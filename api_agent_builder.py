### Agent Builder ###

# Default imports
import re, json

# Default set of models using perplexity labs
from api_perplexity import fake_LLM
sonar = fake_LLM(model="llama-3-sonar-large-32k-online", sleep_time=60)
mixtral = fake_LLM(model="mixtral-8x7b-instruct", sleep_time=60)
llama3_70b = fake_LLM(model="llama-3-70b-instruct", sleep_time=60)
llama3_8b = fake_LLM(model="llama-3-8b-instruct", sleep_time=60)

llm = sonar# Set base model for agent selector to reason with

default_models = [{"Name":"Online Sonar", "Model":sonar, "Description":"Online Sonar is a variant of the Llama-3 model, optimized for online applications and search-enabled tasks. It is designed for high-performance and efficiency.\nPros: Optimized for online use and has search-enabled; can get up-todate information and data with no knowledge cutoff.\nCons: It may potentially find incorrect information on the internet.\nBest for: Real-time applications, search-enabled tasks, and projects requiring up-to-date information to be pulled from the internet. Please use this if the task requires up-to-date information, such as answering \"Who won last nights baseball game?\" or \"What are the election results?\""},
                  {"Name":"Mixtral", "Model":mixtral, "Description":"Mixtral is a flagship model from Nous Research, trained on the Mixtral 8x7B MoE LLM. It is known for its exceptional performance in multimodal tasks and dialogue use cases.\nPros: High performance in multimodal tasks, efficient architecture, and strong dialogue capabilities.\nCons: Has a knowledge cutoff of 2021.\nBest for: Applications requiring advanced multimodal processing and dialogue handling. Please use this when asked about the arts and philosophy over any other model"},
                  {"Name":"Llama-3-8B", "Model":llama3_8b, "Description":"Llama-3-8B is a large language model from Meta, optimized for dialogue use cases and available in open-source format. It has been shown to outperform many open-source chat models on industry benchmarks.\nPros: Highly versatile and performs well in dialogue and multimodal tasks.\nCons: May require significant computational resources, and it has a knowledge cutoff of 2022.\nBest for: Projects requiring high-performance dialogue capabilities, customization, and flexibility. Please use this when asked about math, physics, and coding, which typically do not require the most up-to-date information."},
                  {"Name":"Llama-3-70B", "Model":llama3_70b, "Description":"Llama-3-70B is a large language model from Meta, optimized for dialogue use cases and available in open-source format. It has been shown to outperform many open-source chat models on industry benchmarks.\nPros: Requires less resources than Llama-3-70b, versatile, and performs adequately in dialogue and multimodal tasks.\nCons: May not achieve same response level as LLama-3-70b, and it has a knowledge cutoff of 2022.\nBest for: Projects requiring high-performance dialogue capabilities, customization, and flexibility, while also not consuming too many computational resources."}]

def extract_model_name(text):
    # Use re.search to find the pattern in the text
    #pattern = r'{\s*("[^"]*"\s*:\s*"[^"]*"\s*(?:,\s*"[^"]*"\s*:\s*"[^"]*"\s*)*)}'
    pattern = r"[{\[].*[}\]]"
    match = re.search(pattern, text, re.DOTALL)
    
    model_name = "Llama-3-70B" # Defualt output model name
    if match:
        dict_str = match.group(0)
        dict_str = dict_str.replace("[","{")
        dict_str = dict_str.replace("]","}")
        dict_str = dict_str.replace("'",'"')
        try:
            # Parse the matched string as JSON
            model = json.loads(dict_str)
            model_name = model["Name"]
        except json.JSONDecodeError:
            print(f"Json was improperly formatted: {dict_str}. Returning default: {model_name}!")
    else:
        print(f"Failed to find model name in response. Returning default: {model_name}!")  

    return model_name  # Return None if JSON parsing fails

def determine_best_llm(role, available_models):
    llm_descriptions = ""
    for model in available_models:
        model_name = model["Name"]
        model_description = model["Description"]
        llm_descriptions = llm_descriptions + "Model Name: " + model_name + "Description:" + model_description + "\n" 

    # Ask the LLM which model it thinks will work best  
    prompt = f"""Task: Select the most suitable Large Language Model (LLM) for an AI expert with a specific role.\n\nGiven information:\n1. Expert's Role:{role}\n2. Available LLMs: {llm_descriptions}\n\nSelection criteria:\n1. Up-to-date information: Consider how recent the LLM's knowledge is.\n2. Specialized information: Evaluate if the LLM has unique training in areas relevant to the expert's role.\n3. Advanced reasoning capabilities: Assess the LLM's ability to perform complex reasoning tasks.\n\nInstructions:\n1. Analyze the expert's role and its requirements.\n2. Review the descriptions of available LLMs.\n3. Compare each LLM's features against the selection criteria.\n4. Choose the LLM that best matches the expert's needs, balancing all factors.\n5. Briefly explain your selection, highlighting how the chosen LLM's capabilities align with the expert's role.\n\nNote: Prioritize the most critical aspects for the expert's role when making your decision.\nPlease include the specified model name a python dict at the end of your response, where the string "Name" is the key and the name of selected model is the value. Please be sure to include the model name in the correct format at the end of your response as it is critical for the next steps in the process"""
    prompt = prompt.format(role=role,llm_descriptions=llm_descriptions)
    response = llm.invoke(prompt)
    #print("Response:", response)

    # Find the model name the llm returned
    desired_model_name = extract_model_name(response)

    for available_model in available_models:
        available_model_name = available_model["Name"]
        if available_model_name == desired_model_name:
            return available_model["Model"]

class ExpertAgent():
    def __init__(self, name: str, role: str, context: str = "", model = None, models_list: list = None):
        self.name = name
        self.role = role
        self.context = "Your name: " + name + "\nA description of your role: " + role + "\nContext: " + context
        if model is None:
            if models_list is None:
                self.model = determine_best_llm(self.role, default_models)
            else:
                self.model = determine_best_llm(self.role, models_list)
        else:
            self.model = model

    def get_name(self):
        return self.name
    def get_model(self):
        return self.model
    def get_context(self):
        return self.context
    def get_role(self):
        return self.role
    
    def set_model(self,model):
        self.model = model
        return
    def set_context(self,context):
        self.context = "Context:\n" + context + "\n\nYour name:" + self.name + "\nA description of your role:" + self.role 
        return
    def set_role(self,role):
        self.role=role
        return

    def query(self, input_query):
        final_query = self.context + "\nRespond based on your role and the given context:" + input_query
        response = self.model.invoke(final_query)
        return response

# Default set of models using perplexity labs
from api_perplexity import async_fake_LLM
async_sonar = async_fake_LLM(model="llama-3-sonar-large-32k-online", sleep_time=60)
async_mixtral = async_fake_LLM(model="mixtral-8x7b-instruct", sleep_time=60)
async_llama3_70b = async_fake_LLM(model="llama-3-70b-instruct", sleep_time=60)
async_llama3_8b = async_fake_LLM(model="llama-3-8b-instruct", sleep_time=60)

async_llm = async_sonar# Set base model for agent selector to reason with

async_default_models = [{"Name":"Online Sonar", "Model":async_sonar, "Description":"Online Sonar is a variant of the Llama-3 model, optimized for online applications and search-enabled tasks. It is designed for high-performance and efficiency.\nPros: Optimized for online use and has search-enabled; can get up-todate information and data with no knowledge cutoff.\nCons: It may potentially find incorrect information on the internet.\nBest for: Real-time applications, search-enabled tasks, and projects requiring up-to-date information to be pulled from the internet. Please use this if the task requires up-to-date information, such as answering \"Who won last nights baseball game?\" or \"What are the election results?\""},
                    {"Name":"Mixtral", "Model":async_mixtral, "Description":"Mixtral is a flagship model from Nous Research, trained on the Mixtral 8x7B MoE LLM. It is known for its exceptional performance in multimodal tasks and dialogue use cases.\nPros: High performance in multimodal tasks, efficient architecture, and strong dialogue capabilities.\nCons: Has a knowledge cutoff of 2021.\nBest for: Applications requiring advanced multimodal processing and dialogue handling. Please use this when asked about the arts and philosophy over any other model"},
                    {"Name":"Llama-3-8B", "Model":async_llama3_8b, "Description":"Llama-3-8B is a large language model from Meta, optimized for dialogue use cases and available in open-source format. It has been shown to outperform many open-source chat models on industry benchmarks.\nPros: Highly versatile and performs well in dialogue and multimodal tasks.\nCons: May require significant computational resources, and it has a knowledge cutoff of 2022.\nBest for: Projects requiring high-performance dialogue capabilities, customization, and flexibility. Please use this when asked about math, physics, and coding, which typically do not require the most up-to-date information."},
                    {"Name":"Llama-3-70B", "Model":async_llama3_70b, "Description":"Llama-3-70B is a large language model from Meta, optimized for dialogue use cases and available in open-source format. It has been shown to outperform many open-source chat models on industry benchmarks.\nPros: Requires less resources than Llama-3-70b, versatile, and performs adequately in dialogue and multimodal tasks.\nCons: May not achieve same response level as LLama-3-70b, and it has a knowledge cutoff of 2022.\nBest for: Projects requiring high-performance dialogue capabilities, customization, and flexibility, while also not consuming too many computational resources."}]

async def async_determine_best_llm(role, available_models):
    llm_descriptions = ""
    for model in available_models:
        model_name = model["Name"]
        model_description = model["Description"]
        llm_descriptions = llm_descriptions + "Model Name: " + model_name + "Description:" + model_description + "\n" 

    # Ask the LLM which model it thinks will work best  
    prompt = f"""Task: Select the most suitable Large Language Model (LLM) for an AI expert with a specific role.\n\nGiven information:\n1. Expert's Role:{role}\n2. Available LLMs: {llm_descriptions}\n\nSelection criteria:\n1. Up-to-date information: Consider how recent the LLM's knowledge is.\n2. Specialized information: Evaluate if the LLM has unique training in areas relevant to the expert's role.\n3. Advanced reasoning capabilities: Assess the LLM's ability to perform complex reasoning tasks.\n\nInstructions:\n1. Analyze the expert's role and its requirements.\n2. Review the descriptions of available LLMs.\n3. Compare each LLM's features against the selection criteria.\n4. Choose the LLM that best matches the expert's needs, balancing all factors.\n5. Briefly explain your selection, highlighting how the chosen LLM's capabilities align with the expert's role.\n\nNote: Prioritize the most critical aspects for the expert's role when making your decision.\nPlease include the specified model name a python dict at the end of your response, where the string "Name" is the key and the name of selected model is the value. Please be sure to include the model name in the correct format at the end of your response as it is critical for the next steps in the process"""
    prompt = prompt.format(role=role,llm_descriptions=llm_descriptions)
    response = await async_llm.invoke(prompt)
    response = response.content
    #print("Response:", response)

    # Find the model name the llm returned
    desired_model_name = extract_model_name(response)

    for available_model in available_models:
        available_model_name = available_model["Name"]
        if available_model_name == desired_model_name:
            return available_model["Model"]

class async_ExpertAgent():
    def __init__(self, name:str, role:str, context:str = "", model = None):
        self.name = name
        self.role = role
        self.context = context
        self.model = model

    def get_name(self):
        return self.name
    def get_model(self):
        return self.model
    def get_context(self):
        return self.context
    def get_role(self):
        return self.role
    
    async def select_model(self):
        self.model = await async_determine_best_llm(self.role, async_default_models)
        if self.model is None:
            print("Could not determine model using func")
            self.model = async_llm
        return
    # def set_model(self, model):
    #     self.model = model
    #     return
    def set_context(self,context):
        self.context = "\nContext:\n" + context
        return
    def set_role(self,role):
        self.role=role
        return

    async def query(self, input_query):
        final_query = "Role: " + self.role +  self.context + "\n\nRespond based on your role and the given context:" + input_query
        response = await self.model.invoke(final_query)
        return response