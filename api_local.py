
import os
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Setup huggingface cache and login for downloading models
from huggingface_hub import login
#hf_token = os.environ['HF_TOKEN'] # May be required to download models
hf_token = ""
login(token=hf_token)
# model_cache_dir = os.path.expandvars("${HF_HUB_CACHE}/model")
# print(f"Path to model cache:{model_cache_dir}")

# Here are the default models
available_models = ["meta-llama/Meta-Llama-3-8B-Instruct", "microsoft/Phi-3-mini-128k-instruct", "mistralai/Mixtral-8x7B-Instruct-v0.1"]
default_model = "meta-llama/Meta-Llama-3-8B-Instruct"

def initialize_llm(model_name:str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=8192)

    return pipe


class ChatLocal():
    def __init__(self, model_name:str = ""):
        if model_name in available_models:
            print(f"Using model {model_name}...")
            self.pipe = initialize_llm("microsoft/Phi-3-mini-128k-instruct")
        else:
            print(f"Defaulting to {default_model}...")
            self.pipe = initialize_llm(default_model)

    def invoke(self, query:str):
        messages = [{"role": "user", "content":query}]
        response = self.pipe(query)
        return response
