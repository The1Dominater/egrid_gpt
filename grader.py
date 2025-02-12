### Grade's LLM framework responses against base model and SOTA ###
import subprocess, time, datetime, os, json, re, asyncio

# Connect to LLM to grade responses
from api_perplexity import async_fake_LLM
async_llm = async_fake_LLM(model="llama-3-sonar-large-32k-online", sleep_time=80)

# List of script names to grade
import app_nova, app_nova2, app_nova3, app_rewoo, app_moe, app_mase

def load_qa(questions_path:str = ""):
    # Description: Loads a list of all inputs and targets as strs from json file
    # Input: Path to json file
    # Output: python list containing {"input" : "prompt", "target" : "response"} pairs
    with open(questions_path) as f:
        qna_dict = json.load(f)
    
    return qna_dict["examples"]

def extract_grade(grader_response:str):
    # Extracts the grade from the response if possible
    try:
        # First grab the grade portion of the response
        pattern = r"[{\[]?['\"]?grade['\"]?\s?:\s?.*[}\]]?"
        grade_str = re.search(pattern, grader_response)
        grade_str = grade_str.group()
        # Next grab the integer value from the grade
        pattern = r"\d\d?"
        grade_str = re.search(pattern, grade_str).group()
        grade_value = int(grade_str)
        
    except:
        grade_value = None
        print(f"Error: grade could not be retrieved! Grader said:{grader_response}")
    
    return grade_value

def store_grade(framework_name:str,question:str,grade,ttf:float,grader_analysis:str="",target_response:str="",base_response:str="",framework_response:str=""):
        # Generate current date and time
        current_datetime = datetime.datetime.now()
        # Format the datetime as a string to include in the file name
        datetime_str = current_datetime.strftime("%m-%d_%H-%M")
        # Construct the file name with the timestamp
        question = question.replace(".json","")
        question = question.replace("short_","")
        file_name = f"{question}_{framework_name}_{datetime_str}.md"
        # Convert it to a file path
        path = os.path.expandvars("$PWD/test_output/")
        path = os.path.join(path, file_name)
        #print("Path:", path)
        ttf = str(round(ttf,3) / 60.0)
        final_response = "Framework:" + framework_name + "\nQuestion:" + question + "\nGrade:" + str(grade) + "\nTTF:" + ttf + " minutes\n\n##Grader Analysis:\n" + grader_analysis + "\n\n##Target Response:\n" + target_response + "\n\nBase Grade:" + base_grade + "\nBase Response:" + base_response + "\n\n##Framework Response:\n" + framework_response
        
        # Open the file in write mode ('w')
        with open(path, 'w') as file:
            # Write the string to the file
            file.write(final_response)
        
        return

async def query_framework(input_prompt:str,framework, framework_name:str):
    # try:
    start_time = time.time()
    response = await framework.main(input_prompt)
    ttf = time.time() - start_time
    # except:
    #     print(f"Error with framework: {framework_name}")
    #     response = "ERROR: LLM unable to respond..."
    #     ttf = 0.0
    
    return response, ttf

async def grade_framework(qna_pairs:list,question_file:str,framework,framework_name:str):
    # Initialize variables to track LLM stats for the given problem
    total_ttf = 0
    total_grade = 0
    num_questions = 0

    for pair in qna_pairs:
        input_prompt = pair["input"]
        target_response = pair["target"]
        framework_response, ttf = await query_framework(input_prompt,framework,framework_name)

        # This prompt will give a score between 1 and 10; used for non-deterministic questions with open ended solutions
        grader_prompt = "You are a grading assistant who grades responses. Do NOT respond to the question yourself. An AI was asked this question: " + input_prompt + "\nThe target response is: " + target_response + "\nThe AI responded with: " + framework_response + "\n\nTasks:\n - Please compare the AI response to the target response and determine if it is better, equivalent, or worse.\n - Analyze the AI's response for unique insights, beneficial ideas, and accuracy.\n - Consider how well the AI's response addresses the original question.\n - Grade the AI's response on a scale between 0-10. At the end of your response, please return the final grade as a python dict in the following the format: <curly brace>\n\"grade\":\n\"<grade value>\n\"<curly brace>."
        # Asks the main LLM to grade the response against the target response
        grader_response = await async_llm.invoke(grader_prompt)
        grader_response = grader_response.content
        grade = extract_grade(grader_response)

        base_response = await async_llm.invoke(input_prompt)
        base_response = base_response.content
        grader_prompt = "You are a grading assistant who grades responses. Do NOT respond to the question yourself. Two AI were asked this question: " + input_prompt + "\nThe first AI said: " + base_response + "\nThe second AI said: " + framework_response + "\n\nTasks:\n - Please compare the AI responses and determine if one is better, equivalent, or worse.\n - Analyze each AI's response for unique insights, beneficial ideas, and accuracy.\n - Consider how well the AI's response addresses the original question.\n - Grade the AI's responses on a scale between 0-10. At the end of your response, please return the final grades as a python dict in the following the format: <curly brace>\n\"First AI grade\":\"<Grade for the first AI response>\n,\"Second AI grade\":\"<Grade for the second AI response>\"\n\"<curly brace>."
        # Asks the main LLM to grade the response against the base response
        grader_response = await async_llm.invoke(grader_prompt)
        grader_response = grader_response.content
        if grade is not None:
            num_questions = num_questions + 1
            total_ttf = total_ttf + ttf
            total_grade = total_grade + grade
            # Store the results of grader response
            store_grade(framework_name,input_prompt,grade,ttf,grader_response,target_response,base_response,framework_response)

    avg_grade = total_grade / float(num_questions)
    avg_ttf = total_ttf / float(num_questions)

    return avg_grade, avg_ttf

async def main():
    #frameworks = {"app_nova":app_nova,"app_nova2":app_nova2,"app_nova3":app_nova3,"app_rewoo":app_rewoo}
    frameworks = {"app_nova":app_nova,"app_nova2":app_nova2,"app_nova3":app_nova3}
    input_dir = "./bbh_short"  # Load the question files from the given dir
    #question_files = os.listdir(input_dir)
    #question_files = ["_short_microgrid_management.json", "short_boolean_expressions.json","short_casual_judgement.json"]
    question_files = ["_short_ps_grid_planning.json","_short_microgrid_management"]

    # Check all frameworks answering abilities on the current question
    for framework_name, framework in frameworks.items():
        for question_file in question_files:
            try:
                path = input_dir + "/" + question_file
                qna_pairs = load_qa(path)
            except:
                print(f"Could not find question file:{question_file}")
                continue
            # Grade the frameworks response and time to finish on all problem sets
            avg_grade, avg_ttf = await grade_framework(qna_pairs,question_file,framework,framework_name)

            # Store and display results
            store_grade(framework_name,question_file,f"{avg_grade}(Avg)",f"{avg_ttf}(Avg)","N/A","N/A","N/A")
            print(f"Framework {framework_name} for {question_file} received grade: {avg_grade} in {avg_ttf} minutes")

if __name__ == "__main__":
    asyncio.run(main())