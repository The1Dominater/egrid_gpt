### Reason Without Observation ###

import os, re
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "ReWOO"
os.environ['LANGCHAIN_API_KEY'] = ""
os.environ["TAVILY_API_KEY"] = ""

from typing import List, TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, END, StateGraph
from api_perplexity import fake_LLM
from langchain_community.tools.tavily_search import TavilySearchResults

class ReWOOState(TypedDict):
    task: str
    plan_string: str
    steps: List
    results: dict
    result: str

class ReWOO():
    def __init__(self):
        self.llm = fake_LLM(model="llama-3-sonar-large-32k-online", sleep_time=60)
        self.search = fake_LLM(model="llama-3-sonar-large-32k-online", sleep_time=30)

    def get_plan(self, state):
        plan_prompt = """Task: {task}
For the given task, make plans that can solve the problem step by step. For each plan, indicate \
which external tool together with tool input to retrieve evidence. You can store the evidence into a \
variable #E that can be called by later tools. (Plan, #E1, Plan, #E2, Plan, ...)

Tools can be one of the following:
(1) Google[input]: Worker that searches results from Google. Useful when you need to find short
and succinct answers about a specific topic. The input should be a search query.
(2) LLM[input]: A pretrained LLM like yourself. Useful when you need to act with general
world knowledge and common sense. Prioritize it when you are confident in solving the problem
yourself. Input can be any instruction.

For example,
Task: What is the hometown of the 2024 Australian Open Mens-Singles champion?
Plan: Determine who won the 2024 Australian Open Mens-Singles. #E1 = Google[Who won the 2024 Australian Open Mens-Singles]
Plan: Find information about #E1's hometown online. #E2 = Google[What is #E1's hometown]
Plan: Determine the hometown of #E1 using #E2. #E3 = LLM[Search #E2 for the hometown of #E1]

Describe your plans with rich details. Each Plan should be followed by only one #E.
Please ensure each line of your response containing plans is formatted extactly as follows:
Plan: [Action to complete] #E = <tool>[<tool input>]

Remember to follow the format as it is essentially for the next steps in the process."""
        plan_pattern = r"Plan:\s*(.+)\s*(#E\d+)\s*=\s*(\w+)\s*\[([^\]]+)\]"
        task = state["task"]
        # Generate a plan in the specified format
        result = self.llm.invoke(plan_prompt.format(task=task)).content
        # Find all matches in the sample text
        matches = re.findall(plan_pattern, result)
        # If the plan can be parsed from the response
            
        return {"steps": matches, "plan_string": result}

    def _get_current_task(self, state):
        if state["results"] is None:
            return 1
        elif len(state["results"]) == len(state["steps"]):
            return None
        else:
            return len(state["results"]) + 1

    def tool_execution(self, state):
        """Worker node that executes the tools of a given plan."""
        for attempts in range(3):
            try:
                _step = self._get_current_task(state)
                _, step_name, tool, tool_input = state["steps"][_step - 1]
                break
            except:
                print("Error: Num steps:", len(state["steps"]), "step index-",_step)
                self.get_plan(state)
            
        _results = state["results"] or {}
        for k, v in _results.items():
            tool_input = tool_input.replace(k, v)
        if tool == "Google":
            result = self.search.invoke(tool_input).content
        elif tool == "LLM":
            result = self.llm.invoke(tool_input).content
        else:
            print(tool)
            raise ValueError
        _results[step_name] = str(result)
        return {"results": _results}

    def solve(self, state):
        solve_prompt = """Solve the following task or problem. To solve the problem, we have made step-by-step Plan and \
retrieved corresponding Evidence to each Plan. Use them with caution since long evidence might \
contain irrelevant information.

{plan}

Now solve the question or task according to provided Evidence above. Respond with the answer
directly with no extra words.

Task: {task}
Response:"""
        plan = ""
        for _plan, step_name, tool, tool_input in state["steps"]:
            _results = state["results"] or {}
            for k, v in _results.items():
                tool_input = tool_input.replace(k, v)
                step_name = step_name.replace(k, v)
            plan += f"Plan: {_plan}\n{step_name} = {tool}[{tool_input}]"
        prompt = solve_prompt.format(plan=plan, task=state["task"])
        result = self.llm.invoke(prompt).content
        return {"result": result}

    def _route(self, state):
        _step = self._get_current_task(state)
        if _step is None:
            # We have executed all tasks
            return "solve"
        else:
            # We are still executing tasks, loop back to the "tool" node
            return "tool"

    def invoke(self, task):
        graph_builder = StateGraph(ReWOOState)

        
        graph_builder.add_node("plan", self.get_plan)
        graph_builder.add_node("tool", self.tool_execution)
        graph_builder.add_node("solve", self.solve)

        graph_builder.add_edge(START, "plan")
        graph_builder.add_edge("plan", "tool")
        graph_builder.add_edge("solve", END)
        graph_builder.add_conditional_edges("tool", self._route)
        
        graph = graph_builder.compile()
        
        conversation = graph.invoke({"task": task})

        return conversation["result"]