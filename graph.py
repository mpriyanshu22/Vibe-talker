import os
import subprocess
from typing import Annotated
from langchain_core.tools import tool
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START

class State(TypedDict):
    messages: Annotated[list, add_messages]

# Base directory where your code lives
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
CHAT_GPT_DIR = os.path.join(os.path.dirname(BASE_DIR), "chat_gpt")
os.makedirs(CHAT_GPT_DIR, exist_ok=True)

@tool
def run_command(cmd: str) -> str:
    """Runs a Windows PowerShell or CMD command locally and returns output."""
    try:
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        output = res.stdout if res.returncode == 0 else res.stderr
        return output if output else f"Command completed with exit code {res.returncode}"
    except Exception as e:
        return f"Error executing command: {str(e)}"

llm = init_chat_model(
    model="gemini-3.1-flash-lite",
    model_provider="google_genai"
)
llm_with_tool = llm.bind_tools(tools=[run_command])

def chatbot(state: State):
    # Pass the real absolute path so there is ZERO ambiguity
    system_prompt = f"""You are ChaiCode Agent, running LOCALLY on the user's Windows machine.
You have direct local access to the user's hard drive.
Target folder for all generated files: {CHAT_GPT_DIR}

CRITICAL RULES:
1. You are NOT in a cloud sandbox. You execute commands directly on the user's local Windows OS.
2. When creating files with `run_command`, DO NOT write multiline python strings with '#' comments.
3. Keep Python one-liners on a single continuous line using semicolons, or use PowerShell:
   python -c "import os; open(r'{CHAT_GPT_DIR}\\\\<filename>', 'w', encoding='utf-8').write('''<content>''')"
"""
    message = llm_with_tool.invoke([{"role": "system", "content": system_prompt}] + state["messages"])  
    return {"messages": [message]}

tool_node = ToolNode(tools=[run_command])

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")

def create_chat_graph(checkpointer=None):
    return graph_builder.compile(checkpointer=checkpointer)