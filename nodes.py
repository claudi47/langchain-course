from dotenv import load_dotenv
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode

from react import llm, tools

load_dotenv()

SYSYEM_MESSAGE="""
You are a helpful assistant that can use tools to answer questions.
"""

# Implementing ReAct agent reasoning node and tool node

# Agent reasoning node: prende lo stato, antepone il system message e
# chiama l'llm. L'llm decide se chiamare un tool o rispondere direttamente.
# La risposta viene aggiunta alla lista dei messaggi.
def run_agent_reasoning(state: MessagesState) -> MessagesState:
    """
    Run the agent reasoning node.
    """
    response = llm.invoke([{"role": "system", "content": SYSYEM_MESSAGE}, *state["messages"]])
    return {"messages": [response]}

# "Act" node that calls the tools
# È il nodo che esegue le chiamate ai tool se richieste dall'llm
tool_node = ToolNode(tools)