from dotenv import load_dotenv


from langchain_core.tools import StructuredTool
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode

from schemas import AnswerQuestion, ReviseAnswer

load_dotenv()

tavily_tool = TavilySearch(max_results=5)


def run_queries(search_queries: list[str], **kwargs):
    """Run the generated queries."""
    return tavily_tool.batch([{"query": query} for query in search_queries])

# Nodo di mezzo tra first_responder e revisor: esegue i tool chiamati
# dall'LLM e aggiunge i risultati (ToolMessage) allo stato.
# La funzione run_queries è unica, ma la registriamo sotto due nomi diversi
# (AnswerQuestion / ReviseAnswer) perché il ToolNode esegue un tool solo se
# il suo nome combacia con quello della tool call prodotta dall'LLM:
# il first_responder chiama "AnswerQuestion", il revisor "ReviseAnswer".
execute_tools = ToolNode(
    [
        StructuredTool.from_function(run_queries, name=AnswerQuestion.__name__),
        StructuredTool.from_function(run_queries, name=ReviseAnswer.__name__),
    ]
)
