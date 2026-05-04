from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage

load_dotenv()


def search_web(query: str) -> str:
    """
    Tool that searches over internet
    Args:
        query (str): The query to search for
    Returns:
        the search results"""
    print(f"Searching web for: {query}")
    return f"Search results for '{query}'"

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
tools = [search_web]
agent = create_agent(model=llm, tools=tools)

def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": [HumanMessage(content="What is the capital of France?")]})
    print(result)


if __name__ == "__main__":
    main()
