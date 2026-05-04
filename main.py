from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
tools = [TavilySearch(max_results=3)]
agent = create_agent(model=llm, tools=tools)

def main():
    print("Hello from langchain-course!")
    result = agent.invoke({
        "messages": [HumanMessage(content="What is the actual weather in Beijing?")]
    })
    print(result)


if __name__ == "__main__":
    main()
