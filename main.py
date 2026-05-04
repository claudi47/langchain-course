from dotenv import load_dotenv

from typing import List
from pydantic import BaseModel, Field

from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_tavily import TavilySearch

load_dotenv()

# https://docs.langchain.com/oss/python/langchain/structured-output
class Source(BaseModel):
    """Schema for a source used by the agent."""

    url:str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema for a source used by the agent"""

    answer: str = Field(description="The answer provided by the agent")
    sources: List[Source] = Field(
        default_factory=list,
        description="The list of sources used to generate the answer")
    

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
tools = [TavilySearch(max_results=3)]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

def main():
    print("Hello from langchain-course!")
    result = agent.invoke({
        # "messages": [HumanMessage(content="What is the actual weather in Beijing?")]
        "messages": [HumanMessage(content="Search for 3 job postings for ai engineer using langchain on LinkedIn and list their details")]
    })
    print(result)


if __name__ == "__main__":
    main()
