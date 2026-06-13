import datetime
import os

from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq

from schemas import AnswerQuestion, ReviseAnswer

load_dotenv()

llm = ChatGroq(model=os.environ["GROQ_MODEL"])
# serve nel grafo LangGraph, perché quando esegui i tool devi rispondere al modello
# con un ToolMessage che fa riferimento allo stesso tool_call_id (return_id=True)
parser = JsonOutputToolsParser(return_id=True)
# ritorna un oggetto tipizzato, che è più comodo da manipolare nei nodi del grafo
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])

actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert researcher.
Current time: {time}

1. {first_instruction}
2. Reflect and critique your answer. Be severe to maximize improvement.
3. Recommend search queries to research information and improve your answer.""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format."),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat(),
)


first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer."
)

# Nodo principale: definiamo il system prompt dato in pasto all'llm
# L'llm è forzato a rispondere con tool call e a ritornare un messaggio strutturato come AnswerQuestion
first_responder = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion"
)
 
revise_instructions = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""

# Secondo nodo principale: sostituisce il first_instruction e ripassa all'llm il messaggio
# forzandolo a rispondere con tool call di tipo ReviseAnswer
# Nota: questo nodo non riparte da 0, grazie al MessagePlaceholder con i messaggi attaccati
revisor = actor_prompt_template.partial(
    first_instruction=revise_instructions
) | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")


if __name__ == "__main__":
    human_message = HumanMessage(
        content="Write about AI-Powered SOC / autonomous soc problem domain,"
        " list startups that do that and raised capital."
    )
    chain = (
        first_responder_prompt_template
        | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")
        | parser_pydantic
    )

    res = chain.invoke(input={"messages": [human_message]})
    print(res)
