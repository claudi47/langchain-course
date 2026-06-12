from typing import TypedDict, Annotated

from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from chains import generate_chain, reflect_chain

load_dotenv()

# Qui vi è la formula di LangGraph per aggiornare lo stato:
# Annotated permette di aggiungere metadati alla lista di messaggi, tramite add_messages
# add_messages è un REDUCER, ovvero una funzione che prende lo stato precedente e lo aggiorna
# in questo caso aggiungendo un nuovo messaggio alla lista di quelli esistenti
class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


REFLECT = "reflect"
GENERATE = "generate"

# Quando un nodo ritorna la lista dei messaggi, LangGraph internamente la aggiorna
# chiamando proprio add_messages, che ne appende il risultato a quelli già presenti
def generation_node(state: MessageGraph):
    return {"messages": [generate_chain.invoke({"messages": state["messages"]})]}


def reflection_node(state: MessageGraph):
    res = reflect_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=res.content)]}


builder = StateGraph(state_schema=MessageGraph)
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)


def should_continue(state: MessageGraph):
    if len(state["messages"]) > 6:
        return END
    return REFLECT

# Qui stiamo collegando i nodi, specificando che dopo GENERATE vogliamo valutare se continuare o terminare
# dopo REFLECT invece vogliamo sempre tornare a GENERATE, creando così un ciclo di generazione e riflessione
# fino a quando non decidiamo di terminare
builder.add_conditional_edges(GENERATE, should_continue, path_map={END: END, REFLECT: REFLECT})
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()
print(graph.get_graph().draw_mermaid())
graph.get_graph().print_ascii()

if __name__ == "__main__":
    print("Hello LangGraph")
    inputs: MessageGraph = {
        "messages": [
            HumanMessage(
                content="""Make this tweet better:"
                                    @LangChainAI
            — newly Tool Calling feature is seriously underrated.

            After a long wait, it's  here- making the implementation of agents across different models with function calling - super easy.

            Made a video covering their newest blog post

                                  """
            )
        ]
    }
    response = graph.invoke(inputs)
    print(response)
