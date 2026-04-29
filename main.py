from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

load_dotenv()


def main():
    print("Hello from langchain-course!")
    information = """
    Elon Reeve Musk[1] (AFI:[ˈiːlɒn ˈɹiːv ˈmʌsk]) (Pretoria, 28 giugno 1971) è un imprenditore e politico sudafricano con cittadinanza canadese naturalizzato statunitense.
    Ricopre i ruoli di fondatore, amministratore delegato e direttore tecnico della compagnia aerospaziale SpaceX, fondatore di The Boring Company e della società di intelligenza artificiale xAI, cofondatore di Neuralink e OpenAI, amministratore delegato e product architect della multinazionale automobilistica Tesla, proprietario e presidente di X (precedentemente Twitter). Ha inoltre proposto un sistema di trasporto superveloce conosciuto come Hyperloop One, posta in liquidazione il 21 dicembre 2023. Tramite SpaceX gestisce Starlink, una costellazione satellitare che avrebbe l'obiettivo di fornire Internet ad alta velocità e bassa latenza a tutto il pianeta.
    Secondo Forbes, al 7 febbraio 2026, con un patrimonio stimato di 849,3 miliardi di dollari, risulta essere la persona più ricca del mondo.
    Dal 20 gennaio al 29 maggio 2025 è stato a capo del Dipartimento dell'Efficienza Governativa statunitense.
    """
    summary_template = """
    Given the information {information}, about a person I want you to create:
    1. A short summary
    2. Two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        # The input_variables parameter tells the PromptTemplate which placeholder in the template string
        # need to be replaced with actual values when the template is used
        input_variables=["information"],
        template=summary_template,
    )

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    # llm = ChatOllama(temperature=0, model="gemma3:4b")

    # The | operator is part of the LangChain Expression Language(LCEL) syntax.
    # It creates a chain by connecting components in a pipeline where the output of the left component becomes 
    # the input of the right component.
    # Furthermore, it makes the code more mantainable, allows for easy composition of complex pipelines,
    # provides built-in error handling and enables advanced features like streaming and parallel execution.
    chain = summary_prompt_template | llm

    # invoke() passes the input dict to the PromptTemplate, which formats the template by substituting
    # {information} with the provided value. Then, the formatted prompt is passed to the LLM, wich generates
    # and returns the response
    response = chain.invoke(input={"information": information})
    print(response.content)


if __name__ == "__main__":
    main()
