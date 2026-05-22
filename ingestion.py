import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

BLOG_PATH = Path(__file__).with_name("mediumblog1.txt")
EMBEDDING_MODEL = os.environ.get("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

if __name__ == "__main__":
    # Data Loading -> Data Splitting -> Data Embedding -> Data Storing on Pinecone DB
    print("Ingesting...")
    loader = TextLoader(str(BLOG_PATH))
    document = loader.load()

    print("splitting...")
    # Mai mettere un chunk size troppo piccolo
    # chunk_overlap > 0 se il testo da ingerire contiene chunk non legati tra loro
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    # Modello di embedding usato per la conversione chunk di testo -> vettore
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    print("Storing...")
    PineconeVectorStore.from_documents(
        texts, embeddings, index_name=os.environ["INDEX_NAME"]
    )
    print("finish")
