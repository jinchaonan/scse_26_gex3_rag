### COMPLETE THE CODE ###

from policy_loader import load_policy_documents

## TO LOAD THE DOCUMENT, USE THE FOLLOWING ONLY:
documents = load_policy_documents()

import ollama
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings

EMBEDDING_MODEL_NAME = "nomic-embed-text"
OLLAMA_HOST = "http://127.0.0.1:11434"


class OllamaDirectEmbeddings(Embeddings):
    """Minimal embedding wrapper that talks to Ollama directly."""

    def __init__(self, model: str, host: str):
        self.model = model
        self.client = ollama.Client(host=host)

    def embed_documents(self, texts):
        return [
            self.client.embeddings(model=self.model, prompt=t)["embedding"]
            for t in texts
        ]

    def embed_query(self, text):
        return self.client.embeddings(model=self.model, prompt=text)["embedding"]


def build_vectorstore(persist_directory: str = "./chroma_policy_db"):
    embeddings = OllamaDirectEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        host=OLLAMA_HOST,
    )
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
    return vectorstore


def build_retriever(vectorstore, k: int = 4):
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )