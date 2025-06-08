from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from src.config import Config

def initialize_vector_store():
    embeddings = OllamaEmbeddings(model=Config.EMBEDDING_MODEL)
    return Chroma(
        persist_directory="vector_store",
        embedding_function=embeddings,
        collection_name="travel_ai",  
        collection_metadata={"hnsw:space": "cosine"}
    )

