from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import Config

def create_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? "]
    )