from langchain_ollama import OllamaLLM
from src.config import Config

def create_llm():
    return OllamaLLM(
        model=Config.LLM_MODEL,
        temperature=0.5,
        num_ctx=4096, #context window size
        num_gpu=1
    )