from langchain_chroma import Chroma
from src.config import Config
from transformers import AutoTokenizer, AutoModel
import torch


device = "cuda:0" if torch.cuda.is_available() else "cpu"

class PhoBertEmbeddingFunction:
    
    def __init__(self, model_name):
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1)

    def embed_documents(self, texts):
        return self._embed(texts)

    def embed_query(self, text):
        return self._embed([text])[0]

    def _embed(self, texts):    
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            model_output = self.model(**inputs)
        embeddings = self.mean_pooling(model_output, inputs['attention_mask'])
        return embeddings.numpy().tolist()

def initialize_vector_store():
    embeddings = PhoBertEmbeddingFunction(Config.EMBEDDING_MODEL)
    return Chroma(
        persist_directory="vector_store",
        embedding_function=embeddings,
        collection_name="travel_ai",
        collection_metadata={"hnsw:space": "cosine"}
    )
