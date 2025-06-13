from pathlib import Path
from sentence_transformers import SentenceTransformer
class Config:
    # Đường dẫn
    DATA_DIR = Path("data/initial")
    UPLOAD_DIR = Path("data/uploaded")
    IMAGE_DIR = Path("data/images") 
    
    # Model settings
    EMBEDDING_MODEL = 'VoVanPhuc/sup-SimCSE-VietNamese-phobert-base'
    LLM_MODEL = "llama3.2:latest"
    
    # RAG parameters
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    RETRIEVE_TOP_K = 5
    
    # Security
    ALLOWED_FILE_TYPES = [".pdf"]
    ALLOWED_DOMAINS = ["dulichvietnam.com", "tripadvisor.com","vietnamtravelgroup.com.vn"]

    # Image
    MAX_IMAGES_TO_SHOW = 3  

    HYDE_ENABLED = True  # Bật/tắt HyDE
    HYBRID_ALPHA = 0.7   # Trọng số cho vector retriever (0.7 vector + 0.3 BM25)
    RERANKER_MODEL = "castorini/monot5-base-msmarco"
    MAX_RERANK_DOCS = 20  # Số lượng docs tối đa để rerank  
    
config = Config()