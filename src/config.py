from pathlib import Path

class Config:
    # Đường dẫn
    DATA_DIR = Path("data/initial")
    UPLOAD_DIR = Path("data/uploaded")
    IMAGE_DIR = Path("data/images") 
    
    # Model settings
    EMBEDDING_MODEL = "nomic-embed-text"
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
    
config = Config()