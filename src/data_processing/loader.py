from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    JSONLoader,
    WebBaseLoader
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from langchain.schema import Document
from src.config import Config
import re
import os
import io
import time
from langchain_community.vectorstores.utils import filter_complex_metadata
import fitz  # PyMuPDF
import base64
from PIL import Image
import pandas as pd
from pdfminer.high_level import extract_text

class DataLoader:
    def __init__(self):
        self.allowed_types = Config.ALLOWED_FILE_TYPES
        self.allowed_domains = Config.ALLOWED_DOMAINS

    def _clean_content(self, content):
        content = re.sub(r'\s+', ' ', content)
        content = re.sub(r'<[^>]+>', '', content)
        return content.strip()[:5000]
    
    def _extract_text_from_pdf(self, file_path):
        """
        Extract text from PDF using pdfminer
        """
        try:
            return extract_text(file_path)
        except Exception as e:
            raise ValueError(f"Error extracting text from PDF: {e}")

    def load_file(self, file_path):
        try:
            ext = os.path.splitext(file_path)[1].lower()
            
            if ext == ".pdf":
                try:
                    print(f"Starting PDF text extraction: {file_path}")
                    start_time = time.time()
                    content = self._extract_text_from_pdf(file_path)
                    end_time = time.time()
                    print(f"PDF text extraction completed in {end_time - start_time:.2f} seconds")

                    # Create Document object with proper metadata
                    return [Document(
                        page_content=self._clean_content(content),
                        metadata={
                            "source": str(file_path),
                            "file_type": "pdf"
                        }
                    )]                 
                except Exception as e:
                    raise ValueError(f"Error processing PDF file: {e}")
            else:
                raise ValueError(f"Unsupported file type: {ext}")

        except Exception as e:
            raise ValueError(f"Error loading file {file_path}: {e}")

    def load_web(self, url):
        if not any(d in url for d in self.allowed_domains):
            raise ValueError("Domain không được cho phép")
            
        docs = WebBaseLoader(url).load()
        return [Document(
            page_content=self._clean_content(doc.page_content),
            metadata={"source": url, **doc.metadata}
        ) for doc in docs]
    