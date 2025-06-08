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

class DataLoader:
    def __init__(self):
        self.allowed_types = Config.ALLOWED_FILE_TYPES
        self.allowed_domains = Config.ALLOWED_DOMAINS
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = False
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def _clean_content(self, content):
        content = re.sub(r'\s+', ' ', content)
        content = re.sub(r'<[^>]+>', '', content)
        return content.strip()[:5000]

    def load_file(self, file_path):
        try:
            ext = os.path.splitext(file_path)[1].lower()
            
            if ext == ".pdf":
                try:
                    print(f"Starting PDF conversion: {file_path}")
                    start_time = time.time()
                    result = self.converter.convert(file_path)
                    end_time = time.time()
                    print(f"PDF conversion completed in {end_time - start_time:.2f} seconds")
                    content = result.document.export_to_markdown()

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
    
    