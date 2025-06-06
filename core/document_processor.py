import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import PyPDF2
import logging

class DocumentProcessor:
    """Process documents for the RAG system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the document processor.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{id(self)}")
    
    def read_pdf(self, file_path: str) -> str:
        """Read and extract text from a PDF file."""
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            self.logger.error(f"Failed to read PDF {file_path}: {str(e)}")
            raise
    
    def process_pdf(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """Process a PDF file and return its content and metadata."""
        try:
            text = self.read_pdf(file_path)
            metadata = {
                "filename": os.path.basename(file_path),
                "filepath": file_path,
                "type": "pdf",
                "pages": len(PyPDF2.PdfReader(file_path).pages)
            }
            return text, metadata
        except Exception as e:
            self.logger.error(f"Failed to process PDF {file_path}: {str(e)}")
            raise
    
    def find_pdf_files(self, directory: str) -> list:
        """Recursively find all PDF files in a directory."""
        try:
            pdf_files = []
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.lower().endswith('.pdf'):
                        pdf_files.append(os.path.join(root, file))
            return pdf_files
        except Exception as e:
            self.logger.error(f"Failed to find PDF files in {directory}: {str(e)}")
            raise 