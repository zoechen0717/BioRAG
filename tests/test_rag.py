import os
import unittest
from pathlib import Path
import sys
import yaml
import logging

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.base_rag import BaseRAG
from core.document_processor import DocumentProcessor
from core.initialize import initialize_rag

class TestRAGSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.test_dir = Path(__file__).parent / "test_data"
        cls.config_path = Path(__file__).parent / "test_config.yaml"
        cls.output_file = cls.test_dir / "test_rag.pkl"
        
        # Create test directories
        for dir_name in ["papers", "code", "embeddings", "cache", "logs"]:
            (cls.test_dir / dir_name).mkdir(parents=True, exist_ok=True)
        
        # Load and update config with API key
        with open(cls.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Replace API key placeholder with environment variable
        if '${OPENAI_API_KEY}' in config['openai']['api_key']:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            config['openai']['api_key'] = api_key
        
        # Save updated config
        with open(cls.config_path, 'w') as f:
            yaml.dump(config, f)
    
    def test_initialization(self):
        """Test RAG system initialization."""
        try:
            rag = initialize_rag(
                config_path=str(self.config_path),
                output_file=str(self.output_file)
            )
            self.assertIsInstance(rag, BaseRAG)
            self.assertTrue(self.output_file.exists())
        except Exception as e:
            self.fail(f"Initialization failed: {str(e)}")
    
    def test_document_processor(self):
        """Test document processor functionality."""
        processor = DocumentProcessor()
        self.assertIsNotNone(processor)
    
    def test_add_document(self):
        """Test adding documents to RAG system."""
        rag = BaseRAG(str(self.config_path))
        test_text = "This is a test document for RAG system."
        try:
            rag.add_document(test_text, {"source": "test"})
            self.assertEqual(len(rag.documents), 1)
        except Exception as e:
            self.fail(f"Failed to add document: {str(e)}")
    
    def test_query(self):
        """Test querying the RAG system."""
        rag = BaseRAG(str(self.config_path))
        test_text = "This is a test document about machine learning."
        try:
            rag.add_document(test_text, {"source": "test"})
            response = rag.query("What is the document about?")
            self.assertIsInstance(response, str)
            self.assertTrue(len(response) > 0)
        except Exception as e:
            self.fail(f"Query failed: {str(e)}")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        # Remove test files
        if cls.output_file.exists():
            cls.output_file.unlink()
        
        # Remove test directories
        for dir_name in ["papers", "code", "embeddings", "cache", "logs"]:
            dir_path = cls.test_dir / dir_name
            if dir_path.exists():
                for file in dir_path.glob("*"):
                    file.unlink()
                dir_path.rmdir()

if __name__ == "__main__":
    unittest.main() 