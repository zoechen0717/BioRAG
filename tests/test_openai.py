import os
import pytest
from openai import OpenAI
from core.base_rag import BaseRAG
from core.document_processor import DocumentProcessor

def test_openai_connection():
    """Test OpenAI API connection."""
    client = OpenAI()
    try:
        # Test a simple completion
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}]
        )
        assert response.choices[0].message.content is not None
    except Exception as e:
        pytest.fail(f"OpenAI API connection failed: {str(e)}")

def test_pdf_processing():
    """Test PDF file processing."""
    # Test directory setup
    test_dir = "data/papers"
    assert os.path.exists(test_dir), f"Test directory {test_dir} not found"
    
    # Get list of PDF files
    pdf_files = [f for f in os.listdir(test_dir) if f.endswith('.pdf')]
    assert len(pdf_files) > 0, "No PDF files found in test directory"
    
    # Test document processor
    processor = DocumentProcessor()
    for pdf_file in pdf_files[:1]:  # Test with first PDF file
        try:
            # Process the PDF file
            chunks = processor.process_file(os.path.join(test_dir, pdf_file))
            assert len(chunks) > 0, "No chunks extracted from PDF"
            
            # Test chunk content
            for chunk in chunks:
                assert "text" in chunk, "Chunk missing text field"
                assert "metadata" in chunk, "Chunk missing metadata field"
                assert len(chunk["text"]) > 0, "Empty chunk text"
                
        except Exception as e:
            pytest.fail(f"PDF processing failed: {str(e)}")

def test_rag_system():
    """Test RAG system with OpenAI."""
    try:
        # Initialize RAG system
        rag = BaseRAG()
        
        # Test document addition
        test_text = "This is a test document for RAG system."
        rag.add_document(test_text, {"source": "test"})
        assert len(rag.documents) == 1, "Document not added correctly"
        
        # Test query
        response = rag.query("What is this document about?")
        assert response is not None, "Query failed"
        
    except Exception as e:
        pytest.fail(f"RAG system test failed: {str(e)}")

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__]) 