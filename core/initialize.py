import os
from pathlib import Path
import yaml
import logging
from typing import Dict, Any, Optional, List, Tuple
from .base_rag import BaseRAG
from .document_processor import DocumentProcessor

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load configuration: {str(e)}")

def setup_directories(config: Dict[str, Any]):
    """Create necessary directories."""
    paths = config['paths']
    for path in paths.values():
        Path(path).mkdir(parents=True, exist_ok=True)

def process_papers(
    papers_dir: str,
    processor: DocumentProcessor
) -> List[Tuple[str, Dict[str, Any]]]:
    """Process all papers in the directory."""
    processed_papers = []
    pdf_files = processor.find_pdf_files(papers_dir)
    
    for pdf_file in pdf_files:
        try:
            content, metadata = processor.process_pdf(pdf_file)
            processed_papers.append((content, metadata))
            logging.info(f"Successfully processed {pdf_file}")
        except Exception as e:
            logging.error(f"Failed to process {pdf_file}: {str(e)}")
            continue
    
    return processed_papers

def initialize_rag(
    config_path: str = "config.yaml",
    papers_dir: Optional[str] = None,
    output_file: str = "bio_rag.pkl"
) -> BaseRAG:
    """Initialize the RAG system with papers.
    
    Args:
        config_path: Path to configuration file
        papers_dir: Directory containing PDF papers
        output_file: Path to save the RAG system
    
    Returns:
        Initialized RAG system
    """
    try:
        # Load configuration
        config = load_config(config_path)
        
        # Setup directories
        setup_directories(config)
        
        # Initialize RAG system
        rag = BaseRAG(config_path)
        
        # Process papers if directory is provided
        if papers_dir:
            processor = DocumentProcessor()
            processed_papers = process_papers(papers_dir, processor)
            
            # Add processed papers to RAG system
            for content, metadata in processed_papers:
                rag.add_document(content, metadata)
            
            logging.info(f"Added {len(processed_papers)} papers to RAG system")
        
        # Save the RAG system
        rag.save(output_file)
        logging.info(f"Saved RAG system to {output_file}")
        
        return rag
    except Exception as e:
        logging.error(f"Failed to initialize RAG system: {str(e)}")
        raise

def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize BioRAG system")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--papers", help="Path to papers directory")
    parser.add_argument("--output", default="bio_rag.pkl", help="Output file path")
    args = parser.parse_args()
    
    try:
        rag = initialize_rag(args.config, args.papers, args.output)
        print("RAG system initialized successfully!")
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main() 