import os
import logging
from core.base_rag import BaseRAG

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Load the RAG system
    try:
        rag = BaseRAG.load("bio_rag.pkl")
        logger.info("Successfully loaded RAG system")
    except Exception as e:
        logger.error(f"Failed to load RAG system: {e}")
        return

    print("\nWelcome to BioRAG Bioinformatics Query System!")
    print("This interface is specialized for bioinformatics-related queries.")
    print("Type 'exit' or 'quit' to end the session\n")

    while True:
        try:
            # Get user input
            question = input("\nEnter your bioinformatics question: ").strip()
            
            if question.lower() in ['exit', 'quit']:
                print("\nThank you for using BioRAG Bioinformatics Query System!")
                break
                
            if not question:
                continue

            # Query the RAG system with bioinformatics context
            context = "You are a bioinformatics expert. Please provide a detailed answer focusing on the biological and computational aspects of the question."
            answer = rag.query(question, context=context)
            
            # Print the answer
            print("\nAnswer:")
            print(answer)
            
        except KeyboardInterrupt:
            print("\n\nThank you for using BioRAG Bioinformatics Query System!")
            break
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            print("\nAn error occurred while processing your query. Please try again.")

if __name__ == "__main__":
    main() 