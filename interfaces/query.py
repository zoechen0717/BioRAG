import os
import logging
from typing import Optional, List, Dict, Any
from core.base_rag import BaseRAG
from bioinfo.code_analyzer import CodeAnalyzer

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

class QueryInterface:
    def __init__(self, rag: BaseRAG):
        self.rag = rag
        self.code_analyzer = CodeAnalyzer(rag)
        self.logger = logging.getLogger(__name__)
        self.query_history: List[Dict[str, Any]] = []

    def run(self):
        """Run the query interface."""
        print("\nWelcome to BioRAG Query System!")
        print("Type 'exit' or 'quit' to end the session")
        print("Type 'help' for available commands\n")
        
        while True:
            try:
                # Get user input
                user_input = input("\nEnter your question or command: ").strip()
                
                # Check for exit commands
                if user_input.lower() in ['exit', 'quit']:
                    print("\nThank you for using BioRAG Query System!")
                    break
                    
                # Check for help command
                if user_input.lower() == 'help':
                    self._show_help()
                    continue
                
                # Check for history command
                if user_input.lower() == 'history':
                    self._show_history()
                    continue
                
                # Check for code query
                if user_input.lower().startswith('code:'):
                    self._handle_code_query(user_input[5:].strip())
                    continue
                
                # Handle general query
                self._handle_general_query(user_input)
                    
            except KeyboardInterrupt:
                print("\n\nThank you for using BioRAG Query System!")
                break
            except Exception as e:
                self.logger.error(f"Error in query session: {e}")
                print(f"\nAn error occurred: {str(e)}")

    def _show_help(self):
        """Show available commands."""
        print("\nAvailable commands:")
        print("- Type 'code: <question>' to ask questions about code")
        print("- Type 'history' to view query history")
        print("- Type 'help' to see this message")
        print("- Type 'exit' or 'quit' to end the session")
        print("\nYou can also ask general questions about papers and research.")

    def _show_history(self):
        """Show query history."""
        if not self.query_history:
            print("\nNo queries in history.")
            return
            
        print("\nQuery History:")
        for i, entry in enumerate(self.query_history, 1):
            print(f"\n{i}. Question: {entry['question']}")
            print(f"   Type: {entry['type']}")
            print(f"   Timestamp: {entry['timestamp']}")

    def _handle_code_query(self, question: str):
        """Handle code-related queries."""
        if not question:
            print("Please enter a question.")
            return
            
        print("\nAnalyzing code...")
        try:
            answer = self.code_analyzer.query_code(question)
            print("\nAnswer:")
            print(answer)
            
            # Add to history
            self._add_to_history(question, 'code')
            
        except Exception as e:
            self.logger.error(f"Error processing code query: {e}")
            print(f"\nAn error occurred: {str(e)}")

    def _handle_general_query(self, question: str):
        """Handle general queries."""
        if not question:
            print("Please enter a question.")
            return
            
        print("\nSearching knowledge base...")
        try:
            answer = self.rag.query(question)
            print("\nAnswer:")
            print(answer)
            
            # Add to history
            self._add_to_history(question, 'general')
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            print(f"\nAn error occurred: {str(e)}")

    def _add_to_history(self, question: str, query_type: str):
        """Add a query to history."""
        from datetime import datetime
        self.query_history.append({
            'question': question,
            'type': query_type,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Load the RAG system
        rag = BaseRAG.load("bio_rag.pkl")
        logger.info("Successfully loaded RAG system")
        
        # Create and run the query interface
        interface = QueryInterface(rag)
        interface.run()
        
    except Exception as e:
        logger.error(f"Failed to start query interface: {e}")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5001))
    main() 