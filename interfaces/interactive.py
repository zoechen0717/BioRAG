import os
import logging
from typing import Optional
from core.base_rag import BaseRAG
from bioinfo.research_assistant import ResearchAssistant

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

class InteractiveAssistant:
    def __init__(self, rag: BaseRAG):
        self.rag = rag
        self.research_assistant = ResearchAssistant(rag)
        self.logger = logging.getLogger(__name__)

    def run(self):
        """Run the interactive assistant."""
        print("\nWelcome to BioRAG Interactive Assistant!")
        print("Type 'exit' or 'quit' to end the session.")
        print("Type 'help' for available commands.")
        
        while True:
            try:
                # Get user input
                user_input = input("\nWhat would you like to do? ").strip()
                
                # Check for exit commands
                if user_input.lower() in ['exit', 'quit']:
                    print("Goodbye!")
                    break
                    
                # Check for help command
                if user_input.lower() == 'help':
                    self._show_help()
                    continue
                
                # Handle different commands
                if user_input.lower() == 'brainstorm':
                    self._handle_brainstorm()
                elif user_input.lower() == 'papers':
                    self._handle_papers()
                elif user_input.lower() == 'implement':
                    self._handle_implementation()
                elif user_input.lower() == 'query':
                    self._handle_query()
                else:
                    # If no specific command, treat as a general query
                    self._handle_general_query(user_input)
                    
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                self.logger.error(f"Error in interactive session: {e}")
                print(f"\nAn error occurred: {str(e)}")

    def _show_help(self):
        """Show available commands."""
        print("\nAvailable commands:")
        print("- Type 'brainstorm' to start a research brainstorming session")
        print("- Type 'papers' to analyze connections between papers and code")
        print("- Type 'implement' to get implementation suggestions")
        print("- Type 'query' to ask general questions")
        print("- Type 'exit' or 'quit' to end the session")
        print("- Type 'help' to see this message")

    def _handle_brainstorm(self):
        """Handle research brainstorming session."""
        topic = input("\nEnter your research topic: ").strip()
        if not topic:
            print("Please enter a topic.")
            return
        print("\nGenerating research ideas based on papers and code...")
        ideas = self.research_assistant.research_brainstorm(topic)
        print("\nResearch Suggestions:")
        print(ideas)

    def _handle_papers(self):
        """Handle paper analysis session."""
        topic = input("\nEnter topic to analyze paper-code connections: ").strip()
        if not topic:
            print("Please enter a topic.")
            return
        print("\nAnalyzing connections between papers and code...")
        analysis = self.research_assistant.analyze_paper_connections(topic)
        print("\nPaper-Code Analysis:")
        print(analysis)

    def _handle_implementation(self):
        """Handle implementation suggestions."""
        topic = input("\nDescribe the implementation task: ").strip()
        if not topic:
            print("Please describe the task.")
            return
        print("\nGenerating implementation suggestions...")
        suggestions = self.research_assistant.get_implementation_suggestions(topic)
        print("\nImplementation Suggestions:")
        print(suggestions)

    def _handle_query(self):
        """Handle general query session."""
        question = input("\nEnter your question: ").strip()
        if not question:
            print("Please enter a question.")
            return
        print("\nSearching knowledge base...")
        answer = self.rag.query(question)
        print("\nAnswer:", answer)

    def _handle_general_query(self, query: str):
        """Handle general query without specific command."""
        print("\nSearching knowledge base...")
        answer = self.rag.query(query)
        print("\nAnswer:", answer)

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Load the RAG system
        rag = BaseRAG.load("bio_rag.pkl")
        logger.info("Successfully loaded RAG system")
        
        # Create and run the interactive assistant
        assistant = InteractiveAssistant(rag)
        assistant.run()
        
    except Exception as e:
        logger.error(f"Failed to start interactive assistant: {e}")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 