import os
import sys
import logging
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

# Add the parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Ensure cache directory exists
cache_dir = os.path.join(parent_dir, "data", "cache")
os.makedirs(cache_dir, exist_ok=True)

from core.base_rag import BaseRAG
from bioinfo.code_analyzer import CodeAnalyzer
from bioinfo.research_assistant import ResearchAssistant

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize RAG system and assistants
try:
    rag_file = os.path.join(parent_dir, "bio_rag.pkl")
    rag = BaseRAG.load(rag_file)
    code_analyzer = CodeAnalyzer(rag)
    research_assistant = ResearchAssistant(rag)
    logger.info("Successfully loaded RAG system and assistants")
except Exception as e:
    logger.error(f"Error initializing RAG system: {str(e)}")
    raise

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    """Handle user queries."""
    try:
        data = request.get_json()
        question = data.get('question')
        query_type = data.get('type', 'general')
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
            
        if query_type == 'code':
            answer = code_analyzer.query(question)
        elif query_type == 'research':
            answer = research_assistant.query(question)
        else:
            answer = rag.query(question)
        
        # Ensure answer is a string and properly formatted
        if answer is None:
            answer = "No answer available."
        elif isinstance(answer, dict):
            # If it's a dictionary, convert it to a formatted string
            answer = "\n".join(f"{k}: {v}" for k, v in answer.items())
        elif isinstance(answer, list):
            # If it's a list, join the elements
            answer = "\n".join(str(item) for item in answer)
        else:
            # Convert any other type to string
            answer = str(answer)
            
        # Clean up the answer string
        answer = answer.strip()
        if not answer:
            answer = "No answer available."
            
        return jsonify({'answer': answer})
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/analyze_papers', methods=['POST'])
def analyze_papers():
    """Analyze papers related to a topic."""
    try:
        data = request.get_json()
        topic = data.get('topic')
        
        if not topic:
            return jsonify({'error': 'No topic provided'}), 400
            
        analysis = research_assistant.analyze_papers(topic)
        return jsonify({'analysis': analysis})
    except Exception as e:
        logger.error(f"Error analyzing papers: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_implementation', methods=['POST'])
def get_implementation():
    """Get implementation suggestions for a topic."""
    try:
        data = request.get_json()
        topic = data.get('topic')
        
        if not topic:
            return jsonify({'error': 'No topic provided'}), 400
            
        suggestions = code_analyzer.get_implementation_suggestions(topic)
        return jsonify({'suggestions': suggestions})
    except Exception as e:
        logger.error(f"Error getting implementation suggestions: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/add_paper', methods=['POST'])
def add_paper():
    """Add a new paper to the RAG system."""
    try:
        data = request.get_json()
        title = data.get('title')
        content = data.get('content')
        metadata = {
            'title': title,
            'authors': data.get('authors', []),
            'year': data.get('year', ''),
            'url': data.get('url', ''),
            'source': 'user_upload'
        }
        
        if not title or not content:
            return jsonify({'error': 'Title and content are required'}), 400
            
        # Add the paper to the RAG system
        rag.add_document(content, metadata)
        
        # Save the updated RAG system
        rag.save(rag_file)
        
        return jsonify({
            'message': f'Successfully added paper: {title}',
            'status': 'success'
        })
    except Exception as e:
        logger.error(f"Error adding paper: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=3000) 