# BioRAG: Biomedical Research Assistant

BioRAG is a powerful tool for analyzing biomedical research papers and code, built using RAG (Retrieval-Augmented Generation) technology.

## Features

- Paper analysis and querying
- Code analysis and documentation
- Research brainstorming
- Web interface for easy interaction
- Command-line interface for advanced usage

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/BioRAG.git
cd BioRAG
```

2. Create and activate a conda environment:
```bash
conda create -n BioRAG python=3.10
conda activate BioRAG
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Web Interface

The web interface provides an easy-to-use way to interact with BioRAG. To start the web interface:

1. Make sure you're in the BioRAG directory and your conda environment is activated:
```bash
conda activate BioRAG
```

2. Start the web server:
```bash
cd web
python run.py
```

3. Open your web browser and navigate to `http://localhost:5000`

The web interface provides the following features:
- General queries about papers and research
- Code analysis and documentation
- Research brainstorming
- Paper analysis by topic
- Implementation suggestions

### Command Line Interface

For advanced usage, you can use the command-line interface:

1. Process papers:
```bash
python process_papers.py
```

2. Query the system:
```bash
python query_rag.py
```

## Project Structure

```
BioRAG/
├── bioinfo/              # Bioinformatics-specific modules
│   ├── code_analyzer.py  # Code analysis functionality
│   └── research_assistant.py  # Research assistance
├── core/                 # Core RAG functionality
│   ├── base_rag.py      # Base RAG implementation
│   ├── document_processor.py  # Document processing
│   └── initialize.py    # System initialization
├── interfaces/           # User interfaces
│   ├── query.py         # Query interface
│   ├── interactive.py   # Interactive interface
│   └── bioinfo_query.py # Bioinformatics query interface
├── utils/               # Utility functions
│   └── rag_manager.py   # RAG system management
├── web/                 # Web interface
│   ├── app.py          # Flask application
│   ├── run.py          # Web server runner
│   └── templates/      # HTML templates
├── tests/              # Test files
├── data/               # Data directory
├── config.yaml         # Configuration file
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Configuration

The system can be configured through `config.yaml`. Key settings include:
- OpenAI API configuration
- Model parameters
- Processing settings
- Web interface settings

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 