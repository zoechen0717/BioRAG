#!/usr/bin/env python3
import os
import sys
import logging
from app import app

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    """Main function to run the Flask application."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Get port from environment variable or use default
        port = int(os.environ.get('PORT', 8000))
        
        # Run the Flask application
        logger.info(f"Starting BioRAG web interface on port {port}")
        app.run(host='0.0.0.0', port=port, debug=True)
    except Exception as e:
        logger.error(f"Error starting web interface: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 