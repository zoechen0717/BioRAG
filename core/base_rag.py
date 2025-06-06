import os
import logging
from typing import List, Dict, Any, Optional
import yaml
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken
import pickle
from pathlib import Path
from utils.cache_manager import CacheManager
import time

class BaseRAG:
    """Base class for RAG systems with improved functionality."""
    
    def __init__(self, config_path: str = "config.yaml", top_k: int = 3):
        """Initialize the RAG system with configuration."""
        self.config = self._load_config(config_path)
        self._setup_storage()
        self._setup_openai()
        self._setup_encoding()
        
        self.documents = []
        self.embeddings = []
        self.metadata = []
        
        # Setup logging after other initializations
        self._setup_logging()
        
        # Initialize cache manager with the correct path
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cache_dir = os.path.join(base_dir, "data", "cache")
        self.cache_manager = CacheManager(cache_dir)
        
        # Set top_k from config or use default
        self.top_k = self.config.get('rag', {}).get('top_k', top_k)
        self.logger.info(f"Initialized RAG system with top_k={self.top_k}")
    
    def _load_config(self, config_path: str) -> dict:
        """
        Load configuration from a YAML file. If not found, search parent directories.
        """
        import yaml
        import os
        orig_path = config_path
        if not os.path.isabs(config_path):
            # Start from current working directory
            search_dir = os.getcwd()
            while True:
                candidate = os.path.join(search_dir, config_path)
                if os.path.isfile(candidate):
                    config_path = candidate
                    break
                parent = os.path.dirname(search_dir)
                if parent == search_dir:
                    # Reached root, not found
                    break
                search_dir = parent
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration: {str(e)} (searched for '{orig_path}')")
    
    def _setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _setup_openai(self):
        """Setup OpenAI client."""
        try:
            self.client = OpenAI(api_key=self.config['openai']['api_key'])
            self.model = self.config['openai']['model']
            self.embedding_model = self.config['openai']['embedding_model']
        except Exception as e:
            raise RuntimeError(f"Failed to setup OpenAI client: {str(e)}")
    
    def _setup_encoding(self):
        """Setup token encoding."""
        try:
            self.encoding = tiktoken.encoding_for_model(self.model)
        except Exception as e:
            raise RuntimeError(f"Failed to setup encoding: {str(e)}")
    
    def _setup_storage(self):
        """Setup storage directories."""
        paths = self.config['paths']
        for path in paths.values():
            Path(path).mkdir(parents=True, exist_ok=True)
    
    def get_embedding(self, text: str, use_cache: bool = True) -> List[float]:
        """Get embedding for text with optional caching."""
        if use_cache and self.config['rag']['cache_embeddings']:
            cache_id = str(hash(text))
            cached_result = self.cache_manager.get(cache_id)
            if cached_result is not None:
                self.logger.info(f"Using cached embedding for text: {text[:50]}...")
                return cached_result
        
        try:
            # Add retry logic for API calls
            max_retries = 3
            retry_delay = 1  # seconds
            
            for attempt in range(max_retries):
                try:
                    response = self.client.embeddings.create(
                        model=self.embedding_model,
                        input=text
                    )
                    embedding = response.data[0].embedding
                    
                    if use_cache and self.config['rag']['cache_embeddings']:
                        self.cache_manager.set(cache_id, embedding)
                    
                    return embedding
                except Exception as e:
                    if attempt < max_retries - 1:
                        self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
                        time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                    else:
                        raise RuntimeError(f"Failed to get embedding after {max_retries} attempts: {str(e)}")
        except Exception as e:
            self.logger.error(f"Failed to get embedding: {str(e)}")
            raise
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks of appropriate size."""
        chunk_size = self.config['rag']['chunk_size']
        tokens = self.encoding.encode(text)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for token in tokens:
            current_chunk.append(token)
            current_size += 1
            
            if current_size >= chunk_size:
                chunks.append(self.encoding.decode(current_chunk))
                current_chunk = []
                current_size = 0
        
        if current_chunk:
            chunks.append(self.encoding.decode(current_chunk))
        
        return chunks
    
    def add_document(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a document to the RAG system."""
        try:
            chunks = self.chunk_text(text)
            for chunk in chunks:
                embedding = self.get_embedding(chunk)
                self.documents.append(chunk)
                self.embeddings.append(embedding)
                self.metadata.append(metadata or {})
            
            self.logger.info(f"Added document with {len(chunks)} chunks")
        except Exception as e:
            self.logger.error(f"Failed to add document: {str(e)}")
            raise
    
    def query(self, question: str) -> str:
        """Query the RAG system."""
        try:
            # Check cache first
            cache_id = str(hash(question))
            cached_result = self.cache_manager.get(cache_id)
            if cached_result is not None:
                self.logger.info(f"Using cached result for query: {question}")
                return cached_result

            # Process query
            result = self._process_query(question)
            
            # Cache result
            self.cache_manager.set(cache_id, result)
            
            return result
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            raise
    
    def _create_prompt(self, question: str, docs: List[str], metadata: List[Dict[str, Any]]) -> str:
        """Create a prompt for the query."""
        context = "\n\n".join(docs)
        return f"""Based on the following context, please answer the question.

Context:
{context}

Question: {question}

Answer:"""
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the model."""
        return "You are a helpful assistant that answers questions based on the provided context."
    
    def __getstate__(self):
        """Prepare object for serialization."""
        state = self.__dict__.copy()
        # Remove non-serializable components
        if 'logger' in state:
            del state['logger']
        if 'client' in state:
            del state['client']
        if 'encoding' in state:
            del state['encoding']
        return state
    
    def __setstate__(self, state):
        """Restore object after deserialization."""
        self.__dict__.update(state)
        # Recreate non-serializable components
        self._setup_openai()
        self._setup_encoding()
        self._setup_logging()
    
    def _ensure_cache_dir(self):
        """Ensure cache directory exists."""
        import os
        cache_dir = os.path.join("data", "cache")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
            self.logger.info(f"Created cache directory: {cache_dir}")

    def _load_cache(self, cache_id: str) -> Optional[Any]:
        """Load data from cache with error handling."""
        try:
            self._ensure_cache_dir()
            cache_path = os.path.join("data", "cache", f"{cache_id}.pkl")
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            return None
        except Exception as e:
            self.logger.warning(f"Failed to load cache {cache_id}: {str(e)}")
            return None

    def _save_cache(self, cache_id: str, data: Any):
        """Save data to cache with error handling."""
        try:
            self._ensure_cache_dir()
            cache_path = os.path.join("data", "cache", f"{cache_id}.pkl")
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            self.logger.warning(f"Failed to save cache {cache_id}: {str(e)}")

    @classmethod
    def load(cls, filepath: str) -> 'BaseRAG':
        """Load RAG system from file."""
        try:
            with open(filepath, 'rb') as f:
                rag = pickle.load(f)
            if not isinstance(rag, cls):
                raise ValueError(f"Loaded object is not an instance of {cls.__name__}")
            # Reinitialize cache manager after loading
            cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cache")
            rag.cache_manager = CacheManager(cache_dir)
            # Ensure top_k is set
            if not hasattr(rag, 'top_k'):
                rag.top_k = rag.config.get('rag', {}).get('top_k', 3)
            return rag
        except Exception as e:
            raise RuntimeError(f"Failed to load RAG system: {str(e)}")

    def save(self, filepath: str):
        """Save RAG system to file."""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
        except Exception as e:
            raise RuntimeError(f"Failed to save RAG system: {str(e)}")
    
    def _process_query(self, question: str) -> str:
        """Process the query and return the result."""
        try:
            # Get embedding for the question
            question_embedding = self.get_embedding(question)
            
            # Calculate similarities
            similarities = cosine_similarity(
                [question_embedding],
                self.embeddings
            )[0]
            
            # Get top k most similar documents
            top_indices = np.argsort(similarities)[-self.top_k:][::-1]
            relevant_docs = [self.documents[i] for i in top_indices]
            relevant_metadata = [self.metadata[i] for i in top_indices]
            
            # Create prompt
            prompt = self._create_prompt(question, relevant_docs, relevant_metadata)
            
            # Add retry logic for API calls
            max_retries = 3
            retry_delay = 1  # seconds
            
            for attempt in range(max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": self._get_system_prompt()},
                            {"role": "user", "content": prompt}
                        ]
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    if attempt < max_retries - 1:
                        self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
                        time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                    else:
                        raise RuntimeError(f"Failed to get completion after {max_retries} attempts: {str(e)}")
        except Exception as e:
            self.logger.error(f"Failed to process query: {str(e)}")
            raise 