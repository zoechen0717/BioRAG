import os
import pickle
import logging
from pathlib import Path
from typing import Any, Optional

class CacheManager:
    """Manages caching operations for the RAG system."""
    
    def __init__(self, cache_dir: str):
        """Initialize the cache manager.
        
        Args:
            cache_dir: Path to the cache directory
        """
        self.cache_dir = cache_dir
        self._setup_logging()
        self._ensure_cache_dir()
    
    def _setup_logging(self):
        """Set up logging for the cache manager."""
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _ensure_cache_dir(self):
        """Ensure the cache directory exists."""
        try:
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir, exist_ok=True)
                self.logger.info(f"Created cache directory at: {self.cache_dir}")
            else:
                self.logger.info(f"Using existing cache directory at: {self.cache_dir}")
        except Exception as e:
            self.logger.error(f"Failed to create cache directory: {str(e)}")
            raise
    
    def get(self, cache_id: str) -> Optional[Any]:
        """Get an item from the cache.
        
        Args:
            cache_id: Unique identifier for the cached item
            
        Returns:
            The cached item if found, None otherwise
        """
        try:
            cache_path = os.path.join(self.cache_dir, f"{cache_id}.pkl")
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            return None
        except Exception as e:
            self.logger.error(f"Failed to get cache item {cache_id}: {str(e)}")
            return None
    
    def set(self, cache_id: str, value: Any):
        """Set an item in the cache.
        
        Args:
            cache_id: Unique identifier for the cached item
            value: Value to cache
        """
        try:
            cache_path = os.path.join(self.cache_dir, f"{cache_id}.pkl")
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
            self.logger.debug(f"Cached item {cache_id}")
        except Exception as e:
            self.logger.error(f"Failed to set cache item {cache_id}: {str(e)}")
    
    def clear(self):
        """Clear all cached items."""
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.pkl'):
                    os.remove(os.path.join(self.cache_dir, filename))
            self.logger.info("Cleared all cache items")
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {str(e)}")
    
    def remove(self, cache_id: str):
        """Remove a specific item from the cache.
        
        Args:
            cache_id: Unique identifier for the cached item
        """
        try:
            cache_path = os.path.join(self.cache_dir, f"{cache_id}.pkl")
            if os.path.exists(cache_path):
                os.remove(cache_path)
                self.logger.debug(f"Removed cache item {cache_id}")
        except Exception as e:
            self.logger.error(f"Failed to remove cache item {cache_id}: {str(e)}") 