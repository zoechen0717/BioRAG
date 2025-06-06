import os
import logging
import datetime
from typing import Optional, Dict, Any
from core.base_rag import BaseRAG

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

class RAGManager:
    def __init__(self, rag_file: str = "bio_rag.pkl"):
        self.rag_file = rag_file
        self.logger = logging.getLogger(__name__)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG system."""
        try:
            rag = BaseRAG.load(self.rag_file)
            stats = {
                "total_documents": len(rag.documents),
                "total_embeddings": len(rag.embeddings),
                "model": rag.model,
                "last_updated": os.path.getmtime(self.rag_file)
            }
            return stats
        except Exception as e:
            self.logger.error(f"Error getting RAG stats: {e}")
            raise

    def create_backup(self, backup_dir: Optional[str] = None) -> str:
        """Create a backup of the RAG system."""
        if not os.path.exists(self.rag_file):
            raise FileNotFoundError(f"RAG file {self.rag_file} not found.")
        
        # Create backup directory if specified
        if backup_dir:
            os.makedirs(backup_dir, exist_ok=True)
            backup_path = os.path.join(backup_dir, os.path.basename(self.rag_file))
        else:
            backup_path = self.rag_file
        
        # Add timestamp to backup filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = f"{backup_path}.{timestamp}.bak"
        
        try:
            with open(self.rag_file, 'rb') as src, open(backup_file, 'wb') as dst:
                dst.write(src.read())
            self.logger.info(f"Backup created: {backup_file}")
            return backup_file
        except Exception as e:
            self.logger.error(f"Error creating backup: {e}")
            raise

    def list_backups(self, backup_dir: Optional[str] = None) -> list:
        """List all available backups."""
        if backup_dir:
            backup_path = os.path.join(backup_dir, os.path.basename(self.rag_file))
        else:
            backup_path = self.rag_file
        
        try:
            backups = [f for f in os.listdir(os.path.dirname(backup_path))
                      if f.startswith(os.path.basename(backup_path)) and f.endswith('.bak')]
            return sorted(backups, reverse=True)
        except Exception as e:
            self.logger.error(f"Error listing backups: {e}")
            raise

    def restore_backup(self, backup_file: str) -> None:
        """Restore a backup of the RAG system."""
        if not os.path.exists(backup_file):
            raise FileNotFoundError(f"Backup file {backup_file} not found.")
        
        try:
            # Create a backup of current state before restoring
            self.create_backup()
            
            # Restore the backup
            with open(backup_file, 'rb') as src, open(self.rag_file, 'wb') as dst:
                dst.write(src.read())
            self.logger.info(f"Restored backup: {backup_file}")
        except Exception as e:
            self.logger.error(f"Error restoring backup: {e}")
            raise

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        manager = RAGManager()
        
        while True:
            print("\nRAG System Management Utility")
            print("1. View RAG statistics")
            print("2. Create backup")
            print("3. List backups")
            print("4. Restore backup")
            print("5. Exit")
            
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == "1":
                stats = manager.get_stats()
                print("\nRAG System Statistics:")
                print(f"Total documents: {stats['total_documents']}")
                print(f"Total embeddings: {stats['total_embeddings']}")
                print(f"Model used: {stats['model']}")
                print(f"Last updated: {datetime.datetime.fromtimestamp(stats['last_updated'])}")
                
            elif choice == "2":
                backup_dir = input("Enter backup directory (optional, press Enter to use default): ").strip()
                backup_file = manager.create_backup(backup_dir if backup_dir else None)
                print(f"Backup created: {backup_file}")
                
            elif choice == "3":
                backup_dir = input("Enter backup directory (optional, press Enter to use default): ").strip()
                backups = manager.list_backups(backup_dir if backup_dir else None)
                print("\nAvailable backups:")
                for backup in backups:
                    print(f"- {backup}")
                    
            elif choice == "4":
                backup_file = input("Enter backup file to restore: ").strip()
                manager.restore_backup(backup_file)
                print("Backup restored successfully.")
                
            elif choice == "5":
                print("Goodbye!")
                break
                
            else:
                print("Invalid choice. Please try again.")
                
    except Exception as e:
        logger.error(f"Error in RAG management: {e}")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 