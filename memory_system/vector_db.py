# memory_system/vector_db.py
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings

from .models import Memory

logger = logging.getLogger(__name__)


class VectorDatabase:
 
    def __init__(self, db_path: str = "./memory_db"):
      
        self.db_path = db_path
        
        
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(
                anonymized_telemetry=False,  
                allow_reset=True
            )
        )
        
        self.collection = self.client.get_or_create_collection(
            name="memories",
            metadata={"description": "Memory embeddings for semantic search"}
        )
        
        logger.info(f"Vector database initialized at: {db_path}")
    
    def add_memory(self, memory: Memory, embedding: List[float]) -> bool:
        try:
            metadata = {
                "user_id": memory.user_id,
                "category": memory.category,
                "confidence": memory.confidence,
                "timestamp": memory.timestamp.isoformat(),
                "conversation_id": memory.conversation_id,
                "tags": ",".join(memory.tags)
            }
            self.collection.add(
                ids=[memory.id],
                embeddings=[embedding],
                documents=[memory.content],
                metadatas=[metadata]
            )
            
            logger.debug(f"Added memory to vector DB: {memory.id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding memory to vector DB: {e}")
            return False
    
    def search_memories(self, query_embedding: List[float], user_id: str, 
                       limit: int = 5, min_similarity: float = 0.0) -> List[Dict[str, Any]]:
        try:
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                where={"user_id": user_id}  # Filter by user
            )
            memory_results = []
            if results['ids'] and results['ids'][0]:  # Check if we have results
                for i in range(len(results['ids'][0])):
                    similarity = 1 - results['distances'][0][i]  # Convert distance to similarity
                    
                    # Filter by minimum similarity
                    if similarity >= min_similarity:
                        memory_results.append({
                            'id': results['ids'][0][i],
                            'content': results['documents'][0][i],
                            'metadata': results['metadatas'][0][i],
                            'similarity': round(similarity, 3)
                        })
            
            logger.info(f"Found {len(memory_results)} similar memories for user {user_id}")
            return memory_results
            
        except Exception as e:
            logger.error(f"Error searching vector DB: {e}")
            return []
    
    def update_memory(self, memory_id: str, new_content: str, 
                     new_embedding: List[float], user_id: str) -> bool:
        try:
            # Check if memory exists and belongs to user
            existing = self.collection.get(
                ids=[memory_id],
                where={"user_id": user_id}
            )
            
            if not existing['ids']:
                logger.warning(f"Memory {memory_id} not found for user {user_id}")
                return False
            
            self.collection.update(
                ids=[memory_id],
                embeddings=[new_embedding],
                documents=[new_content]
            )
            
            logger.info(f"Updated memory in vector DB: {memory_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating memory in vector DB: {e}")
            return False
    
    def delete_memory(self, memory_id: str, user_id: str) -> bool:
        try:
            # Check if memory exists and belongs to user
            existing = self.collection.get(
                ids=[memory_id],
                where={"user_id": user_id}
            )
            
            if not existing['ids']:
                logger.warning(f"Memory {memory_id} not found for user {user_id}")
                return False
            
            # Delete from ChromaDB
            self.collection.delete(ids=[memory_id])
            
            logger.info(f"Deleted memory from vector DB: {memory_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting memory from vector DB: {e}")
            return False
    
    def get_user_memory_count(self, user_id: str) -> int:
        try:
            results = self.collection.get(where={"user_id": user_id})
            count = len(results['ids']) if results['ids'] else 0
            logger.debug(f"User {user_id} has {count} memories in vector DB")
            return count
            
        except Exception as e:
            logger.error(f"Error counting memories for user {user_id}: {e}")
            return 0
    
    def search_by_category(self, user_id: str, category: str, 
                          limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get all memories in a specific category for a user.
        
        Args:
            user_id: User ID to search for
            category: Category to filter by
            limit: Maximum number of results
            
        Returns:
            List of memories in the category
        """
        try:
            results = self.collection.get(
                where={
                    "user_id": user_id,
                    "category": category
                },
                limit=limit
            )
            
            # Format results
            memory_results = []
            if results['ids']:
                for i in range(len(results['ids'])):
                    memory_results.append({
                        'id': results['ids'][i],
                        'content': results['documents'][i],
                        'metadata': results['metadatas'][i]
                    })
            
            logger.info(f"Found {len(memory_results)} memories in category '{category}' for user {user_id}")
            return memory_results
            
        except Exception as e:
            logger.error(f"Error searching by category: {e}")
            return []
    
    def reset_user_data(self, user_id: str) -> bool:
        """
        Delete all memories for a specific user.
        
        Args:
            user_id: User ID to delete data for
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get all memories for the user
            results = self.collection.get(where={"user_id": user_id})
            
            if results['ids']:
                # Delete all user memories
                self.collection.delete(ids=results['ids'])
                logger.info(f"Deleted {len(results['ids'])} memories for user {user_id}")
            else:
                logger.info(f"No memories found for user {user_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error resetting data for user {user_id}: {e}")
            return False
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector database.
        
        Returns:
            Dictionary with database statistics
        """
        try:
            # Get total count
            all_results = self.collection.get()
            total_memories = len(all_results['ids']) if all_results['ids'] else 0
            
            # Count unique users
            unique_users = set()
            if all_results['metadatas']:
                for metadata in all_results['metadatas']:
                    unique_users.add(metadata.get('user_id', 'unknown'))
            
            return {
                'total_memories': total_memories,
                'unique_users': len(unique_users),
                'collection_name': self.collection.name,
                'database_path': self.db_path
            }
            
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {'error': str(e)}