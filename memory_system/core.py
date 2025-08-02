import os
import logging
from uuid import uuid4
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from .models import Memory, MemoryCategory
from .database import DatabaseManager
from .vector_db import VectorDatabase
from .openai_client import OpenAIClient

logger = logging.getLogger(__name__)


class MemorySystem:
    def __init__(self, openai_api_key: Optional[str] = None, 
                 db_path: Optional[str] = None,
                 vector_db_path: Optional[str] = None):
       
        self.db_manager = DatabaseManager(db_path or "./memory_system.db")
        self.vector_db = VectorDatabase(vector_db_path or "./memory_db")
        self.openai_client = OpenAIClient(openai_api_key)
        
        logger.info("Memory system initialized successfully")
    
    def process_message(self, user_id: str, message: str, 
                       conversation_id: str) -> Dict[str, Any]:
        start_time = datetime.now()
        extracted_memories = []
        
        try:
            # Get recent memories for context
            recent_memories = self._get_recent_memories_for_context(user_id, limit=5)
            
            memory_data_list = self.openai_client.extract_memories_from_text(message, user_id, recent_memories)
            
            # Handle memory updates if needed
            if memory_data_list and len(memory_data_list) > 0:
                memory_data = memory_data_list[0]  # Should only be one memory per message
                
                # Check if this is an UPDATE action that needs old memory deletion
                if memory_data.get('_action') == 'UPDATE':
                    memory_to_replace = memory_data.get('_memory_to_replace', '')
                    if memory_to_replace:
                        # Find and delete the old memory
                        self._delete_conflicting_memory(user_id, memory_to_replace)
            
            for memory_data in memory_data_list:
                # Clean up internal metadata before creating memory
                clean_memory_data = {k: v for k, v in memory_data.items() if not k.startswith('_')}
                
                memory = self._create_memory_from_data(
                    clean_memory_data, user_id, conversation_id
                )
                
                if memory:
                    if self._save_memory(memory):
                        extracted_memories.append(memory)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "success": True,
                "memories_extracted": len(extracted_memories),
                "memories": [self._memory_to_dict(m) for m in extracted_memories],
                "processing_time": round(processing_time, 2),
                "message": f"Processed message and extracted {len(extracted_memories)} memories"
            }
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return {
                "success": False,
                "error": str(e),
                "memories_extracted": 0,
                "memories": []
            }
    
    def search_memories(self, user_id: str, query: str, 
                       limit: int = 5) -> Dict[str, Any]:
        start_time = datetime.now()
        
        try:
            # Generate embedding for the query
            query_embedding = self.openai_client.generate_embedding(query)
            if not query_embedding:
                return {
                    "success": False,
                    "error": "Failed to generate query embedding",
                    "memories": []
                }
            
            # Search in vector database
            vector_results = self.vector_db.search_memories(
                query_embedding, user_id, limit
            )
            
            # Get full memory details from SQL database
            memories = []
            for result in vector_results:
                memory = self.db_manager.get_memory(result['id'], user_id)
                if memory:
                    memory_dict = self._memory_to_dict(memory)
                    memory_dict['similarity'] = result['similarity']
                    memories.append(memory_dict)
            
            # Generate enhanced response
            memory_contents = [m['content'] for m in memories]
            enhanced_response = self.openai_client.enhance_memory_search(
                query, memory_contents
            )
            
            search_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "success": True,
                "query": query,
                "memories": memories,
                "enhanced_response": enhanced_response,
                "search_time": round(search_time, 2),
                "total_found": len(memories)
            }
            
        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            return {
                "success": False,
                "error": str(e),
                "memories": []
            }
    
    def delete_memory_by_content(self, user_id: str, content_to_delete: str, 
                                reason: str) -> Dict[str, Any]:
       
        try:
            # Search for similar memories to delete
            search_result = self.search_memories(user_id, content_to_delete, limit=3)
            
            if not search_result['success'] or not search_result['memories']:
                return {
                    "success": False,
                    "message": "No matching memories found to delete"
                }
            
            # Delete the most similar memory
            memory_to_delete = search_result['memories'][0]  # Most similar
            memory_id = memory_to_delete['id']
            
            # Delete from both databases
            sql_success = self.db_manager.delete_memory(memory_id, user_id, reason)
            vector_success = self.vector_db.delete_memory(memory_id, user_id)
            
            if sql_success and vector_success:
                return {
                    "success": True,
                    "deleted_memory": memory_to_delete['content'],
                    "reason": reason,
                    "message": "Memory deleted successfully"
                }
            else:
                return {
                    "success": False,
                    "message": "Failed to delete memory from one or both databases"
                }
                
        except Exception as e:
            logger.error(f"Error deleting memory: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_user_memories(self, user_id: str, category: Optional[str] = None, 
                         limit: Optional[int] = None) -> Dict[str, Any]:
        
        try:
            memories = self.db_manager.get_user_memories(user_id, category, limit)
            
            return {
                "success": True,
                "user_id": user_id,
                "memories": [self._memory_to_dict(m) for m in memories],
                "total_count": len(memories),
                "filtered_by_category": category
            }
            
        except Exception as e:
            logger.error(f"Error getting user memories: {e}")
            return {
                "success": False,
                "error": str(e),
                "memories": []
            }
    
    def get_memory_stats(self, user_id: str) -> Dict[str, Any]:
        
        try:
            sql_stats = self.db_manager.get_memory_stats(user_id)
            vector_count = self.vector_db.get_user_memory_count(user_id)
            
            return {
                "success": True,
                "sql_database": sql_stats,
                "vector_database_count": vector_count,
                "databases_in_sync": sql_stats.get('total_memories', 0) == vector_count
            }
            
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_system(self) -> Dict[str, Any]:
        
        results = {
            "openai_connection": False,
            "database_connection": False,
            "vector_db_connection": False,
            "overall_status": "failed"
        }
        
        try:
            # Test OpenAI connection
            results["openai_connection"] = self.openai_client.test_connection()
            
            # Test database (try to get stats)
            db_stats = self.db_manager.get_memory_stats("test_user")
            results["database_connection"] = "error" not in db_stats
            
            # Test vector database (try to get stats)
            vector_stats = self.vector_db.get_database_stats()
            results["vector_db_connection"] = "error" not in vector_stats
            
            # Overall status
            all_working = all([
                results["openai_connection"],
                results["database_connection"], 
                results["vector_db_connection"]
            ])
            results["overall_status"] = "healthy" if all_working else "partial"
            
        except Exception as e:
            logger.error(f"Error testing system: {e}")
            results["error"] = str(e)
        
        return results
    
    def _create_memory_from_data(self, memory_data: Dict[str, Any], 
                                user_id: str, conversation_id: str) -> Optional[Memory]:
        try:
            # Validate required fields
            if not memory_data.get('content'):
                logger.warning("Memory data missing content")
                return None
            
            # Create memory object
            memory = Memory(
                id=str(uuid4()),
                user_id=user_id,
                content=memory_data['content'],
                category=memory_data.get('category', MemoryCategory.OTHER),
                confidence=memory_data.get('confidence', 0.8),
                timestamp=datetime.now(),
                conversation_id=conversation_id,
                tags=memory_data.get('tags', [])
            )
            
            return memory
            
        except Exception as e:
            logger.error(f"Error creating memory from data: {e}")
            return None
    
    def _save_memory(self, memory: Memory) -> bool:
       
        try:
            # Generate embedding
            embedding = self.openai_client.generate_embedding(memory.content)
            if not embedding:
                logger.error(f"Failed to generate embedding for memory {memory.id}")
                return False
            
            # Save to SQL database
            sql_success = self.db_manager.save_memory(memory)
            
            # Save to vector database
            vector_success = self.vector_db.add_memory(memory, embedding)
            
            success = sql_success and vector_success
            if success:
                logger.info(f"Successfully saved memory: {memory.id}")
            else:
                logger.error(f"Failed to save memory to one or both databases: {memory.id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error saving memory: {e}")
            return False
    
    def _memory_to_dict(self, memory: Memory) -> Dict[str, Any]:
       
        return {
            "id": memory.id,
            "content": memory.content,
            "category": memory.category,
            "confidence": memory.confidence,
            "timestamp": memory.timestamp.isoformat(),
            "conversation_id": memory.conversation_id,
            "tags": memory.tags
        }
    
    def _get_recent_memories_for_context(self, user_id: str, limit: int = 5) -> List[str]:
        """
        Get recent memories for context in decision making.
        
        Args:
            user_id: User ID to get memories for
            limit: Maximum number of recent memories to return
            
        Returns:
            List of memory content strings for context
        """
        try:
            # Get recent memories from SQL database
            memories = self.db_manager.get_user_memories(user_id, limit=limit)
            
            # Return just the content for context
            memory_contents = [memory.content for memory in memories]
            
            logger.info(f"Retrieved {len(memory_contents)} memories for context for user {user_id}")
            return memory_contents
            
        except Exception as e:
            logger.error(f"Error getting recent memories for context: {e}")
            return []
    
    def _delete_conflicting_memory(self, user_id: str, memory_content_to_replace: str):
        """
        Find and delete a memory that conflicts with new information.
        
        Args:
            user_id: User ID
            memory_content_to_replace: Content of the memory to find and delete
        """
        try:
            # Search for memories with similar content
            user_memories = self.db_manager.get_user_memories(user_id, limit=20)
            
            # Find the memory that matches the content to replace
            memory_to_delete = None
            for memory in user_memories:
                # Simple content matching - could be enhanced with similarity matching
                if memory_content_to_replace.lower() in memory.content.lower() or memory.content.lower() in memory_content_to_replace.lower():
                    memory_to_delete = memory
                    break
            
            if memory_to_delete:
                # Delete from both SQL and vector databases
                reason = f"Replaced by new information"
                sql_success = self.db_manager.delete_memory(memory_to_delete.id, user_id, reason)
                vector_success = self.vector_db.delete_memory(memory_to_delete.id, user_id)
                
                if sql_success and vector_success:
                    logger.info(f"Successfully deleted conflicting memory: '{memory_to_delete.content}'")
                else:
                    logger.error(f"Failed to delete conflicting memory: '{memory_to_delete.content}'")
            else:
                logger.warning(f"Could not find memory to replace: '{memory_content_to_replace}'")
                
        except Exception as e:
            logger.error(f"Error deleting conflicting memory: {e}")