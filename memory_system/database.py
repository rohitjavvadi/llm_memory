"""
Database layer for the LLM Memory System.
Handles all database operations for storing and retrieving memories.
"""

import sqlite3
import os
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from contextlib import contextmanager

from .models import Memory

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages SQLite database operations for memory storage.
    Handles database initialization, CRUD operations, and connection management.
    """
    
    def __init__(self, db_path: str = "./memory_system.db"):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.init_database()
        logger.info(f"Database initialized at: {db_path}")
    
    def init_database(self):
        """Create database tables if they don't exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Create memories table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    category TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    tags TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create index for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_memories 
                ON memories(user_id, is_active)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_memory_category 
                ON memories(user_id, category, is_active)
            """)
            
            # Create memory_relationships table for tracking updates/deletions
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memory_relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id TEXT NOT NULL,
                    related_memory_id TEXT,
                    relationship_type TEXT NOT NULL,
                    reason TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (memory_id) REFERENCES memories (id),
                    FOREIGN KEY (related_memory_id) REFERENCES memories (id)
                )
            """)
            
            conn.commit()
            logger.info("Database tables created successfully")
    
    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections.
        Ensures connections are properly closed.
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Enable column access by name
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def save_memory(self, memory: Memory) -> bool:
        """
        Save a memory to the database.
        
        Args:
            memory: Memory object to save
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                memory_data = memory.to_dict()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO memories 
                    (id, user_id, content, category, confidence, timestamp, 
                     conversation_id, tags, is_active, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    memory_data['id'],
                    memory_data['user_id'],
                    memory_data['content'],
                    memory_data['category'],
                    memory_data['confidence'],
                    memory_data['timestamp'],
                    memory_data['conversation_id'],
                    memory_data['tags'],
                    memory_data['is_active'],
                    datetime.now().isoformat()
                ))
                
                conn.commit()
                logger.info(f"Memory saved: {memory.id}")
                return True
                
        except Exception as e:
            logger.error(f"Error saving memory {memory.id}: {e}")
            return False
    
    def get_memory(self, memory_id: str, user_id: str) -> Optional[Memory]:
        """
        Retrieve a specific memory by ID.
        
        Args:
            memory_id: ID of the memory to retrieve
            user_id: User ID for security check
            
        Returns:
            Memory object if found, None otherwise
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM memories 
                    WHERE id = ? AND user_id = ? AND is_active = 1
                """, (memory_id, user_id))
                
                row = cursor.fetchone()
                if row:
                    return Memory.from_dict(dict(row))
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving memory {memory_id}: {e}")
            return None
    
    def get_user_memories(self, user_id: str, category: Optional[str] = None, 
                         limit: Optional[int] = None) -> List[Memory]:
        """
        Get all memories for a user.
        
        Args:
            user_id: User ID to get memories for
            category: Optional category filter
            limit: Optional limit on number of results
            
        Returns:
            List of Memory objects
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                query = """
                    SELECT * FROM memories 
                    WHERE user_id = ? AND is_active = 1
                """
                params = [user_id]
                
                if category:
                    query += " AND category = ?"
                    params.append(category)
                
                query += " ORDER BY timestamp DESC"
                
                if limit:
                    query += " LIMIT ?"
                    params.append(limit)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                memories = [Memory.from_dict(dict(row)) for row in rows]
                logger.info(f"Retrieved {len(memories)} memories for user {user_id}")
                return memories
                
        except Exception as e:
            logger.error(f"Error retrieving memories for user {user_id}: {e}")
            return []
    
    def search_memories_by_content(self, user_id: str, search_term: str, 
                                  limit: int = 10) -> List[Memory]:
        """
        Search memories by content (basic text search).
        This will be enhanced with vector search later.
        
        Args:
            user_id: User ID to search within
            search_term: Text to search for
            limit: Maximum number of results
            
        Returns:
            List of matching Memory objects
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM memories 
                    WHERE user_id = ? AND is_active = 1 
                    AND (content LIKE ? OR tags LIKE ?)
                    ORDER BY confidence DESC, timestamp DESC
                    LIMIT ?
                """, (user_id, f"%{search_term}%", f"%{search_term}%", limit))
                
                rows = cursor.fetchall()
                memories = [Memory.from_dict(dict(row)) for row in rows]
                
                logger.info(f"Found {len(memories)} memories matching '{search_term}'")
                return memories
                
        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            return []
    
    def delete_memory(self, memory_id: str, user_id: str, reason: str) -> bool:
        """
        Soft delete a memory (mark as inactive).
        
        Args:
            memory_id: ID of memory to delete
            user_id: User ID for security check
            reason: Reason for deletion
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Mark memory as inactive
                cursor.execute("""
                    UPDATE memories 
                    SET is_active = 0, updated_at = ?
                    WHERE id = ? AND user_id = ?
                """, (datetime.now().isoformat(), memory_id, user_id))
                
                if cursor.rowcount > 0:
                    # Record the deletion relationship
                    cursor.execute("""
                        INSERT INTO memory_relationships 
                        (memory_id, relationship_type, reason)
                        VALUES (?, 'deleted', ?)
                    """, (memory_id, reason))
                    
                    conn.commit()
                    logger.info(f"Memory {memory_id} deleted: {reason}")
                    return True
                else:
                    logger.warning(f"Memory {memory_id} not found for deletion")
                    return False
                
        except Exception as e:
            logger.error(f"Error deleting memory {memory_id}: {e}")
            return False
    
    def get_memory_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Get statistics about user's memories.
        
        Args:
            user_id: User ID to get stats for
            
        Returns:
            Dictionary with memory statistics
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Total active memories
                cursor.execute("""
                    SELECT COUNT(*) FROM memories 
                    WHERE user_id = ? AND is_active = 1
                """, (user_id,))
                total_memories = cursor.fetchone()[0]
                
                # Memories by category
                cursor.execute("""
                    SELECT category, COUNT(*) FROM memories 
                    WHERE user_id = ? AND is_active = 1
                    GROUP BY category
                """, (user_id,))
                category_counts = dict(cursor.fetchall())
                
                # Average confidence
                cursor.execute("""
                    SELECT AVG(confidence) FROM memories 
                    WHERE user_id = ? AND is_active = 1
                """, (user_id,))
                avg_confidence = cursor.fetchone()[0] or 0.0
                
                return {
                    'total_memories': total_memories,
                    'category_counts': category_counts,
                    'average_confidence': round(avg_confidence, 2),
                    'user_id': user_id
                }
                
        except Exception as e:
            logger.error(f"Error getting memory stats for user {user_id}: {e}")
            return {'error': str(e)}