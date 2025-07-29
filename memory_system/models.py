from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field

@dataclass
class Memory:
    id: str
    user_id: str
    content: str
    category: str
    confidence: float
    timestamp: datetime
    conversation_id: str
    tags: List[str]
    is_active: bool = True
    
    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'user_id': self.user_id,
            'content': self.content,
            'category': self.category,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat(),
            'conversation_id': self.conversation_id,
            'tags': ','.join(self.tags),
            'is_active': self.is_active
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Memory':
        return cls(
            id=data['id'],
            user_id=data['user_id'],
            content=data['content'],
            category=data['category'],
            confidence=data['confidence'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            conversation_id=data['conversation_id'],
            tags=data['tags'].split(',') if data['tags'] else [],
            is_active=bool(data['is_active'])
        )

class MemoryExtractRequest(BaseModel):
    user_id: str = Field(...)
    message: str = Field(...)
    conversation_id: str = Field(...)

class MemorySearchRequest(BaseModel):
    user_id: str = Field(...)
    query: str = Field(...)
    limit: int = Field(5, ge=1, le=20)

class MemoryDeleteRequest(BaseModel):
    user_id: str = Field(...)
    memory_content: str = Field(...)
    reason: str = Field(...)

class MemoryResponse(BaseModel):
    id: str
    content: str
    category: str
    confidence: float
    timestamp: str
    tags: List[str]
    
    @classmethod
    def from_memory(cls, memory: Memory) -> 'MemoryResponse':
        return cls(
            id=memory.id,
            content=memory.content,
            category=memory.category,
            confidence=memory.confidence,
            timestamp=memory.timestamp.isoformat(),
            tags=memory.tags
        )

class MemoryListResponse(BaseModel):
    memories: List[MemoryResponse]
    total_count: int
    user_id: str

class ExtractionResult(BaseModel):
    extracted_memories: List[MemoryResponse]
    message: str
    processing_time: float

class SearchResult(BaseModel):
    memories: List[MemoryResponse]
    query: str
    search_time: float
    total_found: int

class MemoryCategory:
    TOOLS = "tools"
    PREFERENCES = "preferences"
    PERSONAL_INFO = "personal_info"
    WORK_INFO = "work_info"
    SKILLS = "skills"
    GOALS = "goals"
    RELATIONSHIPS = "relationships"
    OTHER = "other"
    
    @classmethod
    def get_all_categories(cls) -> List[str]:
        return [
            cls.TOOLS,
            cls.PREFERENCES,
            cls.PERSONAL_INFO,
            cls.WORK_INFO,
            cls.SKILLS,
            cls.GOALS,
            cls.RELATIONSHIPS,
            cls.OTHER
        ]
    
    @classmethod
    def categorize_content(cls, content: str) -> str:
        content_lower = content.lower()
        if any(word in content_lower for word in ['name', 'age', 'live', 'from', 'born']):
            return cls.PERSONAL_INFO
        if any(word in content_lower for word in ['work', 'job', 'company', 'team', 'project']):
            return cls.WORK_INFO
        if any(word in content_lower for word in ['prefer', 'like', 'favorite', 'enjoy']):
            return cls.PREFERENCES
        if any(word in content_lower for word in ['know', 'skill', 'experience', 'expert', 'good at']):
            return cls.SKILLS
        if any(word in content_lower for word in ['want', 'goal', 'plan', 'hope', 'trying to']):
            return cls.GOALS
        if any(word in content_lower for word in ['use', 'tool', 'software', 'app', 'platform']):
            return cls.TOOLS
        return cls.OTHER