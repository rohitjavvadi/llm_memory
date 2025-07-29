import os
import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI

logger = logging.getLogger(__name__)

class OpenAIClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        self.client = OpenAI(api_key=self.api_key)
        self.chat_model = os.getenv("CHAT_MODEL", "gpt-3.5-turbo")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        logger.info(f"OpenAI client initialized with model: {self.chat_model}")
    def extract_memories_from_text(self, text: str, user_id: str) -> List[Dict[str, Any]]:
        try:
            prompt = f"""
            Analyze this message and extract any important information that should be remembered about the user.
            Focus on:
            - Tools and software they use
            - Preferences and likes/dislikes  
            - Work information
            - Personal details they share
            - Skills they mention
            - Goals they express
            
            Message: "{text}"
            
            If there are memories to extract, respond with a JSON array like this:
            [
                {{
                    "content": "User uses Notion for note-taking",
                    "category": "tools", 
                    "confidence": 0.95,
                    "tags": ["notion", "productivity"]
                }}
            ]
            
            If no important memories found, respond with: []
            
            Only extract clear, factual information. Don't make assumptions.
            """
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": "You are a memory extraction assistant. Extract only clear, factual information that should be remembered about the user."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            response_text = response.choices[0].message.content.strip()
            import json
            try:
                memories = json.loads(response_text)
                if isinstance(memories, list):
                    logger.info(f"Extracted {len(memories)} memories from text")
                    return memories
                else:
                    logger.warning("Response was not a list, returning empty")
                    return []
            except json.JSONDecodeError:
                logger.warning(f"Could not parse JSON response: {response_text}")
                return []
        except Exception as e:
            logger.error(f"Error extracting memories: {e}")
            return []
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            embedding = response.data[0].embedding
            logger.debug(f"Generated embedding for text (length: {len(embedding)})")
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    def generate_embeddings_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        if not texts:
            return []
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=texts
            )
            embeddings = [data.embedding for data in response.data]
            logger.info(f"Generated {len(embeddings)} embeddings in batch")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            return [None] * len(texts)
    def enhance_memory_search(self, query: str, found_memories: List[str]) -> str:
        if not found_memories:
            return "I don't have any memories related to that query."
        try:
            memories_text = "\n".join([f"- {memory}" for memory in found_memories])
            prompt = f"""
            The user asked: "{query}"
            
            Based on these memories I have about the user:
            {memories_text}
            
            Provide a helpful, natural response that directly answers their query using the memories.
            Be conversational and concise.
            """
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on stored memories about the user."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error enhancing search response: {e}")
            return f"Based on what I remember: {', '.join(found_memories)}"
    def test_connection(self) -> bool:
        try:
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            logger.info("OpenAI connection test successful")
            return True
        except Exception as e:
            logger.error(f"OpenAI connection test failed: {e}")
            return False