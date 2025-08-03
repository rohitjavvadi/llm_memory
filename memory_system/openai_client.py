# memory_system/openai_client.py
"""
Simple OpenAI integration for the memory system.
Handles API calls for memory extraction and embedding generation.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI

logger = logging.getLogger(__name__)


class OpenAIClient:
    """
    Simple wrapper for OpenAI API calls.
    Handles memory extraction and embedding generation.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key (will use env var if not provided)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.chat_model = os.getenv("CHAT_MODEL", "gpt-4o")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        
        logger.info(f"OpenAI client initialized with model: {self.chat_model}")
    
    def extract_memories_from_text(self, text: str, user_id: str, recent_memories: List[str] = None) -> List[Dict[str, Any]]:
        """
        Extract memories from text using OpenAI with intelligent memory management.
        
        Args:
            text: Text to extract memories from
            user_id: User ID for context
            
        Returns:
            List of memory dictionaries
        """
        try:
            # Step 1: Decide what memory action to take (ADD/UPDATE/IGNORE)
            memory_decision = self._decide_memory_action(text, user_id, recent_memories or [])
            
            if memory_decision["action"] == "IGNORE":
                logger.info(f"LLM decided to ignore: '{text}'")
                return []
            elif memory_decision["action"] == "UPDATE":
                logger.info(f"LLM decided to update memory for: '{text}'")
                # Return new memory with UPDATE metadata for core.py to handle
                new_memory = memory_decision["new_memory"].copy()
                new_memory["_action"] = "UPDATE"
                new_memory["_memory_to_replace"] = memory_decision.get("memory_to_replace", "")
                return [new_memory]
            elif memory_decision["action"] == "ADD":
                logger.info(f"LLM decided to add new memory for: '{text}'")
                # Extract new memory normally
                return [memory_decision["new_memory"]]
            else:
                logger.warning(f"Unknown memory action: {memory_decision['action']}")
                return []
                
        except Exception as e:
            logger.error(f"Error extracting memories: {e}")
            return []
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for text using OpenAI.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            List of floats representing the embedding, or None if error
        """
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
        """
        Generate embeddings for multiple texts in a single API call.
        More efficient for multiple texts.
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            List of embeddings (same order as input texts)
        """
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
    
    def classify_query_intent(self, query: str) -> str:
        """
        Classify user query to determine appropriate response strategy.
        
        Args:
            query: User's input message
            
        Returns:
            One of: "memory_question", "general_chat", "memory_sharing"
        """
        # First, check for obvious patterns using rule-based classification
        rule_based_result = self._rule_based_classification(query)
        if rule_based_result:
            logger.info(f"Rule-based classification for '{query}': {rule_based_result}")
            return rule_based_result
        
        # If no clear rule matches, use AI classification
        try:
            classification_prompt = f"""
            User message: "{query}"
            
            Classify this message into ONE category based on PERSONAL vs GENERAL context:
            
            1. "memory_question" - User asking about THEIR stored personal information
               Examples: "What is MY name?", "Where do I work?", "What tools do I use?", "What do I prefer?"
               Key indicators: "my", "I", "do I", asking about personal details
            
            2. "general_chat" - Greetings or general questions NOT about personal stored info
               Examples: "Hi", "Hello", "What's good for coding?", "How's the weather?"
               Key indicators: No personal pronouns, seeking general advice
            
            3. "memory_sharing" - User sharing personal information to remember
               Examples: "My name is John", "I use VS Code", "I work at Google"
               Key indicators: Stating facts about themselves
            
            CRITICAL: If they use "I", "my", or "do I" when asking questions, it's usually memory_question!
            
            Respond with only ONE word: memory_question, general_chat, or memory_sharing
            """
            
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": "You are a query classifier. Focus on whether the user is asking about THEIR personal information (memory_question) vs general information (general_chat)."},
                    {"role": "user", "content": classification_prompt}
                ],
                temperature=0.1,
                max_tokens=10
            )
            
            intent = response.choices[0].message.content.strip().lower()
            valid_intents = ["memory_question", "general_chat", "memory_sharing"]
            
            if intent in valid_intents:
                logger.info(f"AI classified '{query}' as: {intent}")
                return intent
            else:
                logger.warning(f"Invalid AI classification '{intent}', defaulting to memory_question")
                return "memory_question"
                
        except Exception as e:
            logger.error(f"Error in AI classification: {e}")
            return "memory_question"  # Safe default
    
    def _rule_based_classification(self, query: str) -> str:
        """
        Rule-based classification for clear patterns.
        
        Args:
            query: User's input message
            
        Returns:
            Classification if pattern matches, None otherwise
        """
        query_lower = query.lower().strip()
        
        # Clear greetings
        if query_lower in ["hi", "hello", "hey", "how are you", "good morning", "good afternoon"]:
            return "general_chat"
        
        # Personal memory questions - key indicators (check these FIRST)
        personal_question_patterns = [
            "what do i",
            "what am i", 
            "where do i",
            "how do i",
            "what is my",
            "what's my",
            "where is my",
            "who am i",
            "what tools do i",
            "what software do i",
            "what do i use",
            "what do i prefer",
            "what tool do i"
        ]
        
        # Check for question patterns first (these take priority)
        for pattern in personal_question_patterns:
            if pattern in query_lower:
                return "memory_question"
        
        # Memory sharing patterns (only after confirming it's not a question)
        sharing_patterns = [
            "my name is",
            "i am",
            "i work at",
            "i prefer",
            "i like"
        ]
        
        # For "i use", make sure it's not a question first
        if "i use" in query_lower and not any(q in query_lower for q in ["what", "do", "which", "?"]):
            return "memory_sharing"
        
        for pattern in sharing_patterns:
            if pattern in query_lower:
                return "memory_sharing"
        
        # General questions (no personal context)
        general_patterns = [
            "what is good",
            "what are good",
            "how to",
            "what's the weather",
            "tell me about",
            "what should i" # This is advice-seeking, not memory
        ]
        
        for pattern in general_patterns:
            if pattern in query_lower:
                return "general_chat"
        
        return None  # No clear pattern, defer to AI
    
    def generate_general_response(self, query: str) -> str:
        """
        Generate conversational response for general queries.
        
        Args:
            query: User's general query
            
        Returns:
            Natural conversational response
        """
        try:
            response_prompt = f"""
            User said: "{query}"
            
            Provide a helpful, natural, friendly response. This is general conversation.
            Be conversational, brief, and appropriate to the query.
            """
            
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": "You are a helpful, friendly assistant having natural conversation."},
                    {"role": "user", "content": response_prompt}
                ],
                temperature=0.7,
                max_tokens=150
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating general response: {e}")
            # Simple fallbacks
            query_lower = query.lower()
            if any(greeting in query_lower for greeting in ["hi", "hello", "hey"]):
                return "Hello! How can I help you today?"
            elif query_lower.startswith(("what", "how", "why", "when", "where")):
                return "I'd be happy to help! Could you provide more details about what you're looking for?"
            else:
                return "I understand. How can I assist you further?"

    def enhance_memory_search(self, query: str, found_memories: List[str]) -> str:
        """
        Use OpenAI to create a natural response based on found memories with smart answer extraction.
        
        Args:
            query: User's search query
            found_memories: List of memory contents that were found
            
        Returns:
            Natural language response with specific answer extraction
        """
        if not found_memories:
            # Determine if it's a specific question that warrants a different response
            query_lower = query.lower()
            if any(q in query_lower for q in ["what is my", "what's my", "where do i", "who am i"]):
                return f"I don't have any specific information about that yet. If you share details about {query_lower}, I'll remember it for future conversations!"
            return "I don't have any memories related to that query."
        
        try:
            memories_text = "\n".join([f"- {memory}" for memory in found_memories])
            
            prompt = f"""
            The user asked: "{query}"
            
            Based on these memories I have about the user:
            {memories_text}
            
            Extract the SPECIFIC answer to their question from the memories and respond naturally.
            
            Examples:
            - If they ask "What is my name?" and memory says "User's name is Sarah" → "Your name is Sarah."
            - If they ask "Where do I work?" and memory says "Sarah works at Tech Corp" → "You work at Tech Corp."
            - If they ask "What tools do I use?" and memory says "User uses VS Code" → "You use VS Code for coding."
            
            Focus on directly answering their specific question using the stored information.
            Be conversational and concise.
            """
            
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that extracts specific answers from stored memories to answer user questions directly and naturally."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error enhancing search response: {e}")
            # Fallback to simple response
            return f"Based on what I remember: {', '.join(found_memories)}"
    
    def test_connection(self) -> bool:
        """
        Test if OpenAI API connection is working.
        
        Returns:
            True if connection works, False otherwise
        """
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
    
    def _should_remember_this(self, text: str, user_id: str) -> bool:
        """
        Decide if we should extract memories from this message.
        Let the LLM make the decision dynamically instead of hardcoded rules.
        
        Args:
            text: The user's message
            user_id: User ID for context
            
        Returns:
            True if we should extract memories, False otherwise
        """
        try:
            decision_prompt = f"""
            A user said: "{text}"
            
            Is the user SHARING information about themselves that I should remember?
            
            YES examples:
            - "I use Spotify for music" (sharing a tool they use)
            - "I prefer dark mode" (sharing a preference)
            - "My name is John" (sharing personal info)
            - "I work at Google" (sharing work info)
            
            NO examples:
            - "What tools do I use?" (asking a question)
            - "Hello" (greeting)
            - "Can you help me?" (request for assistance)
            - "How do I do this?" (asking for instructions)
            
            Key rule: If it starts with "What", "How", "Can", "Do", "Are", "Is" - it's usually a question, answer NO.
            
            Respond with only YES or NO.
            """
            
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": "You decide whether user messages contain information worth remembering. Be thoughtful but not overly restrictive."},
                    {"role": "user", "content": decision_prompt}
                ],
                temperature=0.1,  # Low temperature for consistent decisions
                max_tokens=10     # We only need YES/NO
            )
            
            decision = response.choices[0].message.content.strip().upper()
            should_remember = decision.startswith("YES")
            
            logger.info(f"Memory decision for '{text}': {decision} → {should_remember}")
            return should_remember
            
        except Exception as e:
            logger.error(f"Error in memory decision for '{text}': {e}")
            # Default to True if we can't decide - better to over-remember than miss important info
            return True
    
    def _decide_memory_action(self, text: str, user_id: str, recent_memories: List[str]) -> Dict[str, Any]:
        """
        Decide what memory action to take: ADD, UPDATE, or IGNORE.
        This replaces the simple should_remember_this with intelligent memory management.
        
        Args:
            text: The user's message
            user_id: User ID for context
            
        Returns:
            Dictionary with action and relevant data
        """
        try:
            # Use provided recent memories for context
            existing_memories = recent_memories if recent_memories else []
            
            decision_prompt = f"""
            User said: "{text}"
            
            Their recent memories: {existing_memories}
            
            What memory action should I take?
            
            OPTIONS:
            - ADD: Store completely new information
            - UPDATE: User changed/updated previous information 
            - IGNORE: Don't store anything (questions, greetings, etc.)
            
            EXAMPLES:
            - "I use Spotify" (no existing music app) → ADD
            - "I don't use Spotify anymore, I use Apple Music" (has Spotify memory) → UPDATE
            - "I still use VSCode" (already have VSCode memory) → IGNORE  
            - "What tools do I use?" → IGNORE
            - "Hello" → IGNORE
            
            If UPDATE, identify:
            1. Which existing memory contradicts the new information
            2. What the new memory should be
            
            Respond with JSON only:
            {{
                "action": "ADD|UPDATE|IGNORE",
                "reasoning": "brief explanation",
                "memory_to_replace": "content_of_old_memory_if_updating",
                "new_memory": {{
                    "content": "new memory content",
                    "category": "tools|preferences|personal_info|work_info|skills|goals|other",
                    "confidence": 0.9,
                    "tags": ["relevant", "tags"]
                }}
            }}
            
            If IGNORE, only include action and reasoning.
            """
            
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": "You are a smart memory manager. Analyze user messages and decide how to update their memory profile intelligently."},
                    {"role": "user", "content": decision_prompt}
                ],
                temperature=0.1,  # Low temperature for consistent decisions
                max_tokens=300
            )
            
            response_text = response.choices[0].message.content.strip()
            logger.info(f"Memory decision for '{text}': {response_text}")
            
            # Parse JSON response
            import json, re
            try:
                # Try direct JSON parse first
                decision = json.loads(response_text)
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code blocks
                json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
                if json_match:
                    decision = json.loads(json_match.group(1))
                else:
                    # Try to find any JSON object
                    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                    if json_match:
                        decision = json.loads(json_match.group())
                    else:
                        raise ValueError("No valid JSON found in response")
            
            return decision
            
        except Exception as e:
            logger.error(f"Error in memory decision for '{text}': {e}")
            # Default to ADD behavior if we can't decide
            return {
                "action": "ADD",
                "reasoning": "Fallback due to decision error",
                "new_memory": {
                    "content": f"User mentioned: {text}",
                    "category": "other",
                    "confidence": 0.5,
                    "tags": ["fallback"]
                }
            }
    
    
    def _handle_memory_update(self, decision: Dict[str, Any], user_id: str) -> List[Dict[str, Any]]:
        """
        Handle UPDATE action - remove old memory and return new one.
        
        Args:
            decision: The decision dictionary from _decide_memory_action
            user_id: User ID
            
        Returns:
            List containing the new memory to store
        """
        try:
            # TODO: Implement actual memory deletion
            # For now, we'll just log what should be deleted and return the new memory
            old_memory = decision.get("memory_to_replace", "Unknown")
            logger.info(f"Should delete old memory: '{old_memory}'")
            logger.info(f"Will add new memory: '{decision['new_memory']['content']}'")
            
            return [decision["new_memory"]]
            
        except Exception as e:
            logger.error(f"Error handling memory update: {e}")
            return []