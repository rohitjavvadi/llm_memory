
import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from memory_system import (
    MemoryExtractRequest,
    MemorySearchRequest,
    MemoryDeleteRequest,
    MemoryListResponse,
    ExtractionResult,
    SearchResult
)
from memory_system.core import MemorySystem

load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

memory_system = None


@asynccontextmanager
async def lifespan(app: FastAPI):
  
    global memory_system
    
    try:
        logger.info("Initializing memory system...")
        memory_system = MemorySystem()
        
        health = memory_system.test_system()
        if health["overall_status"] == "healthy":
            logger.info("✅ Memory system initialized successfully")
        else:
            logger.warning(f"⚠️ Memory system partially initialized: {health}")
        
        yield  
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize memory system: {e}")
        raise
    finally:
        logger.info("Shutting down memory system...")


app = FastAPI(
    title="LLM Memory System",
    description="Long-term memory system for Large Language Models",
    version="1.0.0",
    lifespan=lifespan
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "message": "LLM Memory System API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    
    if not memory_system:
        raise HTTPException(status_code=503, detail="Memory system not initialized")
    
    try:
        health_results = memory_system.test_system()
        
        if health_results["overall_status"] == "healthy":
            return {
                "status": "healthy",
                "components": health_results,
                "message": "All systems operational"
            }
        else:
            return {
                "status": "degraded",
                "components": health_results,
                "message": "Some components may not be working properly"
            }
            
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")


@app.post("/extract-memories", response_model=ExtractionResult)
async def extract_memories(request: MemoryExtractRequest):
    if not memory_system:
        raise HTTPException(status_code=503, detail="Memory system not initialized")
    
    try:
        result = memory_system.process_message(
            user_id=request.user_id,
            message=request.message,
            conversation_id=request.conversation_id
        )
        
        if result["success"]:
            return ExtractionResult(
                extracted_memories=[],  # Will be populated by the actual memories
                message=result["message"],
                processing_time=result["processing_time"]
            )
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "Failed to extract memories"))
            
    except Exception as e:
        logger.error(f"Error extracting memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search-memories", response_model=SearchResult)
async def search_memories(request: MemorySearchRequest):
    if not memory_system:
        raise HTTPException(status_code=503, detail="Memory system not initialized")
    
    try:
        result = memory_system.search_memories(
            user_id=request.user_id,
            query=request.query,
            limit=request.limit
        )
        
        if result["success"]:
            return SearchResult(
                memories=[],  # Will be populated by actual memories
                query=result["query"],
                search_time=result["search_time"],
                total_found=result["total_found"]
            )
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "Search failed"))
            
    except Exception as e:
        logger.error(f"Error searching memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat-with-memory")
async def chat_with_memory(request: MemorySearchRequest):
    if not memory_system:
        raise HTTPException(status_code=503, detail="Memory system not initialized")
    
    try:
        result = memory_system.search_memories(
            user_id=request.user_id,
            query=request.query,
            limit=request.limit
        )
        
        if result["success"]:
            return {
                "query": request.query,
                "response": result.get("enhanced_response", "No relevant memories found."),
                "memories_used": len(result["memories"]),
                "search_time": result["search_time"]
            }
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "Chat failed"))
            
    except Exception as e:
        logger.error(f"Error in chat with memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/delete-memory")
async def delete_memory(request: MemoryDeleteRequest):
   
    if not memory_system:
        raise HTTPException(status_code=503, detail="Memory system not initialized")
    
    try:
        result = memory_system.delete_memory_by_content(
            user_id=request.user_id,
            content_to_delete=request.memory_content,
            reason=request.reason
        )
        
        if result["success"]:
            return {
                "message": result["message"],
                "deleted_memory": result.get("deleted_memory"),
                "reason": result.get("reason")
            }
        else:
            raise HTTPException(status_code=404, detail=result.get("message", "Memory not found"))
            
    except Exception as e:
        logger.error(f"Error deleting memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/user-memories/{user_id}", response_model=MemoryListResponse)
async def get_user_memories(user_id: str, category: str = None, limit: int = None):
    
    if not memory_system:
        raise HTTPException(status_code=503, detail="Memory system not initialized")
    
    try:
        result = memory_system.get_user_memories(
            user_id=user_id,
            category=category,
            limit=limit
        )
        
        if result["success"]:
            return MemoryListResponse(
                memories=[],  # Will be populated by actual memories
                total_count=result["total_count"],
                user_id=user_id
            )
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "Failed to get memories"))
            
    except Exception as e:
        logger.error(f"Error getting user memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/user-stats/{user_id}")
async def get_user_stats(user_id: str):
    
    if not memory_system:
        raise HTTPException(status_code=503, detail="Memory system not initialized")
    
    try:
        result = memory_system.get_memory_stats(user_id)
        
        if result["success"]:
            return result
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "Failed to get stats"))
            
    except Exception as e:
        logger.error(f"Error getting user stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process-and-chat")
async def process_and_chat(request: MemoryExtractRequest):
   
    if not memory_system:
        raise HTTPException(status_code=503, detail="Memory system not initialized")
    
    try:
        # Step 1: Extract memories from the message
        extraction_result = memory_system.process_message(
            user_id=request.user_id,
            message=request.message,
            conversation_id=request.conversation_id
        )
        
        # Step 2: Search for relevant memories to enhance response
        search_result = memory_system.search_memories(
            user_id=request.user_id,
            query=request.message,
            limit=5
        )
        
        return {
            "extraction": {
                "success": extraction_result["success"],
                "memories_extracted": extraction_result["memories_extracted"],
                "processing_time": extraction_result["processing_time"]
            },
            "response": {
                "enhanced_response": search_result.get("enhanced_response", "I understand your message."),
                "memories_found": len(search_result.get("memories", [])),
                "search_time": search_result.get("search_time", 0)
            },
            "message": "Message processed and response generated"
        }
        
    except Exception as e:
        logger.error(f"Error in process and chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    # Get configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    print(f"""
🚀 Starting LLM Memory System API
📍 URL: http://{host}:{port}
📚 Docs: http://{host}:{port}/docs
🔍 Health: http://{host}:{port}/health

Make sure your .env file contains:
OPENAI_API_KEY=your_key_here
    """)
    
    # Run the application
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )