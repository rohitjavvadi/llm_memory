# LLM Memory System

A long-term memory system for Large Language Models that enables persistent memory capabilities through intelligent storage and retrieval of user information.

## What it does

This system acts like a digital memory for AI conversations. When you chat with the AI, it automatically remembers important information about you - your preferences, tools you use, work details, and personal information. Later, when you ask questions or have new conversations, the AI can recall and use this stored information to provide more personalized and contextual responses.

## Features

- **Smart Memory Extraction**: Automatically identifies and stores important information from conversations
- **Semantic Search**: Find relevant memories using natural language queries
- **Dual Storage**: Uses both SQL database for structured data and vector database for similarity search
- **Memory Management**: Intelligently updates or replaces outdated information
- **Category Organization**: Automatically categorizes memories (tools, preferences, personal info, work, etc.)
- **RESTful API**: Clean HTTP endpoints for integration with other applications
- **Chat Interface**: Simple Streamlit-based web UI for testing and interaction

## Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key

### Installation

1. Clone the repository and navigate to the project directory
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   ```

### Running the System

1. Start the API server:
   ```bash
   python main.py
   ```

2. In a separate terminal, start the chat interface:
   ```bash
   streamlit run chatbot_ui.py
   ```

3. Open your browser to `http://localhost:8501` to start chatting!

## How to Use

### Sharing Information
Tell the AI about yourself and it will remember:
- "I use VS Code for programming"
- "I prefer dark mode themes"
- "My name is Sarah and I work at Tech Corp"

### Asking Questions
Query your stored memories:
- "What coding tools do I use?"
- "What are my preferences?"
- "What do you remember about my work?"

### API Endpoints
- `POST /process-and-chat` - Main endpoint for chat with memory
- `POST /extract-memories` - Extract memories from text
- `POST /search-memories` - Search stored memories
- `GET /user-memories/{user_id}` - Get all user memories
- `GET /health` - System health check

## Architecture

- **FastAPI Backend**: Handles API requests and memory operations
- **SQLite Database**: Stores structured memory data
- **ChromaDB**: Vector database for semantic similarity search
- **OpenAI Integration**: Uses GPT-4o for memory extraction and response generation
- **Streamlit Frontend**: Simple web interface for testing

## Limitations

- **API Dependency**: Requires active OpenAI API connection and consumes tokens for all operations, which can become costly with heavy usage
- **Memory Quality**: The system may occasionally store irrelevant information or miss important details, as it relies on AI interpretation of conversation context
- **General Query Handling**: Currently focuses on basic memory-related responses and requires further development to handle general conversational queries and broader knowledge and memory questions effectively

## Development

The system is structured as a Python package with clear separation of concerns:
- `memory_system/core.py` - Main orchestration logic
- `memory_system/database.py` - SQL database operations  
- `memory_system/vector_db.py` - Vector database operations
- `memory_system/openai_client.py` - OpenAI API integration
- `main.py` - FastAPI application
- `chatbot_ui.py` - Streamlit interface

## Contributing

