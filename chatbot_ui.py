# chatbot_ui.py
"""
Streamlit Chat UI for LLM Memory System
Simple chat interface that demonstrates memory capabilities.
"""

import streamlit as st
import requests
import json
import time
from datetime import datetime
import uuid

# Configuration
API_BASE_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="LLM Memory Chat",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .memory-info {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        font-size: 0.9em;
    }
    .user-message {
        background-color: #e1f5fe;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .bot-message {
        background-color: #f3e5f5;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .memory-count {
        color: #666;
        font-size: 0.8em;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_id" not in st.session_state:
    st.session_state.user_id = f"user_{str(uuid.uuid4())[:8]}"
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = f"conv_{str(uuid.uuid4())[:8]}"

def check_api_health():
    """Check if the API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def send_message_to_api(message):
    """Send message to the memory system API."""
    try:
        payload = {
            "user_id": st.session_state.user_id,
            "message": message,
            "conversation_id": st.session_state.conversation_id
        }
        
        response = requests.post(
            f"{API_BASE_URL}/process-and-chat",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Connection Error: {str(e)}")
        return None

def get_user_memories():
    """Get all memories for the current user."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/user-memories/{st.session_state.user_id}",
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def get_user_stats():
    """Get user memory statistics."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/user-stats/{st.session_state.user_id}",
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

# Main UI
st.title("🧠 LLM Memory Chat")
st.markdown("Chat with an AI that remembers your preferences and information!")

# Sidebar
with st.sidebar:
    st.header("Memory System")
    
    # API Status
    if check_api_health():
        st.success("🟢 API Connected")
    else:
        st.error("🔴 API Disconnected")
        st.warning("Make sure your memory system API is running on http://localhost:8000")
        st.stop()
    
    # User Info
    st.subheader("Session Info")
    st.text(f"User ID: {st.session_state.user_id[:12]}...")
    st.text(f"Conversation: {st.session_state.conversation_id[:12]}...")
    
    # Memory Stats
    st.subheader("Memory Stats")
    stats = get_user_stats()
    if stats and stats.get('success'):
        sql_stats = stats.get('sql_database', {})
        total_memories = sql_stats.get('total_memories', 0)
        categories = sql_stats.get('category_counts', {})
        
        st.metric("Total Memories", total_memories)
        
        if categories:
            st.write("**Categories:**")
            for category, count in categories.items():
                st.write(f"• {category}: {count}")
    
    # Show Memories Button
    if st.button("📋 Show All Memories"):
        memories = get_user_memories()
        if memories and memories.get('success'):
            st.subheader("Your Memories")
            for memory in memories.get('memories', []):
                with st.expander(f"{memory['category'].title()}: {memory['content'][:50]}..."):
                    st.write(f"**Content:** {memory['content']}")
                    st.write(f"**Category:** {memory['category']}")
                    st.write(f"**Confidence:** {memory['confidence']}")
                    st.write(f"**Tags:** {', '.join(memory['tags'])}")
                    st.write(f"**Created:** {memory['timestamp'][:19]}")
    
    # Reset button
    if st.button("🔄 New Session"):
        st.session_state.messages = []
        st.session_state.user_id = f"user_{str(uuid.uuid4())[:8]}"
        st.session_state.conversation_id = f"conv_{str(uuid.uuid4())[:8]}"
        st.rerun()

# Main chat area
st.subheader("💬 Chat")

# Display chat messages
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f'<div class="user-message">🧑 **You:** {message["content"]}</div>', 
                   unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-message">🤖 **Assistant:** {message["content"]}</div>', 
                   unsafe_allow_html=True)
        
        # Show memory info if available
        if "memory_info" in message:
            info = message["memory_info"]
            st.markdown(f"""
            <div class="memory-info">
                <strong>Memory System:</strong><br>
                • Extracted {info['memories_extracted']} new memories<br>
                • Found {info['memories_found']} relevant memories<br>
                • Processing time: {info['processing_time']}s
            </div>
            """, unsafe_allow_html=True)

# Chat input
user_input = st.chat_input("Type your message here...")

if user_input:
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Show user message immediately
    st.markdown(f'<div class="user-message">🧑 **You:** {user_input}</div>', 
               unsafe_allow_html=True)
    
    # Show thinking indicator
    with st.spinner("🤔 Thinking and updating memories..."):
        # Send to API
        api_response = send_message_to_api(user_input)
    
    if api_response:
        # Extract response data
        extraction = api_response.get('extraction', {})
        response_data = api_response.get('response', {})
        
        bot_response = response_data.get('enhanced_response', 'I understand your message.')
        
        # Add bot response to chat
        bot_message = {
            "role": "assistant", 
            "content": bot_response,
            "memory_info": {
                "memories_extracted": extraction.get('memories_extracted', 0),
                "memories_found": response_data.get('memories_found', 0),
                "processing_time": extraction.get('processing_time', 0)
            }
        }
        st.session_state.messages.append(bot_message)
        
        # Show bot response
        st.markdown(f'<div class="bot-message">🤖 **Assistant:** {bot_response}</div>', 
                   unsafe_allow_html=True)
        
        # Show memory info
        info = bot_message["memory_info"]
        st.markdown(f"""
        <div class="memory-info">
            <strong>Memory System:</strong><br>
            • Extracted {info['memories_extracted']} new memories<br>
            • Found {info['memories_found']} relevant memories<br>
            • Processing time: {info['processing_time']}s
        </div>
        """, unsafe_allow_html=True)
        
        # Don't auto-rerun to avoid loading issues
        # st.rerun()  # Commented out to prevent continuous loading
    else:
        st.error("Failed to get response from the memory system.")

# Example prompts
st.subheader("💡 Try These Examples")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🛠️ Tell about tools"):
        example = "I use Notion for note-taking, Slack for team communication, and VS Code for coding."
        # Just add to messages, don't auto-rerun
        pass

with col2:
    if st.button("❤️ Share preferences"):
        example = "I prefer dark mode themes and I like Python over JavaScript for backend development."
        # Just add to messages, don't auto-rerun
        pass

with col3:
    if st.button("🔍 Ask about memories"):
        example = "What productivity tools do I use?"
        # Just add to messages, don't auto-rerun
        pass

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em;'>
🧠 LLM Memory System - Chat with an AI that remembers!<br>
The AI will automatically remember important information you share and use it in future conversations.
</div>
""", unsafe_allow_html=True)