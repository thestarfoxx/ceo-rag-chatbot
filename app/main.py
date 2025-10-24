import streamlit as st
import sys
import os
from pathlib import Path

# Add src directory to Python path for imports
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Import components
from components.sidebar import create_sidebar
from pages.chat import render_chat_page
from utils.session_state import initialize_session_state
from utils.config import load_config

# Configure page
st.set_page_config(
    page_title="CEO RAG Chatbot Workspace",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main application entry point."""
    
    # Initialize session state
    initialize_session_state()
    
    # Load configuration
    config = load_config()
    
    # Custom CSS for workspace styling
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #1f4e79 0%, #2980b9 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .workspace-container {
        display: flex;
        gap: 1rem;
        height: 80vh;
    }
    
    .email-panel {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #e9ecef;
    }
    
    .chat-panel {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #e9ecef;
        flex: 2;
    }
    
    .drafter-panel {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #e9ecef;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #1f4e79 0%, #2980b9 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #2980b9 0%, #3498db 100%);
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        border-left: 4px solid #2980b9;
    }
    
    .user-message {
        background: #e3f2fd;
        margin-left: 2rem;
    }
    
    .assistant-message {
        background: #f5f5f5;
        margin-right: 2rem;
    }
    
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 5px;
    }
    
    .status-online {
        background-color: #4caf50;
    }
    
    .status-offline {
        background-color: #f44336;
    }
    
    .email-item {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #2196f3;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .email-item:hover {
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transform: translateY(-1px);
    }
    
    .email-subject {
        font-weight: bold;
        color: #1f4e79;
        margin-bottom: 0.5rem;
    }
    
    .email-sender {
        color: #666;
        font-size: 0.9rem;
    }
    
    .email-preview {
        color: #888;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }
    
    .priority-high {
        border-left-color: #f44336 !important;
    }
    
    .priority-medium {
        border-left-color: #ff9800 !important;
    }
    
    .priority-low {
        border-left-color: #4caf50 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 CEO RAG Chatbot Workspace</h1>
        <p>Intelligent Assistant for Executive Decision Making</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create sidebar
    create_sidebar()
    
    # Main workspace
    render_chat_page()
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            "<div style='text-align: center; color: #666;'>"
            "🚀 Powered by OpenAI • 📊 PostgreSQL • 🔍 Vector Search"
            "</div>", 
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()
