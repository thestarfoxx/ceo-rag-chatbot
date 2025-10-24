import streamlit as st
from datetime import datetime
from typing import Dict, Any
import sys
from pathlib import Path

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.session_state import (
    get_or_create_agent, 
    add_chat_message,
    get_conversation_context,
    add_sample_emails,
    get_unread_emails,
    get_emails_by_priority,
    mark_email_read,
    get_email_by_id,
    save_drafted_email
)
from components.email_panel import render_email_panel
from components.email_drafter import render_email_drafter
from components.chat_interface import render_chat_interface

def render_chat_page():
    """Render the main chat page with three-column layout."""
    
    # Initialize sample emails if none exist
    if not st.session_state.emails:
        add_sample_emails()
    
    # Create three columns for the workspace layout
    col1, col2, col3 = st.columns([1.2, 2, 1.2])
    
    with col1:
        render_email_panel()
    
    with col2:
        render_chat_interface()
    
    with col3:
        render_email_drafter()

def process_chat_query(query: str) -> Dict[str, Any]:
    """Process a chat query and return the response."""
    
    if not query.strip():
        return {
            "success": False,
            "error": "Please enter a query"
        }
    
    # Get the conversational agent
    agent = get_or_create_agent()
    if not agent:
        return {
            "success": False,
            "error": "Conversational agent not available"
        }
    
    try:
        # Get conversation context
        context = get_conversation_context()
        
        # Process the query
        with st.spinner("Processing your request..."):
            response = agent.process_query(query, context)
        
        return {
            "success": response.success,
            "response": response,
            "agent": agent
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def handle_chat_interaction():
    """Handle chat interaction logic."""
    
    # Chat input
    if query := st.chat_input("Ask me anything about your business data, draft emails, or general questions..."):
        
        # Add user message to chat history
        add_chat_message("user", query)
        
        # Process the query
        result = process_chat_query(query)
        
        if result["success"]:
            response = result["response"]
            
            # Add assistant response to chat history
            add_chat_message(
                "assistant", 
                response.response_text,
                {
                    "action_taken": response.action_taken.value,
                    "processing_time": response.metadata.get('processing_time', 0),
                    "intent_confidence": response.metadata.get('intent_analysis', {}).get('confidence', 0),
                    "success": response.success,
                    "data": response.data
                }
            )
            
            # Handle specific response types
            handle_response_actions(response)
            
        else:
            # Add error message
            error_msg = f"I encountered an error: {result.get('error', 'Unknown error')}"
            add_chat_message("assistant", error_msg, {"success": False, "error": True})
        
        # Rerun to update the chat display
        st.rerun()

def handle_response_actions(response):
    """Handle specific actions based on response type."""
    
    try:
        from conversation import QueryType
    except ImportError:
        try:
            from src.llm.conversation import QueryType
        except ImportError:
            # Fallback - define locally
            from enum import Enum
            class QueryType(Enum):
                EMAIL_DRAFT = "EMAIL_DRAFT"
                EMAIL_READ = "EMAIL_READ"
                SQL_QUERY = "SQL_QUERY"
                DOCUMENT_SEARCH = "DOCUMENT_SEARCH"
                GENERAL_CHAT = "GENERAL_CHAT"
    
    if response.action_taken == QueryType.EMAIL_DRAFT and response.success:
        # Save drafted email
        if response.data:
            save_drafted_email(response.data)
            st.toast("📧 Email draft saved!", icon="✅")
    
    elif response.action_taken == QueryType.EMAIL_READ and response.success:
        # Show email analysis results
        if response.data:
            st.toast("📊 Email analyzed successfully!", icon="✅")
    
    elif response.action_taken == QueryType.SQL_QUERY and response.success:
        # Show data query results
        if response.data and response.data.get('results'):
            count = len(response.data['results'])
            st.toast(f"📊 Found {count} results!", icon="✅")
    
    elif response.action_taken == QueryType.DOCUMENT_SEARCH and response.success:
        # Show document search results
        if response.data and response.data.get('chunks'):
            count = len(response.data['chunks'])
            st.toast(f"📄 Found {count} relevant chunks!", icon="✅")

def render_chat_examples():
    """Render example queries to help users get started."""
    
    st.markdown("### 💡 Try these examples:")
    
    examples = [
        {
            "category": "📧 Email Tasks",
            "queries": [
                "Draft an email to John about the quarterly meeting",
                "Analyze this email: 'Subject: URGENT - Budget Review'",
                "Write a professional email about project delays"
            ]
        },
        {
            "category": "📊 Data Analysis", 
            "queries": [
                "What were our sales figures last quarter?",
                "Show me revenue breakdown by region",
                "Generate a financial performance report"
            ]
        },
        {
            "category": "📄 Document Search",
            "queries": [
                "Find information about marketing strategy",
                "Search for budget projections",
                "Look for competitor analysis data"
            ]
        },
        {
            "category": "💬 General",
            "queries": [
                "What can you help me with?",
                "Show me the system status",
                "What data sources are available?"
            ]
        }
    ]
    
    for example_group in examples:
        with st.expander(example_group["category"]):
            for query in example_group["queries"]:
                if st.button(f"💭 {query}", key=f"example_{hash(query)}", help="Click to use this example"):
                    # Add to chat and process
                    add_chat_message("user", query)
                    result = process_chat_query(query)
                    
                    if result["success"]:
                        response = result["response"]
                        add_chat_message(
                            "assistant", 
                            response.response_text,
                            {
                                "action_taken": response.action_taken.value,
                                "success": response.success
                            }
                        )
                        handle_response_actions(response)
                    else:
                        error_msg = f"Error: {result.get('error', 'Unknown error')}"
                        add_chat_message("assistant", error_msg, {"success": False, "error": True})
                    
                    st.rerun()

def render_chat_statistics():
    """Render chat statistics and metrics."""
    
    chat_history = st.session_state.get('chat_history', [])
    
    if chat_history:
        with st.expander("📈 Chat Statistics"):
            
            # Basic metrics
            user_messages = [msg for msg in chat_history if msg['role'] == 'user']
            assistant_messages = [msg for msg in chat_history if msg['role'] == 'assistant']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Queries", len(user_messages))
            
            with col2:
                successful_responses = sum(1 for msg in assistant_messages 
                                         if msg.get('metadata', {}).get('success', True))
                success_rate = (successful_responses / max(len(assistant_messages), 1)) * 100
                st.metric("Success Rate", f"{success_rate:.1f}%")
            
            with col3:
                processing_times = [
                    msg.get('metadata', {}).get('processing_time', 0)
                    for msg in assistant_messages
                    if msg.get('metadata', {}).get('processing_time')
                ]
                avg_time = sum(processing_times) / max(len(processing_times), 1)
                st.metric("Avg Response Time", f"{avg_time:.2f}s")
            
            # Intent distribution
            intents = [msg.get('metadata', {}).get('action_taken', 'unknown') 
                      for msg in assistant_messages]
            intent_counts = {}
            for intent in intents:
                intent_counts[intent] = intent_counts.get(intent, 0) + 1
            
            if intent_counts:
                st.write("**Query Types:**")
                for intent, count in sorted(intent_counts.items(), key=lambda x: x[1], reverse=True):
                    if intent != 'unknown':
                        percentage = (count / len(intents)) * 100
                        st.write(f"• {intent.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")

def render_system_info():
    """Render system information panel."""
    
    agent = st.session_state.conversational_agent
    
    if agent:
        with st.expander("🔍 System Information"):
            try:
                status = agent.get_system_status()
                sources = agent.retriever.get_available_sources()
                
                st.json({
                    "system_status": status.get('status', 'unknown'),
                    "capabilities": status.get('capabilities', {}),
                    "data_sources": {
                        "documents": sources.get('vector_sources', {}).get('count', 0),
                        "data_tables": sources.get('data_sources', {}).get('count', 0),
                        "total_chunks": sources.get('vector_sources', {}).get('total_chunks', 0)
                    },
                    "conversation_history": len(st.session_state.get('chat_history', [])),
                    "openai_configured": status.get('openai_configured', False)
                })
                
            except Exception as e:
                st.error(f"Error getting system info: {str(e)}")
    else:
        st.warning("System information not available - agent not initialized")

def render_help_section():
    """Render help and documentation section."""
    
    with st.expander("❓ Help & Documentation"):
        st.markdown("""
        ### How to Use the CEO RAG Chatbot
        
        **Email Management:**
        - Click on emails in the left panel to read and analyze them
        - Use the right panel to draft new emails
        - Ask the chatbot to "analyze this email" or "draft an email to..."
        
        **Data Analysis:**
        - Ask questions about your business data
        - Examples: "What were our sales last quarter?", "Show revenue by region"
        - The system will generate SQL queries automatically
        
        **Document Search:**
        - Search through uploaded PDFs and documents
        - Examples: "Find information about marketing strategy"
        - Uses AI-powered semantic search
        
        **General Chat:**
        - Ask about system capabilities
        - Get help with specific tasks
        - General business questions and conversations
        
        **Tips:**
        - Be specific in your queries for better results
        - Use natural language - no need for technical syntax
        - Check the sidebar for system status and settings
        """)

def render_quick_actions():
    """Render quick action buttons."""
    
    st.markdown("### ⚡ Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📊 System Status", help="Get current system status"):
            query = "What is the current system status?"
            add_chat_message("user", query)
            result = process_chat_query(query)
            
            if result["success"]:
                response = result["response"]
                add_chat_message("assistant", response.response_text)
            st.rerun()
    
    with col2:
        if st.button("📈 Data Summary", help="Get data sources summary"):
            query = "What data sources are available for analysis?"
            add_chat_message("user", query)
            result = process_chat_query(query)
            
            if result["success"]:
                response = result["response"]
                add_chat_message("assistant", response.response_text)
            st.rerun()
    
    with col3:
        if st.button("📧 Email Summary", help="Get email summary"):
            unread = len(get_unread_emails())
            urgent = len(get_emails_by_priority("URGENT"))
            high = len(get_emails_by_priority("HIGH"))
            
            summary = f"You have {unread} unread emails, including {urgent} urgent and {high} high priority messages."
            add_chat_message("assistant", summary)
            st.rerun()
    
    with col4:
        if st.button("💡 Show Examples", help="Show example queries"):
            st.session_state.show_examples = not st.session_state.get('show_examples', False)
            st.rerun()

# Main render function that brings everything together
def render_complete_chat_interface():
    """Render the complete chat interface with all components."""
    
    # Quick actions
    render_quick_actions()
    
    # Main chat interface (will be implemented in chat_interface.py)
    render_chat_interface()
    
    # Handle chat interaction
    handle_chat_interaction()
    
    # Show examples if toggled
    if st.session_state.get('show_examples', False):
        render_chat_examples()
    
    # Statistics and info sections
    col1, col2 = st.columns(2)
    
    with col1:
        render_chat_statistics()
    
    with col2:
        render_system_info()
    
    # Help section
    render_help_section()
