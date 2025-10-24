import streamlit as st
from typing import Dict, Any
import sys
from pathlib import Path

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.session_state import (
    get_or_create_agent, 
    clear_chat_history, 
    add_sample_emails,
    get_unread_emails,
    get_emails_by_priority,
    reset_draft_form
)
from utils.config import load_config, display_config_status, display_config_details

def create_sidebar():
    """Create and populate the sidebar with system information and controls."""
    
    with st.sidebar:
        # Logo and title
        st.markdown("""
        <div style='text-align: center; padding: 1rem 0;'>
            <h2>🤖 Control Panel</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # System status section
        render_system_status()
        
        st.divider()
        
        # Quick actions
        render_quick_actions()
        
        st.divider()
        
        # Email management
        render_email_management()
        
        st.divider()
        
        # Data sources
        render_data_sources()
        
        st.divider()
        
        # Settings
        render_settings()

def render_system_status():
    """Render system status section."""
    st.subheader("🔍 System Status")
    
    # Get or create agent to check status
    agent = get_or_create_agent()
    
    if agent:
        try:
            status = agent.get_system_status()
            
            # Status indicator
            if status.get('status') == 'operational':
                st.success("🟢 System Online")
            else:
                st.error("🔴 System Offline")
            
            # Capabilities
            capabilities = status.get('capabilities', {})
            st.write("**Capabilities:**")
            
            cap_icons = {
                'email_drafting': '📝',
                'email_analysis': '📊',
                'sql_queries': '🗄️',
                'document_search': '📄',
                'general_chat': '💬'
            }
            
            for cap, available in capabilities.items():
                icon = cap_icons.get(cap, '⚙️')
                status_icon = "✅" if available else "❌"
                cap_name = cap.replace('_', ' ').title()
                st.write(f"{icon} {status_icon} {cap_name}")
            
            # Data sources summary
            data_sources = status.get('data_sources', {})
            st.write("**Data Sources:**")
            st.metric("Documents", data_sources.get('total_documents', 0))
            st.metric("Data Tables", data_sources.get('data_tables', 0))
            st.metric("Chunks", data_sources.get('total_chunks', 0))
            
        except Exception as e:
            st.error(f"Error getting system status: {str(e)}")
    else:
        st.error("🔴 Agent Not Initialized")
        st.info("Check configuration and restart")

def render_quick_actions():
    """Render quick action buttons."""
    st.subheader("⚡ Quick Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Refresh", help="Refresh system status"):
            # Clear cached agent to force refresh
            st.session_state.agent_initialized = False
            st.session_state.conversational_agent = None
            st.rerun()
        
        if st.button("🗑️ Clear Chat", help="Clear chat history"):
            clear_chat_history()
            st.success("Chat cleared!")
            st.rerun()
    
    with col2:
        if st.button("📧 Sample Emails", help="Load sample emails"):
            add_sample_emails()
            st.success("Sample emails loaded!")
            st.rerun()
        
        if st.button("📝 Reset Draft", help="Reset email draft form"):
            reset_draft_form()
            st.success("Draft form reset!")
            st.rerun()

def render_email_management():
    """Render email management section."""
    st.subheader("📧 Email Management")
    
    # Email statistics
    unread_emails = get_unread_emails()
    urgent_emails = get_emails_by_priority("URGENT")
    high_priority_emails = get_emails_by_priority("HIGH")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Unread", len(unread_emails))
        st.metric("High Priority", len(high_priority_emails))
    
    with col2:
        st.metric("Urgent", len(urgent_emails))
        st.metric("Total", len(st.session_state.emails))
    
    # Quick email actions
    if st.button("📥 Refresh Emails", help="Refresh email list"):
        st.success("Emails refreshed!")
        st.rerun()
    
    # Email filters
    st.write("**Filters:**")
    email_filter = st.selectbox(
        "Priority Filter",
        ["All", "URGENT", "HIGH", "MEDIUM", "LOW"],
        key="email_filter"
    )
    
    if email_filter != "All":
        filtered_emails = get_emails_by_priority(email_filter)
        st.write(f"Found {len(filtered_emails)} {email_filter} priority emails")

def render_data_sources():
    """Render data sources information."""
    st.subheader("📊 Data Sources")
    
    agent = st.session_state.conversational_agent
    if agent:
        try:
            sources = agent.retriever.get_available_sources()
            
            # Vector sources
            vector_sources = sources.get('vector_sources', {})
            st.write("**Document Sources:**")
            st.write(f"📄 {vector_sources.get('count', 0)} PDF documents")
            st.write(f"🔤 {vector_sources.get('total_chunks', 0)} text chunks")
            st.write(f"🔣 {vector_sources.get('total_tokens', 0)} tokens")
            
            # Data sources
            data_sources = sources.get('data_sources', {})
            st.write("**Structured Data:**")
            st.write(f"📊 {data_sources.get('count', 0)} data tables")
            st.write(f"📈 {data_sources.get('total_rows', 0)} total rows")
            st.write(f"📋 {data_sources.get('total_columns', 0)} total columns")
            
            # Show available tables in expander
            if vector_sources.get('tables') or data_sources.get('tables'):
                with st.expander("📋 View Available Sources"):
                    
                    if vector_sources.get('tables'):
                        st.write("**PDF Documents:**")
                        for table in vector_sources['tables'][:5]:  # Show first 5
                            st.write(f"• {table.get('file_name', 'Unknown')} ({table.get('total_chunks', 0)} chunks)")
                        
                        if len(vector_sources['tables']) > 5:
                            st.write(f"... and {len(vector_sources['tables']) - 5} more")
                    
                    if data_sources.get('tables'):
                        st.write("**Data Tables:**")
                        for table in data_sources['tables'][:5]:  # Show first 5
                            st.write(f"• {table.get('table_name', 'Unknown')} ({table.get('total_rows', 0)} rows)")
                        
                        if len(data_sources['tables']) > 5:
                            st.write(f"... and {len(data_sources['tables']) - 5} more")
            
        except Exception as e:
            st.error(f"Error loading data sources: {str(e)}")
    else:
        st.warning("Agent not available")

def render_settings():
    """Render settings section."""
    st.subheader("⚙️ Settings")
    
    # Configuration status
    config = load_config()
    is_valid, _ = display_config_status()
    
    # Advanced options toggle
    st.session_state.show_advanced_options = st.checkbox(
        "Show Advanced Options",
        value=st.session_state.get('show_advanced_options', False)
    )
    
    if st.session_state.show_advanced_options:
        
        # Search parameters
        st.write("**Search Parameters:**")
        
        # Vector search settings
        vector_config = config.get('vector_search', {})
        similarity_threshold = st.slider(
            "Similarity Threshold",
            min_value=0.0,
            max_value=0.25,  # Maximum allowed
            value=vector_config.get('similarity_threshold', 0.25),
            step=0.01,
            help="Lower values = more strict matching"
        )
        
        max_vector_results = st.number_input(
            "Max Vector Results",
            min_value=1,
            max_value=50,
            value=vector_config.get('max_results', 10),
            help="Maximum number of document chunks to retrieve"
        )
        
        max_sql_results = st.number_input(
            "Max SQL Results",
            min_value=1,
            max_value=200,
            value=config.get('sql_search', {}).get('max_results', 50),
            help="Maximum number of database rows to return"
        )
        
        # Update retriever settings if agent is available
        if st.button("💾 Update Settings"):
            agent = st.session_state.conversational_agent
            if agent:
                agent.retriever.update_search_parameters(
                    similarity_threshold=similarity_threshold,
                    max_vector_results=max_vector_results,
                    max_sql_results=max_sql_results
                )
                st.success("Settings updated!")
            else:
                st.error("Agent not available")
        
        # Email settings
        st.write("**Email Settings:**")
        st.session_state.email_analysis_enabled = st.checkbox(
            "Auto Email Analysis",
            value=st.session_state.get('email_analysis_enabled', True),
            help="Automatically analyze emails when selected"
        )
        
        st.session_state.auto_draft_enabled = st.checkbox(
            "Smart Draft Suggestions",
            value=st.session_state.get('auto_draft_enabled', True),
            help="Provide smart suggestions while drafting emails"
        )
        
        # Display detailed config
        display_config_details(config)
    
    # Debug information
    if config.get('app', {}).get('debug', False):
        with st.expander("🐛 Debug Information"):
            st.write("**Session State Keys:**")
            for key in sorted(st.session_state.keys()):
                if not key.startswith('_'):  # Hide private keys
                    value_type = type(st.session_state[key]).__name__
                    st.write(f"• {key}: {value_type}")
            
            st.write("**Agent Status:**")
            st.write(f"Initialized: {st.session_state.get('agent_initialized', False)}")
            st.write(f"Agent exists: {st.session_state.conversational_agent is not None}")
            
            if st.button("🔍 Show Full Session State"):
                st.json({k: str(v)[:100] + "..." if len(str(v)) > 100 else str(v) 
                        for k, v in st.session_state.items() 
                        if not k.startswith('_')})

def render_performance_metrics():
    """Render performance metrics (if available)."""
    st.subheader("📈 Performance")
    
    # Chat metrics
    chat_history = st.session_state.get('chat_history', [])
    if chat_history:
        successful_queries = sum(1 for msg in chat_history 
                               if msg.get('role') == 'assistant' and 
                               msg.get('metadata', {}).get('success', True))
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Queries", len([m for m in chat_history if m.get('role') == 'user']))
        with col2:
            success_rate = (successful_queries / max(len([m for m in chat_history if m.get('role') == 'assistant']), 1)) * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")
        
        # Average processing time
        processing_times = [
            msg.get('metadata', {}).get('processing_time', 0)
            for msg in chat_history
            if msg.get('role') == 'assistant' and 
            msg.get('metadata', {}).get('processing_time')
        ]
        
        if processing_times:
            avg_time = sum(processing_times) / len(processing_times)
            st.metric("Avg Response Time", f"{avg_time:.2f}s")
    
    # System metrics (if available)
    agent = st.session_state.conversational_agent
    if agent:
        try:
            # This would require additional tracking in the agent
            pass
        except:
            pass

def create_sidebar_footer():
    """Create sidebar footer with additional information."""
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8rem;'>
        <p>🚀 CEO RAG Chatbot v1.0</p>
        <p>💡 Powered by OpenAI & PostgreSQL</p>
    </div>
    """, unsafe_allow_html=True)
