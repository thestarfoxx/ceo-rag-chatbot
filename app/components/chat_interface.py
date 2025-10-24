import streamlit as st
from datetime import datetime
from typing import Dict, Any, List
import sys
from pathlib import Path
import json

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.session_state import (
    get_or_create_agent,
    add_chat_message,
    get_conversation_context
)

def render_chat_interface():
    """Render the main chat interface in the center column."""
    
    # Chat header
    render_chat_header()
    
    # Chat messages container
    render_chat_messages()
    
    # Chat input
    render_chat_input()
    
    # Chat controls
    render_chat_controls()

def render_chat_header():
    """Render the chat header with status and info."""
    
    # Get agent status
    agent = st.session_state.conversational_agent
    is_online = agent is not None and st.session_state.get('agent_initialized', False)
    
    status_color = "#4caf50" if is_online else "#f44336"
    status_text = "Online" if is_online else "Offline"
    
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, #1f4e79 0%, #2980b9 100%); 
                color: white; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h3>💬 CEO RAG Assistant</h3>
                <p>Your intelligent business companion</p>
            </div>
            <div style="text-align: right;">
                <div style="display: flex; align-items: center; justify-content: flex-end;">
                    <div style="width: 10px; height: 10px; background-color: {status_color}; 
                               border-radius: 50%; margin-right: 8px;"></div>
                    <span>{status_text}</span>
                </div>
                <small>Ready to help with emails, data, and documents</small>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_chat_messages():
    """Render the chat messages in a scrollable container."""
    
    chat_history = st.session_state.get('chat_history', [])
    
    # Create a container for messages with custom height
    chat_container = st.container()
    
    with chat_container:
        if not chat_history:
            render_welcome_message()
        else:
            # Display chat messages
            for i, message in enumerate(chat_history):
                render_message(message, i)
    
    # Auto-scroll to bottom (using JavaScript)
    if chat_history:
        st.markdown("""
        <script>
        var chatContainer = window.parent.document.querySelector('[data-testid="stContainer"]');
        if (chatContainer) {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        </script>
        """, unsafe_allow_html=True)

def render_welcome_message():
    """Render welcome message when chat is empty."""
    
    st.markdown("""
    <div style="text-align: center; padding: 2rem; color: #666;">
        <h4>👋 Welcome to your CEO RAG Assistant!</h4>
        <p>I can help you with:</p>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1rem 0;">
            <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px;">
                <strong>📧 Email Management</strong><br>
                Draft, analyze, and manage emails
            </div>
            <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px;">
                <strong>📊 Data Analysis</strong><br>
                Query business data and generate reports
            </div>
            <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px;">
                <strong>📄 Document Search</strong><br>
                Find information in your documents
            </div>
            <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px;">
                <strong>💬 General Chat</strong><br>
                Ask questions and get assistance
            </div>
        </div>
        <p><strong>Try asking:</strong> "What were our sales last quarter?" or "Draft an email to John about the meeting"</p>
    </div>
    """, unsafe_allow_html=True)

def render_message(message: Dict[str, Any], index: int):
    """Render a single chat message."""
    
    role = message.get('role', 'user')
    content = message.get('content', '')
    metadata = message.get('metadata', {})
    timestamp = message.get('timestamp', '')
    
    # Message styling based on role
    if role == 'user':
        render_user_message(content, timestamp, index)
    else:
        render_assistant_message(content, metadata, timestamp, index)

def render_user_message(content: str, timestamp: str, index: int):
    """Render a user message."""
    
    st.markdown(f"""
    <div style="display: flex; justify-content: flex-end; margin: 1rem 0;">
        <div style="background: #e3f2fd; padding: 1rem; border-radius: 15px 15px 5px 15px; 
                   max-width: 80%; border-left: 4px solid #2196f3;">
            <div style="margin-bottom: 0.5rem;">
                <strong>You</strong>
                <span style="color: #666; font-size: 0.8rem; margin-left: 1rem;">{timestamp}</span>
            </div>
            <div>{content}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_assistant_message(content: str, metadata: Dict[str, Any], timestamp: str, index: int):
    """Render an assistant message with metadata."""
    
    # Determine message type and styling
    action_taken = metadata.get('action_taken', 'general_chat')
    success = metadata.get('success', True)
    processing_time = metadata.get('processing_time', 0)
    
    # Action icons
    action_icons = {
        'EMAIL_DRAFT': '📝',
        'EMAIL_READ': '📊',
        'SQL_QUERY': '🗄️',
        'DOCUMENT_SEARCH': '📄',
        'GENERAL_CHAT': '💬'
    }
    
    icon = action_icons.get(action_taken, '🤖')
    border_color = "#4caf50" if success else "#f44336"
    
    st.markdown(f"""
    <div style="display: flex; justify-content: flex-start; margin: 1rem 0;">
        <div style="background: #f5f5f5; padding: 1rem; border-radius: 15px 15px 15px 5px; 
                   max-width: 80%; border-left: 4px solid {border_color};">
            <div style="margin-bottom: 0.5rem; display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <strong>{icon} Assistant</strong>
                    <span style="color: #666; font-size: 0.8rem; margin-left: 1rem;">{timestamp}</span>
                </div>
                <div style="font-size: 0.7rem; color: #888;">
                    {processing_time:.2f}s
                </div>
            </div>
            <div style="white-space: pre-wrap;">{content}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show additional metadata if available
    if metadata and st.session_state.get('show_metadata', False):
        render_message_metadata(metadata, index)

def render_message_metadata(metadata: Dict[str, Any], index: int):
    """Render expandable metadata for a message."""
    
    with st.expander(f"📋 Message Details #{index + 1}", expanded=False):
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Action Taken:**", metadata.get('action_taken', 'Unknown'))
            st.write("**Success:**", "✅" if metadata.get('success', True) else "❌")
            st.write("**Processing Time:**", f"{metadata.get('processing_time', 0):.3f}s")
        
        with col2:
            if 'intent_confidence' in metadata:
                st.write("**Intent Confidence:**", f"{metadata['intent_confidence']:.3f}")
            
            if 'data' in metadata and metadata['data']:
                st.write("**Has Data:**", "Yes")
            
            if 'error' in metadata:
                st.write("**Error:**", metadata.get('error', False))
        
        # Show additional data if available
        if metadata.get('data'):
            with st.expander("📊 Response Data"):
                try:
                    st.json(metadata['data'])
                except:
                    st.text(str(metadata['data']))

def render_chat_input():
    """Render the chat input interface."""
    
    # Create input area
    user_input = st.chat_input(
        placeholder="Ask me anything about your business data, draft emails, or general questions...",
        key="main_chat_input"
    )
    
    # Process input
    if user_input:
        process_user_input(user_input)

def process_user_input(user_input: str):
    """Process user input and generate response."""
    
    if not user_input.strip():
        return
    
    # Add user message to chat
    current_time = datetime.now().strftime("%H:%M:%S")
    st.session_state.current_time = current_time
    
    add_chat_message("user", user_input)
    
    # Get agent
    agent = get_or_create_agent()
    
    if not agent:
        error_msg = "❌ Assistant is not available. Please check your configuration and try again."
        add_chat_message("assistant", error_msg, {"success": False, "error": True})
        st.rerun()
        return
    
    try:
        # Show processing indicator
        with st.spinner("🤔 Thinking..."):
            # Get conversation context
            context = get_conversation_context()
            
            # Process the query
            response = agent.process_query(user_input, context)
        
        # Add assistant response
        response_metadata = {
            "action_taken": response.action_taken.value,
            "processing_time": response.metadata.get('processing_time', 0),
            "intent_confidence": response.metadata.get('intent_analysis', {}).get('confidence', 0),
            "success": response.success,
            "data": response.data
        }
        
        if not response.success:
            response_metadata["error"] = response.error_message
        
        add_chat_message("assistant", response.response_text, response_metadata)
        
        # Handle specific response actions
        handle_response_actions(response)
        
        # Show success/error notification
        if response.success:
            show_success_notification(response)
        else:
            st.error(f"❌ Error: {response.error_message or 'Unknown error occurred'}")
    
    except Exception as e:
        error_msg = f"❌ An error occurred while processing your request: {str(e)}"
        add_chat_message("assistant", error_msg, {"success": False, "error": True})
        st.error(error_msg)
    
    # Rerun to update the interface
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
    
    # Handle email draft responses
    if response.action_taken == QueryType.EMAIL_DRAFT and response.success and response.data:
        # Update email drafter with the generated draft
        st.session_state.draft_recipient = response.data.get('recipient', '')
        st.session_state.draft_subject = response.data.get('subject', '')
        st.session_state.draft_content = response.data.get('full_email', response.data.get('body', ''))
        
        # Save to session state
        from app.utils.session_state import save_drafted_email
        save_drafted_email(response.data)
    
    # Handle email analysis responses
    elif response.action_taken == QueryType.EMAIL_READ and response.success and response.data:
        # Store analysis results for display
        st.session_state.last_email_analysis = response.data
    
    # Handle SQL query responses
    elif response.action_taken == QueryType.SQL_QUERY and response.success and response.data:
        # Store query results for potential export
        st.session_state.last_sql_results = response.data
    
    # Handle document search responses
    elif response.action_taken == QueryType.DOCUMENT_SEARCH and response.success and response.data:
        # Store search results
        st.session_state.last_search_results = response.data

def show_success_notification(response):
    """Show success notification based on response type."""
    
    try:
        from src.llm.conversation import QueryType
    except ImportError:
        from conversation import QueryType
    
    action = response.action_taken
    
    if action == QueryType.EMAIL_DRAFT:
        st.toast("📧 Email draft created! Check the drafter panel.", icon="✅")
    elif action == QueryType.EMAIL_READ:
        st.toast("📊 Email analysis complete!", icon="✅")
    elif action == QueryType.SQL_QUERY:
        count = len(response.data.get('results', [])) if response.data else 0
        st.toast(f"📊 Found {count} data results!", icon="✅")
    elif action == QueryType.DOCUMENT_SEARCH:
        count = len(response.data.get('chunks', [])) if response.data else 0
        st.toast(f"📄 Found {count} relevant documents!", icon="✅")
    else:
        st.toast("✅ Request processed successfully!", icon="✅")

def render_chat_controls():
    """Render chat control buttons and options."""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🗑️ Clear Chat", help="Clear all chat messages"):
            from app.utils.session_state import clear_chat_history
            clear_chat_history()
            st.success("Chat cleared!")
            st.rerun()
    
    with col2:
        if st.button("📊 Chat Stats", help="Show chat statistics"):
            st.session_state.show_chat_stats = not st.session_state.get('show_chat_stats', False)
            st.rerun()
    
    with col3:
        if st.button("⚙️ Debug Mode", help="Toggle debug information"):
            st.session_state.show_metadata = not st.session_state.get('show_metadata', False)
            status = "enabled" if st.session_state.show_metadata else "disabled"
            st.info(f"Debug mode {status}")
            st.rerun()
    
    with col4:
        if st.button("💾 Export Chat", help="Export chat history"):
            export_chat_history()

def export_chat_history():
    """Export chat history as JSON."""
    
    chat_history = st.session_state.get('chat_history', [])
    
    if not chat_history:
        st.warning("No chat history to export")
        return
    
    # Prepare export data
    export_data = {
        "export_timestamp": datetime.now().isoformat(),
        "total_messages": len(chat_history),
        "chat_history": chat_history
    }
    
    # Convert to JSON
    json_str = json.dumps(export_data, indent=2, default=str)
    
    # Offer download
    st.download_button(
        label="📥 Download Chat History",
        data=json_str,
        file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        help="Download your chat history as JSON file"
    )

def render_chat_statistics():
    """Render chat statistics if enabled."""
    
    if not st.session_state.get('show_chat_stats', False):
        return
    
    chat_history = st.session_state.get('chat_history', [])
    
    if not chat_history:
        st.info("No chat statistics available")
        return
    
    with st.expander("📈 Chat Statistics", expanded=True):
        
        # Basic metrics
        user_messages = [msg for msg in chat_history if msg['role'] == 'user']
        assistant_messages = [msg for msg in chat_history if msg['role'] == 'assistant']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Messages", len(chat_history))
        
        with col2:
            st.metric("User Queries", len(user_messages))
        
        with col3:
            successful_responses = sum(1 for msg in assistant_messages 
                                     if msg.get('metadata', {}).get('success', True))
            success_rate = (successful_responses / max(len(assistant_messages), 1)) * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")
        
        with col4:
            processing_times = [
                msg.get('metadata', {}).get('processing_time', 0)
                for msg in assistant_messages
                if msg.get('metadata', {}).get('processing_time')
            ]
            avg_time = sum(processing_times) / max(len(processing_times), 1)
            st.metric("Avg Response", f"{avg_time:.2f}s")
        
        # Intent distribution
        intents = [msg.get('metadata', {}).get('action_taken', 'unknown') 
                  for msg in assistant_messages 
                  if msg.get('metadata', {}).get('action_taken')]
        
        if intents:
            st.write("**Query Types:**")
            intent_counts = {}
            for intent in intents:
                intent_counts[intent] = intent_counts.get(intent, 0) + 1
            
            for intent, count in sorted(intent_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / len(intents)) * 100
                st.write(f"• {intent.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")

def render_typing_indicator():
    """Render typing indicator animation."""
    
    if st.session_state.get('processing', False):
        st.markdown("""
        <div style="display: flex; align-items: center; color: #666; font-style: italic; margin: 1rem 0;">
            <div style="margin-right: 1rem;">🤖 Assistant is thinking</div>
            <div class="typing-animation">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
        
def render_typing_indicator():
    """Render typing indicator animation."""
    
    if st.session_state.get('processing', False):
        st.markdown("""
        <div style="display: flex; align-items: center; color: #666; font-style: italic; margin: 1rem 0;">
            <div style="margin-right: 1rem;">🤖 Assistant is thinking</div>
            <div class="typing-animation">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
        
        <style>
        .typing-animation {
            display: flex;
            gap: 4px;
        }
        
        .typing-animation span {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: #2196f3;
            animation: typing 1.4s infinite ease-in-out;
        }
        
        .typing-animation span:nth-child(1) { animation-delay: 0s; }
        .typing-animation span:nth-child(2) { animation-delay: 0.2s; }
        .typing-animation span:nth-child(3) { animation-delay: 0.4s; }
        
        @keyframes typing {
            0%, 80%, 100% { transform: scale(0.8); opacity: 0.5; }
            40% { transform: scale(1); opacity: 1; }
        }
        </style>
        """, unsafe_allow_html=True)

def render_quick_suggestions():
    """Render quick suggestion buttons for common queries."""
    
    if len(st.session_state.get('chat_history', [])) == 0:
        st.markdown("### 💡 Quick Suggestions")
        
        suggestions = [
            "📊 What is the current system status?",
            "📧 Show me my unread emails",
            "💰 What were our sales last quarter?",
            "📄 Search for budget information",
            "✍️ Help me draft an email"
        ]
        
        cols = st.columns(len(suggestions))
        
        for i, suggestion in enumerate(suggestions):
            with cols[i]:
                if st.button(suggestion, key=f"suggestion_{i}", help="Click to send this query"):
                    # Process the suggestion as if user typed it
                    process_user_input(suggestion.split(" ", 1)[1])  # Remove emoji

def render_context_panel():
    """Render context panel showing current conversation state."""
    
    if st.session_state.get('show_context', False):
        with st.expander("🧠 Conversation Context", expanded=False):
            context = get_conversation_context()
            
            if context:
                st.text_area(
                    "Recent Conversation",
                    value=context,
                    height=150,
                    disabled=True
                )
            else:
                st.info("No conversation context available")
            
            # Show current state
            st.write("**Current State:**")
            selected_email = st.session_state.get('selected_email')
            if selected_email:
                st.write(f"📧 Selected Email: {selected_email.get('subject', 'No subject')}")
            
            drafted_email = st.session_state.get('drafted_email')
            if drafted_email:
                st.write(f"✍️ Current Draft: {drafted_email.get('subject', 'No subject')}")
            
            agent = st.session_state.get('conversational_agent')
            if agent:
                st.write("🤖 Agent Status: Ready")
            else:
                st.write("🤖 Agent Status: Not Available")

def render_help_overlay():
    """Render help overlay with keyboard shortcuts and tips."""
    
    if st.session_state.get('show_help', False):
        st.markdown("""
        <div style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; 
                   background: rgba(0,0,0,0.5); z-index: 9999; display: flex; 
                   justify-content: center; align-items: center;">
            <div style="background: white; padding: 2rem; border-radius: 10px; 
                       max-width: 600px; max-height: 80vh; overflow-y: auto;">
                <h3>❓ Help & Shortcuts</h3>
                
                <h4>💬 Chat Commands</h4>
                <ul>
                    <li><strong>/clear</strong> - Clear chat history</li>
                    <li><strong>/status</strong> - Show system status</li>
                    <li><strong>/help</strong> - Show this help</li>
                </ul>
                
                <h4>⌨️ Keyboard Shortcuts</h4>
                <ul>
                    <li><strong>Enter</strong> - Send message</li>
                    <li><strong>Shift + Enter</strong> - New line</li>
                    <li><strong>Ctrl + /</strong> - Toggle help</li>
                </ul>
                
                <h4>📧 Email Features</h4>
                <ul>
                    <li>Click emails in left panel to view/analyze</li>
                    <li>Use right panel to draft new emails</li>
                    <li>Ask "analyze this email" for AI insights</li>
                    <li>Say "draft email to [person]" for AI assistance</li>
                </ul>
                
                <h4>📊 Data Queries</h4>
                <ul>
                    <li>Ask natural language questions about your data</li>
                    <li>Examples: "sales by region", "top performers"</li>
                    <li>System automatically generates SQL queries</li>
                </ul>
                
                <button onclick="closeHelp()" style="margin-top: 1rem; padding: 0.5rem 1rem; 
                       background: #2196f3; color: white; border: none; border-radius: 5px;">
                    Close Help
                </button>
            </div>
        </div>
        
        <script>
        function closeHelp() {
            // This would need to be handled by Streamlit state management
            // For now, user can click outside or use button
        }
        </script>
        """, unsafe_allow_html=True)

def handle_chat_commands(user_input: str) -> bool:
    """Handle special chat commands. Returns True if command was handled."""
    
    if not user_input.startswith('/'):
        return False
    
    command = user_input.lower().strip()
    
    if command == '/clear':
        from app.utils.session_state import clear_chat_history
        clear_chat_history()
        st.success("Chat history cleared!")
        st.rerun()
        return True
    
    elif command == '/status':
        agent = get_or_create_agent()
        if agent:
            status = agent.get_system_status()
            status_msg = f"System Status: {status.get('status', 'unknown').title()}\n"
            status_msg += f"Available capabilities: {', '.join(cap for cap, available in status.get('capabilities', {}).items() if available)}"
            add_chat_message("assistant", status_msg, {"system_command": True})
        else:
            add_chat_message("assistant", "System Status: Agent not available", {"system_command": True})
        st.rerun()
        return True
    
    elif command == '/help':
        help_msg = """Available Commands:
• /clear - Clear chat history
• /status - Show system status  
• /help - Show this help message

You can also ask me:
• "Draft an email to [person] about [topic]"
• "Analyze this email: [email content]"
• "What were our [metric] last [period]?"
• "Search for information about [topic]"
• "What can you help me with?"
"""
        add_chat_message("assistant", help_msg, {"system_command": True})
        st.rerun()
        return True
    
    else:
        add_chat_message("assistant", f"Unknown command: {command}. Type /help for available commands.", 
                        {"system_command": True, "error": True})
        st.rerun()
        return True

def render_message_actions(message: Dict[str, Any], index: int):
    """Render action buttons for individual messages."""
    
    if message.get('role') == 'assistant':
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("👍", key=f"like_{index}", help="Mark as helpful"):
                st.session_state[f"message_liked_{index}"] = True
                st.success("Feedback recorded!")
        
        with col2:
            if st.button("👎", key=f"dislike_{index}", help="Mark as not helpful"):
                st.session_state[f"message_disliked_{index}"] = True
                st.error("Feedback recorded!")
        
        with col3:
            if st.button("📋", key=f"copy_{index}", help="Copy message"):
                # In a real app, this would copy to clipboard
                st.info("Message copied to clipboard!")
        
        with col4:
            if st.button("🔄", key=f"retry_{index}", help="Retry this query"):
                # Find the corresponding user message and retry
                if index > 0 and st.session_state.chat_history[index-1]['role'] == 'user':
                    user_query = st.session_state.chat_history[index-1]['content']
                    process_user_input(user_query)

def render_chat_footer():
    """Render chat footer with additional information."""
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        chat_count = len([msg for msg in st.session_state.get('chat_history', []) if msg['role'] == 'user'])
        st.metric("Queries Today", chat_count)
    
    with col2:
        agent = st.session_state.get('conversational_agent')
        if agent:
            sources = agent.retriever.get_available_sources()
            total_sources = sources.get('vector_sources', {}).get('count', 0) + sources.get('data_sources', {}).get('count', 0)
            st.metric("Data Sources", total_sources)
        else:
            st.metric("Data Sources", 0)
    
    with col3:
        uptime = "Connected" if st.session_state.get('agent_initialized', False) else "Disconnected"
        st.metric("Status", uptime)

# Main render function that brings everything together
def render_complete_chat_interface():
    """Render the complete chat interface with all components."""
    
    # Chat header
    render_chat_header()
    
    # Quick suggestions (only when chat is empty)
    render_quick_suggestions()
    
    # Typing indicator
    render_typing_indicator()
    
    # Main chat messages
    render_chat_messages()
    
    # Chat input
    render_chat_input()
    
    # Chat controls
    render_chat_controls()
    
    # Statistics (if enabled)
    render_chat_statistics()
    
    # Context panel (if enabled)
    render_context_panel()
    
    # Help overlay (if enabled)
    render_help_overlay()
    
    # Chat footer
    render_chat_footer()

# Utility functions for chat management
def format_response_for_display(response_text: str, action_taken: str) -> str:
    """Format response text for better display."""
    
    # Add action-specific formatting
    if action_taken == 'SQL_QUERY':
        # Format SQL responses with better structure
        if 'SELECT' in response_text.upper():
            # Highlight SQL keywords
            response_text = response_text.replace('SELECT', '**SELECT**')
            response_text = response_text.replace('FROM', '**FROM**')
            response_text = response_text.replace('WHERE', '**WHERE**')
    
    elif action_taken == 'EMAIL_DRAFT':
        # Format email drafts with clear sections
        if 'Subject:' in response_text:
            response_text = response_text.replace('Subject:', '**Subject:**')
        if 'To:' in response_text:
            response_text = response_text.replace('To:', '**To:**')
    
    return response_text

def calculate_message_sentiment(content: str) -> str:
    """Calculate simple sentiment of message (for future use)."""
    
    positive_words = ['good', 'great', 'excellent', 'perfect', 'amazing', 'wonderful', 'fantastic']
    negative_words = ['bad', 'terrible', 'awful', 'horrible', 'wrong', 'error', 'failed']
    
    content_lower = content.lower()
    
    positive_count = sum(1 for word in positive_words if word in content_lower)
    negative_count = sum(1 for word in negative_words if word in content_lower)
    
    if positive_count > negative_count:
        return 'positive'
    elif negative_count > positive_count:
        return 'negative'
    else:
        return 'neutral'

def get_chat_analytics() -> Dict[str, Any]:
    """Get analytics data for the current chat session."""
    
    chat_history = st.session_state.get('chat_history', [])
    
    if not chat_history:
        return {}
    
    user_messages = [msg for msg in chat_history if msg['role'] == 'user']
    assistant_messages = [msg for msg in chat_history if msg['role'] == 'assistant']
    
    # Calculate metrics
    total_messages = len(chat_history)
    total_user_queries = len(user_messages)
    total_responses = len(assistant_messages)
    
    # Success rate
    successful_responses = sum(1 for msg in assistant_messages 
                             if msg.get('metadata', {}).get('success', True))
    success_rate = (successful_responses / max(total_responses, 1)) * 100
    
    # Average response time
    processing_times = [
        msg.get('metadata', {}).get('processing_time', 0)
        for msg in assistant_messages
        if msg.get('metadata', {}).get('processing_time')
    ]
    avg_response_time = sum(processing_times) / max(len(processing_times), 1)
    
    # Intent distribution
    intents = [msg.get('metadata', {}).get('action_taken', 'unknown') 
              for msg in assistant_messages 
              if msg.get('metadata', {}).get('action_taken')]
    
    intent_distribution = {}
    for intent in intents:
        intent_distribution[intent] = intent_distribution.get(intent, 0) + 1
    
    return {
        'total_messages': total_messages,
        'user_queries': total_user_queries,
        'assistant_responses': total_responses,
        'success_rate': success_rate,
        'avg_response_time': avg_response_time,
        'intent_distribution': intent_distribution,
        'most_common_intent': max(intent_distribution.items(), key=lambda x: x[1])[0] if intent_distribution else None
    }
