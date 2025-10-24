import streamlit as st
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import os
import sys
from pathlib import Path

# Add current directory and src to Python path
current_dir = Path(__file__).parent.absolute()
src_dir = current_dir / 'src'

# Add paths
for path in [str(current_dir), str(src_dir)]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Configure Streamlit page
st.set_page_config(
    page_title="CEO RAG Workspace",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Try to import conversation module with better error handling
def try_import_conversation():
    """Try to import the conversation module from various locations."""
    
    # List of possible import paths to try
    import_paths = [
        # Direct module imports
        ('llm.conversation', 'from llm.conversation'),
        ('src.llm.conversation', 'from src.llm.conversation'),
        ('conversation', 'from conversation'),
        
        # File-based imports
        (str(src_dir / 'llm' / 'conversation.py'), 'from file path'),
    ]
    
    for import_path, description in import_paths:
        try:
            if import_path.endswith('.py'):
                # Import from file path
                import importlib.util
                spec = importlib.util.spec_from_file_location("conversation", import_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    st.sidebar.success(f"✅ Imported conversation module {description}")
                    return module
            else:
                # Import using module path
                module = __import__(import_path, fromlist=[''])
                st.sidebar.success(f"✅ Imported conversation module {description}")
                return module
                
        except (ImportError, FileNotFoundError, AttributeError) as e:
            st.sidebar.warning(f"❌ Failed {description}: {str(e)[:100]}")
            continue
    
    return None

# Try to import the conversation module
conversation_module = try_import_conversation()

if conversation_module is None:
    st.error("""
    ## ❌ Cannot Import Conversation Module
    
    **Quick Fix Options:**
    
    ### Option 1: Copy the file to project root
    ```bash
    cp src/llm/conversation.py ./conversation.py
    streamlit run workspace_app.py
    ```
    
    ### Option 2: Create an __init__.py file
    ```bash
    touch src/__init__.py
    touch src/llm/__init__.py
    streamlit run workspace_app.py
    ```
    
    ### Option 3: Run from src directory
    ```bash
    cd src
    streamlit run ../workspace_app.py
    ```
    
    **Debug Information:**
    - Current directory: `{}`
    - Looking for file: `{}`
    - File exists: `{}`
    - Python path: `{}`
    """.format(
        current_dir,
        src_dir / 'llm' / 'conversation.py',
        (src_dir / 'llm' / 'conversation.py').exists(),
        sys.path[:3]
    ))
    
    # Show directory contents for debugging
    st.subheader("📁 Directory Contents")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Project Root:**")
        try:
            for item in sorted(current_dir.iterdir()):
                icon = "📁" if item.is_dir() else "📄"
                st.write(f"{icon} {item.name}")
        except:
            st.write("Cannot read directory")
    
    with col2:
        st.write("**src/llm/ Contents:**")
        llm_dir = src_dir / 'llm'
        if llm_dir.exists():
            try:
                for item in sorted(llm_dir.iterdir()):
                    icon = "📁" if item.is_dir() else "📄"
                    st.write(f"{icon} {item.name}")
            except:
                st.write("Cannot read llm directory")
        else:
            st.write("src/llm/ directory not found")
    
    st.stop()

# Extract required functions from the conversation module
try:
    create_conversational_agent = getattr(conversation_module, 'create_conversational_agent')
    ConversationResponse = getattr(conversation_module, 'ConversationResponse')
    QueryType = getattr(conversation_module, 'QueryType')
    
    # Optional functions (may not exist)
    try:
        format_email_for_analysis = getattr(conversation_module, 'format_email_for_analysis')
        extract_email_from_text = getattr(conversation_module, 'extract_email_from_text')
    except AttributeError:
        # Create dummy functions if they don't exist
        def format_email_for_analysis(subject, sender, body, date=""):
            return f"Subject: {subject}\nFrom: {sender}\nDate: {date}\n\n{body}"
        
        def extract_email_from_text(text):
            lines = text.split('\n')
            return {
                'subject': next((line.replace('Subject:', '').strip() for line in lines if line.startswith('Subject:')), ''),
                'sender': next((line.replace('From:', '').strip() for line in lines if line.startswith('From:')), ''),
                'body': text
            }
    
    st.sidebar.success("✅ All conversation functions loaded successfully")
    
except AttributeError as e:
    st.error(f"""
    ## ❌ Missing Functions in Conversation Module
    
    **Error:** {e}
    
    **Required functions:**
    - `create_conversational_agent()`
    - `ConversationResponse` class
    - `QueryType` enum
    
    **Please check your conversation.py file contains these functions.**
    """)
    st.stop()

# Custom CSS for styling
st.markdown("""
<style>
    /* Main workspace styling */
    .main-workspace {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    /* Panel styling */
    .panel {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        height: 70vh;
        overflow-y: auto;
    }
    
    /* Chat messages */
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        background: #f8f9ff;
    }
    
    .user-message {
        background: #e3f2fd;
        border-left-color: #2196f3;
        margin-left: 2rem;
    }
    
    .assistant-message {
        background: #f3e5f5;
        border-left-color: #9c27b0;
        margin-right: 2rem;
    }
    
    /* Email styling */
    .email-item {
        background: #fff;
        border: 1px solid #e0e0e0;
        border-radius: 6px;
        padding: 1rem;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .email-item:hover {
        background: #f5f5f5;
        border-color: #667eea;
        transform: translateY(-1px);
    }
    
    .email-priority-high { border-left: 4px solid #f44336; }
    .email-priority-medium { border-left: 4px solid #ff9800; }
    .email-priority-low { border-left: 4px solid #4caf50; }
    
    /* Draft area styling */
    .draft-area {
        background: #fafafa;
        border: 2px dashed #ccc;
        border-radius: 8px;
        padding: 1.5rem;
        text-align: center;
        min-height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 5px;
    }
    
    .status-online { background-color: #4caf50; }
    .status-busy { background-color: #ff9800; }
    .status-offline { background-color: #f44336; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize Streamlit session state variables."""
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'emails' not in st.session_state:
        st.session_state.emails = [
            {
                'id': 1,
                'sender': 'john.smith@company.com',
                'subject': 'Q4 Financial Review Meeting',
                'preview': 'Hi, I wanted to schedule our quarterly financial review...',
                'priority': 'high',
                'time': '2 hours ago',
                'unread': True,
                'full_content': """Subject: Q4 Financial Review Meeting
From: john.smith@company.com
Date: Today, 2:00 PM

Hi,

I wanted to schedule our quarterly financial review meeting for next week. We need to discuss:

1. Revenue performance vs targets
2. Cost optimization initiatives 
3. Budget planning for next quarter
4. Market expansion opportunities

Please let me know your availability for Tuesday or Wednesday afternoon.

Best regards,
John Smith
CFO"""
            },
            {
                'id': 2,
                'sender': 'sarah.johnson@company.com',
                'subject': 'URGENT: System Maintenance Window',
                'preview': 'We need to schedule an emergency maintenance window...',
                'priority': 'high',
                'time': '1 hour ago',
                'unread': True,
                'full_content': """Subject: URGENT: System Maintenance Window
From: sarah.johnson@company.com
Date: Today, 3:00 PM

URGENT: We need to schedule an emergency maintenance window this weekend due to critical security updates.

Impact: All systems will be offline for approximately 4 hours
Proposed Time: Saturday 11 PM - Sunday 3 AM
Required Actions: Notify all departments, prepare backup procedures

Please approve ASAP so we can begin preparations.

Sarah Johnson
CTO"""
            }
        ]
    
    if 'drafted_emails' not in st.session_state:
        st.session_state.drafted_emails = []
    
    if 'agent_initialized' not in st.session_state:
        st.session_state.agent_initialized = False

def initialize_agent():
    """Initialize the conversational agent."""
    try:
        if not st.session_state.agent_initialized:
            with st.spinner("Initializing AI Assistant..."):
                st.session_state.agent = create_conversational_agent()
                st.session_state.agent_initialized = True
            st.success("AI Assistant initialized successfully!")
        return True
    except Exception as e:
        st.error(f"Failed to initialize AI Assistant: {str(e)}")
        st.info("Please check your OpenAI API key and database configuration.")
        return False

def display_header():
    """Display the workspace header."""
    st.markdown("""
    <div class="main-workspace">
        <h1 style="color: white; text-align: center; margin: 0;">
            🏢 CEO RAG Workspace
        </h1>
        <p style="color: white; text-align: center; opacity: 0.9; margin: 0.5rem 0 0 0;">
            Intelligent Assistant for Executive Decision Making
        </p>
    </div>
    """, unsafe_allow_html=True)

def display_email_panel():
    """Display the email panel on the left."""
    st.markdown("### 📧 Email Inbox")
    
    # Email filters
    col1, col2 = st.columns(2)
    with col1:
        filter_priority = st.selectbox("Priority", ["All", "High", "Medium", "Low"], key="email_filter")
    with col2:
        filter_status = st.selectbox("Status", ["All", "Unread", "Read"], key="email_status")
    
    # Filter emails
    filtered_emails = st.session_state.emails
    if filter_priority != "All":
        filtered_emails = [e for e in filtered_emails if e['priority'] == filter_priority.lower()]
    if filter_status == "Unread":
        filtered_emails = [e for e in filtered_emails if e['unread']]
    elif filter_status == "Read":
        filtered_emails = [e for e in filtered_emails if not e['unread']]
    
    # Display emails
    for email in filtered_emails:
        priority_class = f"email-priority-{email['priority']}"
        unread_indicator = "🔵" if email['unread'] else "⚪"
        
        with st.container():
            st.markdown(f"""
            <div class="email-item {priority_class}">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong>{unread_indicator} {email['sender']}</strong>
                        <div style="font-size: 0.9em; color: #666;">{email['time']}</div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 0.8em; color: #999;">{email['priority'].upper()}</div>
                    </div>
                </div>
                <div style="margin: 0.5rem 0;">
                    <strong>{email['subject']}</strong>
                </div>
                <div style="color: #666; font-size: 0.9em;">
                    {email['preview']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Action buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("📖 Read", key=f"read_{email['id']}"):
                    read_email(email)
            with col2:
                if st.button("📝 Reply", key=f"reply_{email['id']}"):
                    start_reply(email)
            with col3:
                if st.button("🗑️ Archive", key=f"archive_{email['id']}"):
                    archive_email(email['id'])

def read_email(email):
    """Process email reading through the conversational agent."""
    if st.session_state.agent:
        # Mark as read
        for e in st.session_state.emails:
            if e['id'] == email['id']:
                e['unread'] = False
                break
        
        # Send to chat
        query = f"Analyze this email: {email['full_content']}"
        process_chat_query(query)
    else:
        st.error("AI Assistant not initialized")

def start_reply(email):
    """Start drafting a reply to the email."""
    reply_context = f"Draft a reply to this email:\n\nOriginal Email:\n{email['full_content']}"
    
    # Send to chat
    if st.session_state.agent:
        process_chat_query(f"Draft a professional reply to the email from {email['sender']} about {email['subject']}")

def archive_email(email_id):
    """Archive an email."""
    st.session_state.emails = [e for e in st.session_state.emails if e['id'] != email_id]
    st.rerun()

def display_chat_panel():
    """Display the central chat panel."""
    st.markdown("### 🤖 AI Assistant")
    
    # Agent status
    if st.session_state.agent_initialized:
        st.markdown('<span class="status-indicator status-online"></span> **Online**', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-indicator status-offline"></span> **Offline**', unsafe_allow_html=True)
    
    # Chat history
    if st.session_state.chat_history:
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong><br>
                    {message['content']}
                    <div style="font-size: 0.8em; color: #666; margin-top: 0.5rem;">
                        {message['timestamp']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>AI Assistant ({message.get('intent', 'Chat')}):</strong><br>
                    {message['content']}
                    <div style="font-size: 0.8em; color: #666; margin-top: 0.5rem;">
                        {message['timestamp']} • {message.get('processing_time', 0):.2f}s
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align: center; color: #666; padding: 2rem;">
            👋 Welcome to your AI workspace! <br>
            I can help you with emails, data analysis, document search, and more.<br>
            Try asking me something like "Draft an email to the team" or "What are our Q4 results?"
        </div>
        """, unsafe_allow_html=True)
    
    # Chat input
    st.markdown("---")
    
    # Example queries
    with st.expander("💡 Example Queries"):
        examples = [
            "Draft an email to the board about quarterly results",
            "Analyze the email from John about the financial review",
            "What were our sales figures last quarter?",
            "Search for information about our marketing strategy",
            "Help me prioritize my emails"
        ]
        
        for example in examples:
            if st.button(f"💬 {example}", key=f"example_{hash(example)}"):
                process_chat_query(example)
    
    # Chat input form
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area(
            "Type your message...", 
            placeholder="Ask me to draft emails, analyze data, search documents, or just chat!",
            height=100,
            key="chat_input"
        )
        
        col1, col2 = st.columns([1, 1])
        with col1:
            submit_button = st.form_submit_button("Send 📤", use_container_width=True)
        with col2:
            clear_button = st.form_submit_button("Clear 🗑️", use_container_width=True)
        
        if submit_button and user_input.strip():
            process_chat_query(user_input.strip())
        
        if clear_button:
            st.session_state.chat_history = []
            st.rerun()

def process_chat_query(query: str):
    """Process a chat query through the conversational agent."""
    if not st.session_state.agent:
        st.error("AI Assistant not initialized")
        return
    
    # Add user message to history
    st.session_state.chat_history.append({
        'role': 'user',
        'content': query,
        'timestamp': datetime.now().strftime("%H:%M:%S")
    })
    
    try:
        # Process with agent
        with st.spinner("Processing..."):
            context = ""
            if len(st.session_state.chat_history) > 1:
                # Get recent context
                recent_messages = st.session_state.chat_history[-6:-1]
                context_parts = []
                for msg in recent_messages:
                    context_parts.append(f"{msg['role']}: {msg['content'][:100]}")
                context = "\n".join(context_parts)
            
            response = st.session_state.agent.process_query(query, context)
            
            # Add assistant response to history
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': response.response_text,
                'intent': response.action_taken.value,
                'timestamp': datetime.now().strftime("%H:%M:%S"),
                'processing_time': response.metadata.get('processing_time', 0) if response.metadata else 0,
                'success': response.success,
                'data': response.data
            })
            
            # Handle specific actions
            if response.action_taken == QueryType.EMAIL_DRAFT and response.data:
                handle_email_draft_response(response.data)
            
            st.rerun()
            
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        
        # Add error message to history
        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': f"I encountered an error: {str(e)}",
            'intent': 'Error',
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            'processing_time': 0,
            'success': False
        })

def handle_email_draft_response(email_data):
    """Handle email draft response by adding to draft area."""
    drafted_email = {
        'id': len(st.session_state.drafted_emails) + 1,
        'subject': email_data.get('subject', 'No Subject'),
        'recipient': email_data.get('recipient', ''),
        'content': email_data.get('full_email', email_data.get('body', '')),
        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'status': 'draft'
    }
    
    st.session_state.drafted_emails.append(drafted_email)

def display_draft_panel():
    """Display the email draft panel on the right."""
    st.markdown("### ✍️ Email Drafts")
    
    # Draft actions
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📝 New Draft", use_container_width=True):
            new_draft_modal()
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    
    # Show drafts
    if st.session_state.drafted_emails:
        for draft in reversed(st.session_state.drafted_emails):
            with st.container():
                st.markdown(f"""
                <div class="email-item">
                    <div style="display: flex; justify-content: between; align-items: center;">
                        <div>
                            <strong>📋 Draft #{draft['id']}</strong>
                            <div style="font-size: 0.9em; color: #666;">{draft['created_at']}</div>
                        </div>
                    </div>
                    <div style="margin: 0.5rem 0;">
                        <strong>To:</strong> {draft['recipient'] or 'Not specified'}<br>
                        <strong>Subject:</strong> {draft['subject']}
                    </div>
                    <div style="color: #666; font-size: 0.9em; max-height: 100px; overflow: hidden;">
                        {draft['content'][:200]}{'...' if len(draft['content']) > 200 else ''}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Draft actions
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    if st.button("👁️", key=f"view_draft_{draft['id']}", help="View full draft"):
                        show_draft_modal(draft)
                with col2:
                    if st.button("✏️", key=f"edit_draft_{draft['id']}", help="Edit draft"):
                        edit_draft_modal(draft)
                with col3:
                    if st.button("📧", key=f"send_draft_{draft['id']}", help="Send email"):
                        send_email(draft)
                with col4:
                    if st.button("🗑️", key=f"delete_draft_{draft['id']}", help="Delete draft"):
                        delete_draft(draft['id'])
    else:
        st.markdown("""
        <div class="draft-area">
            <h4>📭 No drafts yet</h4>
            <p>Ask the AI assistant to draft an email, or click "New Draft" to start writing.</p>
            <p><em>Try: "Draft an email to the team about the meeting"</em></p>
        </div>
        """, unsafe_allow_html=True)

def new_draft_modal():
    """Handle new draft creation."""
    query = "I'd like to draft a new email. Please help me with the recipient, subject, and content."
    process_chat_query(query)

def show_draft_modal(draft):
    """Show full draft in an expander."""
    with st.expander(f"📧 Draft #{draft['id']} - {draft['subject']}", expanded=True):
        st.text_input("To:", value=draft['recipient'], disabled=True)
        st.text_input("Subject:", value=draft['subject'], disabled=True)
        st.text_area("Content:", value=draft['content'], height=300, disabled=True)

def edit_draft_modal(draft):
    """Edit draft modal."""
    query = f"Please help me edit this email draft:\n\nTo: {draft['recipient']}\nSubject: {draft['subject']}\nContent: {draft['content'][:500]}..."
    process_chat_query(query)

def send_email(draft):
    """Simulate sending an email."""
    st.success(f"📧 Email sent to {draft['recipient']}!")
    # Remove from drafts
    st.session_state.drafted_emails = [d for d in st.session_state.drafted_emails if d['id'] != draft['id']]
    st.rerun()

def delete_draft(draft_id):
    """Delete a draft."""
    st.session_state.drafted_emails = [d for d in st.session_state.drafted_emails if d['id'] != draft_id]
    st.rerun()

def main():
    """Main application function."""
    # Initialize session state
    init_session_state()
    
    # Display header
    display_header()
    
    # Initialize agent
    if not st.session_state.agent_initialized:
        if not initialize_agent():
            st.warning("AI Assistant is offline. You can still use the interface, but AI features won't work.")
    
    # Main workspace layout
    col1, col2, col3 = st.columns([1, 2, 1])
    
    # Left panel - Emails
    with col1:
        with st.container():
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            display_email_panel()
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Center panel - Chat
    with col2:
        with st.container():
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            display_chat_panel()
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Right panel - Drafts
    with col3:
        with st.container():
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            display_draft_panel()
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Status bar
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        unread_count = sum(1 for email in st.session_state.emails if email['unread'])
        st.metric("Unread Emails", unread_count)
    
    with col2:
        st.metric("Draft Emails", len(st.session_state.drafted_emails))
    
    with col3:
        st.metric("Chat Messages", len(st.session_state.chat_history))

if __name__ == "__main__":
    main()
