import streamlit as st
from typing import Dict, Any, List
import sys
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

try:
    from conversation import create_conversational_agent
except ImportError:
    # Fallback if module structure is different
    try:
        from src.llm.conversation import create_conversational_agent
    except ImportError:
        # If neither works, we'll handle it in the function
        create_conversational_agent = None

def initialize_session_state():
    """Initialize all session state variables."""
    
    # Chat-related state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'current_query' not in st.session_state:
        st.session_state.current_query = ""
    
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    
    # Email-related state
    if 'emails' not in st.session_state:
        st.session_state.emails = []
    
    if 'selected_email' not in st.session_state:
        st.session_state.selected_email = None
    
    if 'draft_recipient' not in st.session_state:
        st.session_state.draft_recipient = ""
    
    if 'draft_subject' not in st.session_state:
        st.session_state.draft_subject = ""
    
    if 'draft_content' not in st.session_state:
        st.session_state.draft_content = ""
    
    if 'draft_tone' not in st.session_state:
        st.session_state.draft_tone = "professional"
    
    if 'drafted_email' not in st.session_state:
        st.session_state.drafted_email = None
    
    # System state
    if 'agent_initialized' not in st.session_state:
        st.session_state.agent_initialized = False
    
    if 'conversational_agent' not in st.session_state:
        st.session_state.conversational_agent = None
    
    if 'system_status' not in st.session_state:
        st.session_state.system_status = {}
    
    if 'available_sources' not in st.session_state:
        st.session_state.available_sources = {}
    
    # Settings
    if 'show_advanced_options' not in st.session_state:
        st.session_state.show_advanced_options = False
    
    if 'email_analysis_enabled' not in st.session_state:
        st.session_state.email_analysis_enabled = True
    
    if 'auto_draft_enabled' not in st.session_state:
        st.session_state.auto_draft_enabled = True

def get_or_create_agent():
    """Get existing agent or create new one."""
    try:
        if not st.session_state.agent_initialized or st.session_state.conversational_agent is None:
            with st.spinner("Initializing Conversational Agent..."):
                agent = create_conversational_agent()
                st.session_state.conversational_agent = agent
                st.session_state.agent_initialized = True
                
                # Get system status
                st.session_state.system_status = agent.get_system_status()
                st.session_state.available_sources = agent.retriever.get_available_sources()
                
        return st.session_state.conversational_agent
    except Exception as e:
        st.error(f"Failed to initialize conversational agent: {str(e)}")
        st.info("Please check your configuration and try again.")
        return None

def add_chat_message(role: str, content: str, metadata: Dict[str, Any] = None):
    """Add a message to chat history."""
    message = {
        "role": role,
        "content": content,
        "metadata": metadata or {},
        "timestamp": st.session_state.get('current_time', '')
    }
    st.session_state.chat_history.append(message)
    
    # Keep only last 20 messages for performance
    if len(st.session_state.chat_history) > 20:
        st.session_state.chat_history = st.session_state.chat_history[-20:]

def clear_chat_history():
    """Clear all chat messages."""
    st.session_state.chat_history = []
    if st.session_state.conversational_agent:
        st.session_state.conversational_agent.clear_conversation_history()

def add_sample_emails():
    """Add sample emails for demonstration."""
    sample_emails = [
        {
            "id": 1,
            "subject": "Q4 Financial Results - Board Review Required",
            "sender": "CFO Sarah Johnson <sarah.johnson@company.com>",
            "content": "Dear CEO,\n\nI've completed the Q4 financial analysis and we need to schedule a board review. Key highlights:\n\n- Revenue increased 15% YoY\n- Operating margin improved to 22%\n- Cash flow exceeded projections by $2.3M\n\nPlease review the attached report and let me know your availability for the board presentation next week.\n\nBest regards,\nSarah",
            "priority": "HIGH",
            "timestamp": "2024-01-15 09:30:00",
            "read": False,
            "category": "financial"
        },
        {
            "id": 2,
            "subject": "Marketing Campaign Performance Update",
            "sender": "Marketing Director Mike Chen <mike.chen@company.com>",
            "content": "Hi,\n\nQuick update on our Q1 marketing campaigns:\n\n- Digital campaigns ROI: 340%\n- Brand awareness up 28%\n- Lead generation exceeded targets by 45%\n\nDetailed report attached. The social media strategy is performing exceptionally well.\n\nThanks,\nMike",
            "priority": "MEDIUM", 
            "timestamp": "2024-01-15 11:15:00",
            "read": True,
            "category": "marketing"
        },
        {
            "id": 3,
            "subject": "URGENT: System Security Incident",
            "sender": "IT Security Team <security@company.com>",
            "content": "URGENT ATTENTION REQUIRED\n\nWe've detected unusual activity on our main servers. No data breach confirmed, but immediate action needed:\n\n1. All admin passwords reset\n2. Security audit initiated\n3. External consultants contacted\n\nPlease confirm receipt and availability for emergency meeting at 2 PM today.\n\nSecurity Team",
            "priority": "URGENT",
            "timestamp": "2024-01-15 13:45:00", 
            "read": False,
            "category": "security"
        },
        {
            "id": 4,
            "subject": "Partnership Opportunity - TechCorp",
            "sender": "Business Development <bd@company.com>",
            "content": "Hello,\n\nTechCorp has reached out regarding a potential strategic partnership. Initial discussions look promising:\n\n- Market expansion opportunities\n- Technology sharing benefits\n- Revenue potential: $50M+ annually\n\nThey're requesting a meeting with executive leadership. Would you be available next Friday?\n\nBest,\nBD Team",
            "priority": "HIGH",
            "timestamp": "2024-01-15 14:20:00",
            "read": False,
            "category": "business"
        },
        {
            "id": 5,
            "subject": "Team Building Event Planning",
            "sender": "HR Manager Lisa Park <lisa.park@company.com>",
            "content": "Hi CEO,\n\nWe're planning the annual team building event for Q2. Options include:\n\n- Off-site retreat (2 days)\n- Local activities (1 day)\n- Virtual team challenges\n\nBudget range: $50K-100K for 200 employees. Please let me know your preference and any specific requirements.\n\nRegards,\nLisa",
            "priority": "LOW",
            "timestamp": "2024-01-15 16:10:00",
            "read": True,
            "category": "hr"
        }
    ]
    
    if not st.session_state.emails:
        st.session_state.emails = sample_emails

def update_email_status(email_id: int, status_updates: Dict[str, Any]):
    """Update email status (read, priority, etc.)."""
    for email in st.session_state.emails:
        if email['id'] == email_id:
            email.update(status_updates)
            break

def get_emails_by_priority(priority: str = None) -> List[Dict[str, Any]]:
    """Get emails filtered by priority."""
    if priority:
        return [email for email in st.session_state.emails if email['priority'] == priority]
    return st.session_state.emails

def get_unread_emails() -> List[Dict[str, Any]]:
    """Get all unread emails."""
    return [email for email in st.session_state.emails if not email['read']]

def mark_email_read(email_id: int):
    """Mark an email as read."""
    update_email_status(email_id, {'read': True})

def get_email_by_id(email_id: int) -> Dict[str, Any]:
    """Get email by ID."""
    for email in st.session_state.emails:
        if email['id'] == email_id:
            return email
    return None

def reset_draft_form():
    """Reset email draft form."""
    st.session_state.draft_recipient = ""
    st.session_state.draft_subject = ""
    st.session_state.draft_content = ""
    st.session_state.draft_tone = "professional"
    st.session_state.drafted_email = None

def save_drafted_email(email_data: Dict[str, Any]):
    """Save drafted email to session state."""
    st.session_state.drafted_email = email_data

def get_conversation_context() -> str:
    """Get recent conversation context as string."""
    if st.session_state.conversational_agent:
        return st.session_state.conversational_agent.get_conversation_context()
    return ""
