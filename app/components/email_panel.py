import streamlit as st
from datetime import datetime
from typing import Dict, Any, List
import sys
from pathlib import Path

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.session_state import (
    get_unread_emails,
    get_emails_by_priority,
    mark_email_read,
    get_email_by_id,
    update_email_status,
    get_or_create_agent,
    add_chat_message
)

def render_email_panel():
    """Render the left email panel with email list and viewer."""
    
    st.markdown("""
    <div class="email-panel">
        <h3>📧 Email Center</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Email filters and controls
    render_email_filters()
    
    # Email list
    render_email_list()
    
    # Email viewer
    if st.session_state.selected_email:
        render_email_viewer()

def render_email_filters():
    """Render email filtering controls."""
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Priority filter
        priority_options = ["All", "URGENT", "HIGH", "MEDIUM", "LOW"]
        selected_priority = st.selectbox(
            "Priority",
            priority_options,
            key="email_priority_filter"
        )
    
    with col2:
        # Read status filter
        read_options = ["All", "Unread", "Read"]
        selected_read_status = st.selectbox(
            "Status", 
            read_options,
            key="email_read_filter"
        )
    
    # Update filtered emails in session state
    filtered_emails = get_filtered_emails(selected_priority, selected_read_status)
    st.session_state.filtered_emails = filtered_emails

def get_filtered_emails(priority_filter: str, read_filter: str) -> List[Dict[str, Any]]:
    """Get emails based on current filters."""
    
    emails = st.session_state.get('emails', [])
    
    # Apply priority filter
    if priority_filter != "All":
        emails = [email for email in emails if email.get('priority') == priority_filter]
    
    # Apply read status filter
    if read_filter == "Unread":
        emails = [email for email in emails if not email.get('read', False)]
    elif read_filter == "Read":
        emails = [email for email in emails if email.get('read', False)]
    
    # Sort by timestamp (newest first)
    emails.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    
    return emails

def render_email_list():
    """Render the list of emails."""
    
    filtered_emails = st.session_state.get('filtered_emails', st.session_state.get('emails', []))
    
    if not filtered_emails:
        st.info("No emails found matching current filters.")
        return
    
    st.write(f"**{len(filtered_emails)} emails**")
    
    # Container for email list with custom styling
    with st.container():
        for email in filtered_emails:
            render_email_item(email)

def render_email_item(email: Dict[str, Any]):
    """Render a single email item in the list."""
    
    email_id = email['id']
    is_selected = st.session_state.get('selected_email_id') == email_id
    is_read = email.get('read', False)
    priority = email.get('priority', 'MEDIUM')
    
    # Priority styling
    priority_class = f"priority-{priority.lower()}"
    
    # Read status styling
    font_weight = "normal" if is_read else "bold"
    background_color = "#e3f2fd" if is_selected else "#ffffff"
    
    # Email preview
    subject = email.get('subject', 'No Subject')[:50]
    sender = email.get('sender', 'Unknown Sender').split('<')[0].strip()[:30]
    content_preview = email.get('content', '')[:80] + "..." if len(email.get('content', '')) > 80 else email.get('content', '')
    timestamp = email.get('timestamp', '')[:16] if email.get('timestamp') else ''
    
    # Create clickable email item
    email_html = f"""
    <div class="email-item {priority_class}" style="
        background-color: {background_color}; 
        font-weight: {font_weight};
        margin-bottom: 0.5rem;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid var(--priority-color);
        cursor: pointer;
    ">
        <div class="email-subject" style="font-weight: bold; color: #1f4e79; margin-bottom: 0.3rem;">
            {subject}
        </div>
        <div class="email-sender" style="color: #666; font-size: 0.9rem; margin-bottom: 0.3rem;">
            {sender}
        </div>
        <div class="email-preview" style="color: #888; font-size: 0.85rem; margin-bottom: 0.3rem;">
            {content_preview}
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="color: #999; font-size: 0.8rem;">{timestamp}</span>
            <span style="
                background-color: {'#f44336' if priority == 'URGENT' else '#ff9800' if priority == 'HIGH' else '#4caf50' if priority == 'LOW' else '#2196f3'};
                color: white;
                padding: 0.2rem 0.5rem;
                border-radius: 12px;
                font-size: 0.7rem;
                font-weight: bold;
            ">{priority}</span>
        </div>
    </div>
    """
    
    # Display the email item
    st.markdown(email_html, unsafe_allow_html=True)
    
    # Create button for selection (invisible but functional)
    if st.button(f"Select Email {email_id}", key=f"select_email_{email_id}", 
                help=f"Click to view email: {subject}", label_visibility="collapsed"):
        st.session_state.selected_email_id = email_id
        st.session_state.selected_email = email
        
        # Mark as read when selected
        if not is_read:
            mark_email_read(email_id)
        
        st.rerun()

def render_email_viewer():
    """Render the email viewer for the selected email."""
    
    email = st.session_state.selected_email
    if not email:
        return
    
    st.markdown("---")
    st.markdown("### 📖 Email Viewer")
    
    # Email header
    with st.container():
        st.markdown(f"**Subject:** {email.get('subject', 'No Subject')}")
        st.markdown(f"**From:** {email.get('sender', 'Unknown Sender')}")
        st.markdown(f"**Date:** {email.get('timestamp', 'Unknown Date')}")
        st.markdown(f"**Priority:** {email.get('priority', 'MEDIUM')}")
        st.markdown(f"**Category:** {email.get('category', 'general').title()}")
    
    st.markdown("---")
    
    # Email content
    st.markdown("**Content:**")
    content = email.get('content', 'No content available')
    st.text_area(
        "Email Content",
        value=content,
        height=200,
        disabled=True,
        label_visibility="collapsed"
    )
    
    # Email actions
    render_email_actions(email)

def render_email_actions(email: Dict[str, Any]):
    """Render action buttons for the selected email."""
    
    st.markdown("### ⚡ Email Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Analyze Email", help="Analyze email importance and content"):
            analyze_email(email)
    
    with col2:
        if st.button("↩️ Reply", help="Draft a reply to this email"):
            draft_reply(email)
    
    with col3:
        if st.button("📝 Forward", help="Draft a forward for this email"):
            draft_forward(email)
    
    # Additional actions
    col4, col5, col6 = st.columns(3)
    
    with col4:
        if st.button("⭐ Mark Important", help="Mark as important"):
            update_email_status(email['id'], {'priority': 'HIGH'})
            st.success("Marked as important!")
            st.rerun()
    
    with col5:
        current_read_status = email.get('read', False)
        status_text = "Mark Unread" if current_read_status else "Mark Read"
        if st.button(f"👁️ {status_text}", help=f"{status_text.lower()} this email"):
            update_email_status(email['id'], {'read': not current_read_status})
            st.success(f"Email marked as {'unread' if current_read_status else 'read'}!")
            st.rerun()
    
    with col6:
        if st.button("🗑️ Archive", help="Archive this email"):
            # Remove from current list (simulate archiving)
            st.session_state.emails = [e for e in st.session_state.emails if e['id'] != email['id']]
            st.session_state.selected_email = None
            st.session_state.selected_email_id = None
            st.success("Email archived!")
            st.rerun()

def analyze_email(email: Dict[str, Any]):
    """Analyze the selected email using the conversational agent."""
    
    # Format email for analysis
    email_text = f"""Subject: {email.get('subject', '')}
From: {email.get('sender', '')}
Date: {email.get('timestamp', '')}

{email.get('content', '')}"""
    
    # Create analysis query
    query = f"Analyze this email:\n\n{email_text}"
    
    # Add to chat and process
    add_chat_message("user", f"Analyze email: {email.get('subject', 'Untitled')}")
    
    # Get agent and process
    agent = get_or_create_agent()
    if agent:
        try:
            response = agent.process_query(query)
            add_chat_message("assistant", response.response_text, {
                "action_taken": response.action_taken.value,
                "success": response.success,
                "email_analysis": True
            })
            
            if response.success:
                st.success("✅ Email analysis complete! Check the chat for details.")
            else:
                st.error(f"❌ Analysis failed: {response.error_message}")
                
        except Exception as e:
            st.error(f"❌ Error analyzing email: {str(e)}")
    else:
        st.error("❌ Conversational agent not available")

def draft_reply(email: Dict[str, Any]):
    """Draft a reply to the selected email."""
    
    # Extract sender for reply
    sender = email.get('sender', '')
    if '<' in sender:
        reply_to = sender.split('<')[1].replace('>', '').strip()
    else:
        reply_to = sender.strip()
    
    # Create reply subject
    subject = email.get('subject', '')
    if not subject.startswith('Re:'):
        subject = f"Re: {subject}"
    
    # Update draft form with reply information
    st.session_state.draft_recipient = reply_to
    st.session_state.draft_subject = subject
    st.session_state.draft_content = f"\n\n--- Original Message ---\nFrom: {email.get('sender', '')}\nSubject: {email.get('subject', '')}\n\n{email.get('content', '')}"
    
    # Create draft query for the chat
    query = f"Draft a reply to this email from {reply_to} with subject '{email.get('subject', '')}'"
    add_chat_message("user", query)
    
    # Get agent and process
    agent = get_or_create_agent()
    if agent:
        try:
            response = agent.process_query(query)
            add_chat_message("assistant", response.response_text, {
                "action_taken": response.action_taken.value,
                "success": response.success,
                "email_draft": True
            })
            
            st.success("✅ Reply draft started! Check the email drafter on the right.")
            
        except Exception as e:
            st.error(f"❌ Error drafting reply: {str(e)}")
    else:
        st.error("❌ Conversational agent not available")

def draft_forward(email: Dict[str, Any]):
    """Draft a forward for the selected email."""
    
    # Create forward subject
    subject = email.get('subject', '')
    if not subject.startswith('Fwd:'):
        subject = f"Fwd: {subject}"
    
    # Update draft form with forward information
    st.session_state.draft_recipient = ""
    st.session_state.draft_subject = subject
    st.session_state.draft_content = f"\n\n--- Forwarded Message ---\nFrom: {email.get('sender', '')}\nDate: {email.get('timestamp', '')}\nSubject: {email.get('subject', '')}\n\n{email.get('content', '')}"
    
    # Create draft query for the chat
    query = f"Help me draft a forward for this email with subject '{email.get('subject', '')}'"
    add_chat_message("user", query)
    
    st.success("✅ Forward draft started! Check the email drafter on the right.")

def render_email_summary():
    """Render email summary statistics."""
    
    emails = st.session_state.get('emails', [])
    unread_count = len([e for e in emails if not e.get('read', False)])
    urgent_count = len([e for e in emails if e.get('priority') == 'URGENT'])
    high_count = len([e for e in emails if e.get('priority') == 'HIGH'])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total", len(emails))
    
    with col2:
        st.metric("Unread", unread_count)
    
    with col3:
        st.metric("Urgent", urgent_count)
    
    with col4:
        st.metric("High Priority", high_count)

def render_email_search():
    """Render email search functionality."""
    
    search_query = st.text_input(
        "🔍 Search emails",
        placeholder="Search by subject, sender, or content...",
        key="email_search"
    )
    
    if search_query:
        # Filter emails based on search query
        emails = st.session_state.get('emails', [])
        search_results = []
        
        for email in emails:
            if (search_query.lower() in email.get('subject', '').lower() or
                search_query.lower() in email.get('sender', '').lower() or
                search_query.lower() in email.get('content', '').lower()):
                search_results.append(email)
        
        st.session_state.filtered_emails = search_results
        
        if search_results:
            st.success(f"Found {len(search_results)} matching emails")
        else:
            st.info("No emails found matching your search")

# Integration function to render the complete email panel
def render_complete_email_panel():
    """Render the complete email panel with all components."""
    
    # Panel header
    st.markdown("""
    <div style="background: linear-gradient(90deg, #1f4e79 0%, #2980b9 100%); 
                color: white; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
        <h3>📧 Email Management Center</h3>
        <p>Manage, analyze, and respond to your emails</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Email summary
    render_email_summary()
    
    st.markdown("---")
    
    # Email search
    render_email_search()
    
    # Email filters and list
    render_email_filters()
    
    # Email list
    render_email_list()
    
    # Email viewer (if email selected)
    if st.session_state.get('selected_email'):
        render_email_viewer()
    
    # Quick actions
    if st.session_state.get('emails'):
        st.markdown("### ⚡ Quick Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Mark All Read", help="Mark all emails as read"):
                for email in st.session_state.emails:
                    update_email_status(email['id'], {'read': True})
                st.success("All emails marked as read!")
                st.rerun()
        
        with col2:
            if st.button("🔄 Refresh Emails", help="Refresh email list"):
                # In a real implementation, this would fetch new emails
                st.success("Email list refreshed!")
                st.rerun()
