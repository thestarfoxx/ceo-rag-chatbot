import streamlit as st
from typing import Dict, Any
import sys
from pathlib import Path

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.session_state import (
    get_or_create_agent,
    add_chat_message,
    save_drafted_email,
    reset_draft_form
)

def render_email_drafter():
    """Render the right email drafter panel."""
    
    st.markdown("""
    <div style="background: linear-gradient(90deg, #1f4e79 0%, #2980b9 100%); 
                color: white; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
        <h3>✍️ Email Drafter</h3>
        <p>Compose and send professional emails</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Draft form
    render_draft_form()
    
    # AI Assistant section
    render_ai_assistant_section()
    
    # Recent drafts
    render_recent_drafts()

def render_draft_form():
    """Render the email drafting form."""
    
    st.markdown("### 📝 Compose Email")
    
    # Email recipient
    recipient = st.text_input(
        "To:",
        value=st.session_state.get('draft_recipient', ''),
        placeholder="recipient@company.com",
        key="draft_recipient_input",
        help="Enter the recipient's email address"
    )
    st.session_state.draft_recipient = recipient
    
    # Email subject
    subject = st.text_input(
        "Subject:",
        value=st.session_state.get('draft_subject', ''),
        placeholder="Enter email subject",
        key="draft_subject_input",
        help="Enter a clear, descriptive subject line"
    )
    st.session_state.draft_subject = subject
    
    # Email tone selection
    col1, col2 = st.columns(2)
    
    with col1:
        tone = st.selectbox(
            "Tone:",
            ["professional", "friendly", "formal", "casual", "urgent"],
            index=0,
            key="draft_tone_select",
            help="Choose the tone for your email"
        )
        st.session_state.draft_tone = tone
    
    with col2:
        email_type = st.selectbox(
            "Type:",
            ["general", "meeting", "request", "update", "announcement", "follow-up"],
            index=0,
            key="draft_type_select",
            help="Choose the type of email"
        )
    
    # Email content
    content = st.text_area(
        "Content:",
        value=st.session_state.get('draft_content', ''),
        placeholder="Enter your message here or use AI assistance below...",
        height=200,
        key="draft_content_input",
        help="Enter the main content of your email"
    )
    st.session_state.draft_content = content
    
    # Action buttons
    render_draft_action_buttons(recipient, subject, content, tone, email_type)

def render_draft_action_buttons(recipient: str, subject: str, content: str, tone: str, email_type: str):
    """Render action buttons for the draft form."""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🤖 AI Draft", help="Generate email using AI", type="primary"):
            generate_ai_draft(recipient, subject, content, tone, email_type)
    
    with col2:
        if st.button("💾 Save Draft", help="Save current draft"):
            save_current_draft(recipient, subject, content, tone)
    
    with col3:
        if st.button("🗑️ Clear", help="Clear all fields"):
            reset_draft_form()
            st.success("Draft form cleared!")
            st.rerun()
    
    # Send button (full width)
    if st.button("📤 Send Email", help="Send the email", type="secondary", use_container_width=True):
        send_email(recipient, subject, content, tone)

def generate_ai_draft(recipient: str, subject: str, content_request: str, tone: str, email_type: str):
    """Generate an AI-powered email draft."""
    
    if not recipient:
        st.error("Please enter a recipient email address")
        return
    
    if not subject and not content_request:
        st.error("Please enter either a subject or describe what you want to communicate")
        return
    
    # Create a comprehensive prompt for the AI
    if content_request:
        ai_prompt = f"Draft a {tone} {email_type} email to {recipient} with subject '{subject}' about: {content_request}"
    else:
        ai_prompt = f"Draft a {tone} {email_type} email to {recipient} with subject '{subject}'"
    
    # Add to chat
    add_chat_message("user", ai_prompt)
    
    # Get agent and process
    agent = get_or_create_agent()
    if agent:
        try:
            with st.spinner("Generating AI draft..."):
                response = agent.process_query(ai_prompt)
            
            add_chat_message("assistant", response.response_text, {
                "action_taken": response.action_taken.value,
                "success": response.success,
                "email_draft": True
            })
            
            if response.success and response.data:
                # Update form with AI-generated content
                email_data = response.data
                
                st.session_state.draft_subject = email_data.get('subject', subject)
                st.session_state.draft_content = email_data.get('full_email', email_data.get('body', ''))
                
                # Save the draft
                save_drafted_email(email_data)
                
                st.success("✅ AI draft generated successfully!")
                st.rerun()
            else:
                st.error(f"❌ Failed to generate draft: {response.error_message or 'Unknown error'}")
                
        except Exception as e:
            st.error(f"❌ Error generating AI draft: {str(e)}")
    else:
        st.error("❌ AI assistant not available")

def save_current_draft(recipient: str, subject: str, content: str, tone: str):
    """Save the current draft to session state."""
    
    if not recipient or not subject:
        st.error("Please enter at least recipient and subject to save draft")
        return
    
    draft_data = {
        "recipient": recipient,
        "subject": subject,
        "body": content,
        "full_email": content,
        "tone": tone,
        "timestamp": st.session_state.get('current_time', ''),
        "status": "draft"
    }
    
    save_drafted_email(draft_data)
    
    # Also save to a drafts list
    if 'saved_drafts' not in st.session_state:
        st.session_state.saved_drafts = []
    
    st.session_state.saved_drafts.append(draft_data)
    
    # Keep only last 10 drafts
    if len(st.session_state.saved_drafts) > 10:
        st.session_state.saved_drafts = st.session_state.saved_drafts[-10:]
    
    st.success("✅ Draft saved successfully!")

def send_email(recipient: str, subject: str, content: str, tone: str):
    """Simulate sending an email."""
    
    if not recipient:
        st.error("Please enter a recipient email address")
        return
    
    if not subject:
        st.error("Please enter an email subject")
        return
    
    if not content:
        st.error("Please enter email content")
        return
    
    # Simulate sending (in real implementation, this would integrate with email service)
    with st.spinner("Sending email..."):
        import time
        time.sleep(1)  # Simulate sending delay
    
    # Save to sent emails
    sent_email = {
        "recipient": recipient,
        "subject": subject,
        "body": content,
        "tone": tone,
        "timestamp": st.session_state.get('current_time', ''),
        "status": "sent"
    }
    
    if 'sent_emails' not in st.session_state:
        st.session_state.sent_emails = []
    
    st.session_state.sent_emails.append(sent_email)
    
    # Add to chat log
    add_chat_message("assistant", f"📤 Email sent successfully to {recipient} with subject '{subject}'")
    
    # Clear form after sending
    reset_draft_form()
    
    st.success(f"✅ Email sent successfully to {recipient}!")
    st.balloons()
    st.rerun()

def render_ai_assistant_section():
    """Render the AI assistant section for email help."""
    
    st.markdown("---")
    st.markdown("### 🤖 AI Email Assistant")
    
    # Quick AI actions
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💡 Suggest Subject", help="Get AI suggestion for email subject"):
            suggest_subject()
        
        if st.button("🎯 Improve Tone", help="Improve email tone"):
            improve_tone()
    
    with col2:
        if st.button("✍️ Expand Content", help="Expand email content"):
            expand_content()
        
        if st.button("📝 Proofread", help="Proofread and improve email"):
            proofread_email()
    
    # AI prompt input
    ai_prompt = st.text_input(
        "Ask AI for help:",
        placeholder="e.g., 'Make this more formal' or 'Add a call to action'",
        key="ai_prompt_input",
        help="Ask the AI to help improve your email"
    )
    
    if st.button("🚀 Ask AI", help="Get AI assistance") and ai_prompt:
        get_ai_assistance(ai_prompt)

def suggest_subject():
    """Get AI suggestion for email subject."""
    
    content = st.session_state.get('draft_content', '')
    recipient = st.session_state.get('draft_recipient', '')
    
    if not content:
        st.error("Please enter some email content first")
        return
    
    prompt = f"Suggest a professional subject line for an email to {recipient} with this content: {content[:200]}..."
    
    agent = get_or_create_agent()
    if agent:
        try:
            response = agent.process_query(prompt)
            
            # Extract suggested subject from response
            suggestion = response.response_text
            if "subject" in suggestion.lower():
                # Try to extract just the subject line
                lines = suggestion.split('\n')
                for line in lines:
                    if 'subject' in line.lower() and ':' in line:
                        suggested_subject = line.split(':', 1)[1].strip().strip('"')
                        st.session_state.draft_subject = suggested_subject
                        st.success(f"✅ Subject updated: {suggested_subject}")
                        st.rerun()
                        return
            
            st.info(f"💡 AI Suggestion: {suggestion}")
            
        except Exception as e:
            st.error(f"❌ Error getting subject suggestion: {str(e)}")

def improve_tone():
    """Improve the tone of the email."""
    
    content = st.session_state.get('draft_content', '')
    tone = st.session_state.get('draft_tone', 'professional')
    
    if not content:
        st.error("Please enter email content first")
        return
    
    prompt = f"Rewrite this email content to be more {tone} in tone: {content}"
    
    agent = get_or_create_agent()
    if agent:
        try:
            response = agent.process_query(prompt)
            
            # Update content with improved version
            improved_content = response.response_text
            st.session_state.draft_content = improved_content
            
            st.success("✅ Email tone improved!")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error improving tone: {str(e)}")

def expand_content():
    """Expand the email content."""
    
    content = st.session_state.get('draft_content', '')
    
    if not content:
        st.error("Please enter some initial content first")
        return
    
    prompt = f"Expand and improve this email content while keeping it professional and concise: {content}"
    
    agent = get_or_create_agent()
    if agent:
        try:
            response = agent.process_query(prompt)
            
            # Update content with expanded version
            expanded_content = response.response_text
            st.session_state.draft_content = expanded_content
            
            st.success("✅ Email content expanded!")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error expanding content: {str(e)}")

def proofread_email():
    """Proofread and improve the email."""
    
    subject = st.session_state.get('draft_subject', '')
    content = st.session_state.get('draft_content', '')
    
    if not content:
        st.error("Please enter email content first")
        return
    
    email_text = f"Subject: {subject}\n\n{content}"
    prompt = f"Proofread and improve this email for grammar, clarity, and professionalism: {email_text}"
    
    agent = get_or_create_agent()
    if agent:
        try:
            response = agent.process_query(prompt)
            
            st.info(f"📝 Proofreading suggestions:\n{response.response_text}")
            
        except Exception as e:
            st.error(f"❌ Error proofreading email: {str(e)}")

def get_ai_assistance(prompt: str):
    """Get AI assistance with custom prompt."""
    
    content = st.session_state.get('draft_content', '')
    subject = st.session_state.get('draft_subject', '')
    
    full_prompt = f"Help me with this email (Subject: {subject}): {content}\n\nRequest: {prompt}"
    
    agent = get_or_create_agent()
    if agent:
        try:
            response = agent.process_query(full_prompt)
            
            add_chat_message("user", prompt)
            add_chat_message("assistant", response.response_text, {
                "email_assistance": True
            })
            
            st.success("✅ AI assistance provided! Check the chat for details.")
            
        except Exception as e:
            st.error(f"❌ Error getting AI assistance: {str(e)}")

def render_recent_drafts():
    """Render recent drafts section."""
    
    saved_drafts = st.session_state.get('saved_drafts', [])
    sent_emails = st.session_state.get('sent_emails', [])
    
    if saved_drafts or sent_emails:
        st.markdown("---")
        st.markdown("### 📋 Recent Activity")
        
        # Tabs for drafts and sent emails
        tab1, tab2 = st.tabs(["💾 Saved Drafts", "📤 Sent Emails"])
        
        with tab1:
            if saved_drafts:
                for i, draft in enumerate(reversed(saved_drafts[-5:])):  # Show last 5
                    with st.expander(f"📝 {draft.get('subject', 'No Subject')[:30]}..."):
                        st.write(f"**To:** {draft.get('recipient', 'Unknown')}")
                        st.write(f"**Tone:** {draft.get('tone', 'professional')}")
                        st.write(f"**Saved:** {draft.get('timestamp', 'Unknown')}")
                        st.text_area("Content", draft.get('body', ''), height=100, disabled=True, key=f"draft_preview_{i}")
                        
                        if st.button(f"Load Draft", key=f"load_draft_{i}"):
                            st.session_state.draft_recipient = draft.get('recipient', '')
                            st.session_state.draft_subject = draft.get('subject', '')
                            st.session_state.draft_content = draft.get('body', '')
                            st.session_state.draft_tone = draft.get('tone', 'professional')
                            st.success("Draft loaded!")
                            st.rerun()
            else:
                st.info("No saved drafts")
        
        with tab2:
            if sent_emails:
                for i, email in enumerate(reversed(sent_emails[-5:])):  # Show last 5
                    with st.expander(f"📤 {email.get('subject', 'No Subject')[:30]}..."):
                        st.write(f"**To:** {email.get('recipient', 'Unknown')}")
                        st.write(f"**Sent:** {email.get('timestamp', 'Unknown')}")
                        st.text_area("Content", email.get('body', ''), height=100, disabled=True, key=f"sent_preview_{i}")
            else:
                st.info("No sent emails")

def render_email_templates():
    """Render email templates section."""
    
    templates = {
        "Meeting Request": {
            "subject": "Meeting Request - [Topic]",
            "content": "Hi [Name],\n\nI hope this email finds you well. I would like to schedule a meeting to discuss [topic/project]. \n\nWould you be available for a [duration] meeting sometime next week? I'm flexible with timing and can accommodate your schedule.\n\nPlease let me know what works best for you.\n\nBest regards,\n[Your name]"
        },
        "Follow-up": {
            "subject": "Following up on [Previous Discussion]",
            "content": "Hi [Name],\n\nI wanted to follow up on our conversation about [topic] from [when/where].\n\n[Specific follow-up points or questions]\n\nI look forward to hearing your thoughts and next steps.\n\nBest regards,\n[Your name]"
        },
        "Project Update": {
            "subject": "Project Update - [Project Name]",
            "content": "Hi Team,\n\nI wanted to share a quick update on the [project name] project:\n\n✅ Completed:\n- [Task 1]\n- [Task 2]\n\n🔄 In Progress:\n- [Task 3]\n- [Task 4]\n\n📅 Next Steps:\n- [Task 5] (Due: [date])\n- [Task 6] (Due: [date])\n\nPlease let me know if you have any questions or concerns.\n\nBest regards,\n[Your name]"
        },
        "Thank You": {
            "subject": "Thank You - [Occasion/Reason]",
            "content": "Hi [Name],\n\nI wanted to take a moment to thank you for [specific reason]. Your [help/support/contribution] has been invaluable to [project/goal/outcome].\n\n[Specific details about impact or appreciation]\n\nI truly appreciate your time and effort.\n\nBest regards,\n[Your name]"
        },
        "Announcement": {
            "subject": "Important Announcement - [Topic]",
            "content": "Dear Team,\n\nI'm writing to inform you about [announcement topic].\n\n[Main announcement details]\n\nKey points:\n- [Point 1]\n- [Point 2]\n- [Point 3]\n\nThis change will take effect on [date]. If you have any questions, please don't hesitate to reach out.\n\nThank you for your attention.\n\nBest regards,\n[Your name]"
        }
    }
    
    with st.expander("📄 Email Templates"):
        template_choice = st.selectbox(
            "Choose a template:",
            ["Select a template..."] + list(templates.keys()),
            key="template_selector"
        )
        
        if template_choice != "Select a template...":
            template = templates[template_choice]
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Preview:**")
                st.text_area(
                    "Template Content",
                    value=f"Subject: {template['subject']}\n\n{template['content']}",
                    height=200,
                    disabled=True,
                    key="template_preview"
                )
            
            with col2:
                st.write("**Actions:**")
                if st.button("📋 Use This Template", key="use_template"):
                    st.session_state.draft_subject = template['subject']
                    st.session_state.draft_content = template['content']
                    st.success(f"✅ {template_choice} template loaded!")
                    st.rerun()
                
                if st.button("🎨 Customize Template", key="customize_template"):
                    # Open customization interface
                    st.session_state.customizing_template = template_choice
                    st.rerun()

def render_email_statistics():
    """Render email statistics and insights."""
    
    sent_emails = st.session_state.get('sent_emails', [])
    saved_drafts = st.session_state.get('saved_drafts', [])
    
    if sent_emails or saved_drafts:
        with st.expander("📊 Email Statistics"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Emails Sent", len(sent_emails))
            
            with col2:
                st.metric("Drafts Saved", len(saved_drafts))
            
            with col3:
                # Calculate most used tone
                tones = [email.get('tone', 'professional') for email in sent_emails]
                most_used_tone = max(set(tones), key=tones.count) if tones else "N/A"
                st.metric("Preferred Tone", most_used_tone)
            
            # Show tone distribution
            if sent_emails:
                st.write("**Tone Distribution:**")
                tone_counts = {}
                for email in sent_emails:
                    tone = email.get('tone', 'professional')
                    tone_counts[tone] = tone_counts.get(tone, 0) + 1
                
                for tone, count in sorted(tone_counts.items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / len(sent_emails)) * 100
                    st.write(f"• {tone.title()}: {count} ({percentage:.1f}%)")

# Main render function that brings everything together
def render_complete_email_drafter():
    """Render the complete email drafter with all components."""
    
    # Header
    st.markdown("""
    <div style="background: linear-gradient(90deg, #1f4e79 0%, #2980b9 100%); 
                color: white; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
        <h3>✍️ Smart Email Drafter</h3>
        <p>AI-powered email composition and management</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main draft form
    render_draft_form()
    
    # AI assistant section
    render_ai_assistant_section()
    
    # Email templates
    render_email_templates()
    
    # Recent activity
    render_recent_drafts()
    
    # Statistics
    render_email_statistics()
    
    # Quick tips
    with st.expander("💡 Email Writing Tips"):
        st.markdown("""
        **Professional Email Best Practices:**
        
        📝 **Subject Line:**
        - Be specific and action-oriented
        - Keep it under 50 characters
        - Avoid spam trigger words
        
        ✍️ **Content:**
        - Start with a clear purpose
        - Use bullet points for multiple items
        - Include a clear call-to-action
        - Keep paragraphs short
        
        🎯 **Tone:**
        - Match the relationship and context
        - Be respectful and professional
        - Consider cultural differences
        
        📤 **Before Sending:**
        - Proofread for errors
        - Check recipient addresses
        - Ensure attachments are included
        - Review tone and clarity
        """)

# Integration with the chat system
def handle_email_draft_from_chat(email_data: Dict[str, Any]):
    """Handle email draft data coming from the chat system."""
    
    if email_data:
        # Update draft form with chat-generated content
        st.session_state.draft_recipient = email_data.get('recipient', '')
        st.session_state.draft_subject = email_data.get('subject', '')
        st.session_state.draft_content = email_data.get('full_email', email_data.get('body', ''))
        
        # Save the draft
        save_drafted_email(email_data)
        
        # Show success message
        st.success("✅ Email draft updated from chat!")

def handle_email_context_from_chat(context: str):
    """Handle email context or content from chat for drafting."""
    
    if context:
        current_content = st.session_state.get('draft_content', '')
        
        # Append context to existing content
        if current_content:
            st.session_state.draft_content = f"{current_content}\n\n--- From Chat ---\n{context}"
        else:
            st.session_state.draft_content = context
        
        st.success("✅ Content added from chat!")

# Utility functions for email validation
def validate_email_address(email: str) -> bool:
    """Validate email address format."""
    import re
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}
    return re.match(pattern, email) is not None

def validate_draft_form() -> tuple[bool, list[str]]:
    """Validate the current draft form and return errors."""
    
    errors = []
    
    recipient = st.session_state.get('draft_recipient', '')
    subject = st.session_state.get('draft_subject', '')
    content = st.session_state.get('draft_content', '')
    
    if not recipient:
        errors.append("Recipient email is required")
    elif not validate_email_address(recipient):
        errors.append("Invalid email address format")
    
    if not subject.strip():
        errors.append("Subject line is required")
    
    if not content.strip():
        errors.append("Email content is required")
    
    return len(errors) == 0, errors

def get_draft_word_count() -> int:
    """Get word count of current draft."""
    content = st.session_state.get('draft_content', '')
    return len(content.split()) if content else 0

def get_draft_character_count() -> int:
    """Get character count of current draft."""
    content = st.session_state.get('draft_content', '')
    return len(content)

def render_draft_metrics():
    """Render draft metrics (word count, character count, etc.)."""
    
    word_count = get_draft_word_count()
    char_count = get_draft_character_count()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Words", word_count)
    
    with col2:
        st.metric("Characters", char_count)
    
    # Reading time estimate (average 200 words per minute)
    reading_time = max(1, word_count // 200)
    if reading_time > 1:
        st.info(f"📖 Estimated reading time: {reading_time} minutes")

# Error handling and user feedback
def show_draft_validation_errors():
    """Show validation errors for the current draft."""
    
    is_valid, errors = validate_draft_form()
    
    if not is_valid:
        st.error("Please fix the following errors:")
        for error in errors:
            st.error(f"• {error}")
    
    return is_valid
