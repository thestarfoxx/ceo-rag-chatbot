#!/usr/bin/env python3
"""
Console-based Conversational RAG System with Email Functionality (LM Studio version)

This module provides a console interface that allows a CEO (or other
executives) to interact with a Retrieval-Augmented Generation (RAG)
system. The assistant can search documents, query structured data,
draft/analyze emails, and chat—using a local LM Studio-compatible API.

Changes in this version:
- Replaced OpenAI SDK usage with LM Studio-style client initialization:
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")
- All model calls now use `client.chat.completions.create(...)`.
- Model name and base URL are configurable via environment variables.
"""

import logging
import json
import re
import os
import sys
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from openai import OpenAI  # LM Studio-compatible client

# Import the retriever; exit with a helpful message if missing.
try:
    from retriever import HybridRetriever, RetrievalResult, create_retriever
except ImportError:
    print(
        "Error: Could not import retriever module. Make sure retriever.py is in the same directory."
    )
    sys.exit(1)

# --- Configuration helpers ---

def create_lmstudio_client() -> OpenAI:
    """
    Create an OpenAI-compatible client pointing to LM Studio (or another compatible server).
    Environment overrides:
      LMSTUDIO_BASE_URL (default: http://localhost:1234/v1)
      LMSTUDIO_API_KEY  (default: not-needed)
    """
    base_url = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
    api_key = os.getenv("LMSTUDIO_API_KEY", "not-needed")
    return OpenAI(base_url=base_url, api_key=api_key)


def get_chat_model_name() -> str:
    """
    Resolve chat model name for LM Studio.
    Environment override:
      LMSTUDIO_MODEL_NAME (default: openai/gpt-oss-20b)
    """
    return os.getenv("LMSTUDIO_MODEL_NAME", "openai/gpt-oss-20b")


# Configure logging to file and stdout.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("data/logs/conversation.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class QueryType(Enum):
    DOCUMENT_SEARCH = "document_search"
    DATA_QUERY = "data_query"
    EMAIL_DRAFT = "email_draft"
    EMAIL_ANALYZE = "email_analyze"
    GENERAL_CHAT = "general_chat"
    HELP = "help"
    SYSTEM_STATUS = "system_status"


@dataclass
class ConversationContext:
    history: List[Dict[str, Any]]
    current_topic: Optional[str] = None
    last_query_type: Optional[QueryType] = None
    session_start: datetime = datetime.now()

    def add_exchange(
        self,
        user_query: str,
        assistant_response: str,
        query_type: QueryType,
        metadata: Dict[str, Any] = None,
    ) -> None:
        self.history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "user_query": user_query,
                "assistant_response": assistant_response,
                "query_type": query_type.value,
                "metadata": metadata or {},
            }
        )
        self.last_query_type = query_type
        if len(self.history) > 10:
            self.history = self.history[-10:]

    def get_context_string(self, max_exchanges: int = 3) -> str:
        if not self.history:
            return ""
        recent = self.history[-max_exchanges:]
        context_parts: List[str] = []
        for exchange in recent:
            context_parts.append(f"User: {exchange['user_query']}")
            response = exchange["assistant_response"]
            if len(response) > 200:
                response = response[:200] + "..."
            context_parts.append(f"Assistant: {response}")
        return "\n".join(context_parts)


class EmailProcessor:
    """Handles drafting and analyzing emails via LM Studio-compatible chat API."""

    def __init__(self, client: OpenAI, model_name: str) -> None:
        self.client = client
        self.model_name = model_name

    def draft_email(
        self,
        recipient: str = "",
        subject: str = "",
        content_request: str = "",
        tone: str = "professional",
        context: str = "",
    ) -> Dict[str, Any]:
        try:
            prompt_parts = [
                f"Draft a {tone} email with the following specifications:",
                f"Recipient: {recipient or 'To be specified'}",
                f"Subject: {subject or 'To be determined based on content'}",
                f"Content Request: {content_request}",
            ]
            if context:
                prompt_parts.append(f"Additional Context: {context}")
            prompt_parts.extend(
                [
                    "",
                    "Requirements:",
                    f"1. Use appropriate {tone} tone throughout",
                    "2. Include proper email structure (greeting, body, closing)",
                    "3. Be clear, concise, and actionable",
                    "4. Include relevant business context if applicable",
                    "5. Use proper formatting and professional language",
                    "6. If recipient or subject not specified, suggest appropriate ones",
                    "",
                    "Return the email in a structured format with:",
                    "- Suggested recipient (if not provided)",
                    "- Subject line",
                    "- Complete email body",
                    "- Brief explanation of tone and approach used",
                ]
            )
            full_prompt = "\n".join(prompt_parts)

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert business email writer specializing in executive communications. "
                            "Create professional, effective emails that achieve the user's goals."
                        ),
                    },
                    {"role": "user", "content": full_prompt},
                ],
                max_tokens=1000,
                temperature=0.3,
            )
            email_content = response.choices[0].message.content.strip()

            subject_match = re.search(
                r"Subject:\s*(.+?)(?:\n|$)", email_content, re.IGNORECASE
            )
            recipient_match = re.search(
                r"(?:To|Recipient):\s*(.+?)(?:\n|$)", email_content, re.IGNORECASE
            )
            extracted_subject = subject_match.group(1).strip() if subject_match else subject
            extracted_recipient = recipient_match.group(1).strip() if recipient_match else recipient

            return {
                "success": True,
                "email": {
                    "recipient": extracted_recipient,
                    "subject": extracted_subject,
                    "body": email_content,
                    "tone": tone,
                    "content_request": content_request,
                },
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "model_used": self.model_name,
                    "has_context": bool(context),
                },
            }
        except Exception as e:
            logger.error(f"Error drafting email: {str(e)}")
            return {"success": False, "error": str(e), "email": None}

    def analyze_email(
        self,
        email_content: str,
        sender: str = "",
        received_date: str = "",
    ) -> Dict[str, Any]:
        try:
            analysis_prompt = f"""
Analyze the following email comprehensively and provide structured insights:

Sender: {sender or 'Unknown'}
Received: {received_date or 'Not specified'}

Email Content:
{email_content}

Provide a detailed analysis covering:

1. SUMMARY: Concise 2-3 sentence summary of the email's main purpose
2. KEY POINTS: List the most important points or requests
3. URGENCY LEVEL: Assess as LOW, MEDIUM, HIGH, or URGENT with reasoning
4. SENTIMENT: Overall tone (POSITIVE, NEUTRAL, NEGATIVE, MIXED)
5. ACTION ITEMS: Specific actions required from the recipient
6. DEADLINES: Any mentioned deadlines or time-sensitive items
7. PRIORITY SCORE: Rate from 1-10 with justification
8. RESPONSE REQUIRED: Whether and when a response is needed
9. CATEGORY: Classify the email type (meeting, financial, hr, operational, legal, etc.)
10. BUSINESS IMPACT: Potential impact on business operations
11. STAKEHOLDERS: Who else might need to be involved
12. RECOMMENDATIONS: Suggested next steps or actions

Be thorough and consider both explicit and implicit information in the email.
"""
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert executive assistant specializing in email analysis and communications. "
                            "Provide detailed, actionable insights."
                        ),
                    },
                    {"role": "user", "content": analysis_prompt},
                ],
                max_tokens=800,
                temperature=0.1,
            )
            analysis_content = response.choices[0].message.content.strip()

            urgency_match = re.search(
                r"URGENCY LEVEL:\s*(LOW|MEDIUM|HIGH|URGENT)", analysis_content, re.IGNORECASE
            )
            priority_match = re.search(r"PRIORITY SCORE:\s*(\d+)", analysis_content)
            category_match = re.search(r"CATEGORY:\s*([^\n]+)", analysis_content, re.IGNORECASE)

            urgency = urgency_match.group(1).upper() if urgency_match else "MEDIUM"
            priority = int(priority_match.group(1)) if priority_match else 5
            category = category_match.group(1).strip() if category_match else "general"

            return {
                "success": True,
                "analysis": {
                    "full_analysis": analysis_content,
                    "urgency_level": urgency,
                    "priority_score": priority,
                    "category": category,
                    "sender": sender,
                    "received_date": received_date,
                    "content_length": len(email_content),
                    "word_count": len(email_content.split()),
                },
                "metadata": {
                    "analyzed_at": datetime.now().isoformat(),
                    "model_used": self.model_name,
                },
            }
        except Exception as e:
            logger.error(f"Error analyzing email: {str(e)}")
            return {"success": False, "error": str(e), "analysis": None}


class QueryClassifier:
    """Classifies user queries via LM Studio-compatible chat API."""

    def __init__(self, client: OpenAI, model_name: str) -> None:
        self.client = client
        self.model_name = model_name

    def classify_query(
        self, query: str, context: str = ""
    ) -> Tuple[QueryType, Dict[str, Any]]:
        try:
            classification_prompt = f"""
Analyze this user query and classify it into one of these categories:

1. DOCUMENT_SEARCH - Search through documents/PDFs/reports
2. DATA_QUERY - Structured data / SQL / metrics
3. EMAIL_DRAFT - Compose or draft an email
4. EMAIL_ANALYZE - Analyze a received email
5. HELP - Help/instructions/commands
6. SYSTEM_STATUS - System status, available data
7. GENERAL_CHAT - General conversation

CONVERSATION CONTEXT:
{context}

USER QUERY: {query}

Respond with just the category name (e.g., "DOCUMENT_SEARCH") and extract any relevant details:
- For EMAIL_DRAFT: recipient, subject, content request, tone
- For EMAIL_ANALYZE: email content
- For DOCUMENT_SEARCH: search terms
- For DATA_QUERY: data/analysis needed
"""
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a query classifier. Classify user queries accurately and extract any details.",
                    },
                    {"role": "user", "content": classification_prompt},
                ],
                max_tokens=150,
                temperature=0.1,
            )
            classification_content = response.choices[0].message.content.strip().upper()

            query_type: Optional[QueryType] = None
            for qt in QueryType:
                if qt.value.upper() in classification_content:
                    query_type = qt
                    break

            if not query_type:
                query_lower = query.lower()
                if any(w in query_lower for w in ["draft", "write", "compose", "email to"]):
                    query_type = QueryType.EMAIL_DRAFT
                elif any(w in query_lower for w in ["analyze email", "read email", "email from"]):
                    query_type = QueryType.EMAIL_ANALYZE
                elif any(w in query_lower for w in ["search", "find", "document", "report"]):
                    query_type = QueryType.DOCUMENT_SEARCH
                elif any(w in query_lower for w in ["data", "statistics", "ranking", "top"]):
                    query_type = QueryType.DATA_QUERY
                elif any(w in query_lower for w in ["help", "how to", "what can"]):
                    query_type = QueryType.HELP
                elif any(w in query_lower for w in ["status", "available", "sources"]):
                    query_type = QueryType.SYSTEM_STATUS
                else:
                    query_type = QueryType.GENERAL_CHAT

            extracted_info = self._extract_query_info(query, query_type)
            return query_type, extracted_info

        except Exception as e:
            logger.error(f"Error classifying query: {str(e)}")
            # Heuristic fallback
            query_lower = query.lower()
            if any(w in query_lower for w in ["draft", "write", "compose", "email to"]):
                query_type = QueryType.EMAIL_DRAFT
            elif any(w in query_lower for w in ["analyze email", "read email", "email from"]):
                query_type = QueryType.EMAIL_ANALYZE
            elif any(w in query_lower for w in ["search", "find", "document", "report"]):
                query_type = QueryType.DOCUMENT_SEARCH
            elif any(w in query_lower for w in ["data", "statistics", "ranking", "top"]):
                query_type = QueryType.DATA_QUERY
            elif any(w in query_lower for w in ["help", "how to", "what can"]):
                query_type = QueryType.HELP
            elif any(w in query_lower for w in ["status", "available", "sources"]):
                query_type = QueryType.SYSTEM_STATUS
            else:
                query_type = QueryType.GENERAL_CHAT
            extracted_info = self._extract_query_info(query, query_type)
            return query_type, extracted_info

    def _extract_query_info(
        self, query: str, query_type: QueryType
    ) -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        if query_type == QueryType.EMAIL_DRAFT:
            recipient_match = re.search(
                r"(?:to|email)\s+(.+?)(?:\s+about|\s+regarding|$)", query, re.IGNORECASE
            )
            subject_match = re.search(
                r"(?:about|regarding|subject)\s+(.+)", query, re.IGNORECASE
            )
            info["recipient"] = recipient_match.group(1).strip() if recipient_match else ""
            info["subject"] = subject_match.group(1).strip() if subject_match else ""
            info["content_request"] = query
            info["tone"] = "professional"
        elif query_type == QueryType.EMAIL_ANALYZE:
            info["email_content"] = query
        elif query_type == QueryType.DOCUMENT_SEARCH:
            info["search_terms"] = query
            info["document_types"] = []
        elif query_type == QueryType.DATA_QUERY:
            info["data_request"] = query
        return info


class ResponseAnalyzer:
    """Synthesizes and analyzes retrieved documents/data via LM Studio chat API."""

    def __init__(self, client: OpenAI, model_name: str) -> None:
        self.client = client
        self.model_name = model_name

    def analyze_document_chunks(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        conversation_context: str = "",
    ) -> str:
        try:
            chunk_info = []
            for i, chunk in enumerate(chunks[:5], 1):
                chunk_summary = {
                    "index": i,
                    "source": chunk.get("file_name", "Unknown document"),
                    "page": chunk.get("page_number", 1),
                    "similarity": chunk.get("similarity_score", 0),
                    "content": chunk.get("text", "")[:1500],
                }
                chunk_info.append(chunk_summary)

            synthesis_prompt = f"""
You are an expert business analyst reviewing document search results for a CEO. 
Analyze the retrieved information and provide a comprehensive, executive-level response.

ORIGINAL QUERY: {query}

CONVERSATION CONTEXT:
{conversation_context}

RETRIEVED INFORMATION:
"""
            for chunk in chunk_info:
                synthesis_prompt += f"""
Document {chunk['index']}: {chunk['source']} (Page {chunk['page']}, Relevance: {chunk['similarity']:.3f})
Content: {chunk['content']}
---
"""
            synthesis_prompt += """

ANALYSIS INSTRUCTIONS:
1. Synthesize the information from all sources to directly answer the user's query
2. Identify key themes, patterns, and insights across the documents
3. Highlight the most important findings that address the query
4. Note any contradictions or gaps in the information
5. Provide specific details with document references
6. Include quantitative data where available
7. Structure the response with clear sections and bullet points
8. If information is insufficient, clearly state what's missing
9. Provide actionable insights or recommendations where appropriate
10. Maintain an executive-level tone suitable for a CEO

RESPONSE FORMAT:
- Direct answer to the query
- Detailed analysis with document references
- Key insights and implications
- Data sources and reliability
- Recommendations or next steps
"""
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert business intelligence analyst for executive decision-making.",
                    },
                    {"role": "user", "content": synthesis_prompt},
                ],
                max_tokens=1200,
                temperature=0.2,
            )
            synthesized_response = response.choices[0].message.content.strip()
            unique_sources = {chunk['source'] for chunk in chunk_info}
            metadata_footer = f"""

📊 **Analysis Metadata:**
• Sources analyzed: {len(unique_sources)} documents ({', '.join(sorted(unique_sources))})
• Information chunks: {len(chunks)} total, {len(chunk_info)} analyzed in detail
• Average relevance score: {sum(chunk.get('similarity_score', 0) for chunk in chunks) / max(len(chunks),1):.3f}
• Analysis model: {self.model_name}"""
            return synthesized_response + metadata_footer
        except Exception as e:
            logger.error(f"Error analyzing document chunks: {str(e)}")
            return f"❌ Error analyzing retrieved documents: {str(e)}"

    def analyze_data_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        generated_sql: str = "",
        conversation_context: str = "",
    ) -> str:
        try:
            if not results:
                return "No data found for your query."

            data_summary = {
                "row_count": len(results),
                "columns": list(results[0].keys()) if results else [],
                "sample_data": results[:10],
                "data_types": {},
            }
            if results:
                first_row = results[0]
                for col, value in first_row.items():
                    if col in ["source_type", "generated_sql", "selected_table"]:
                        continue
                    if isinstance(value, (int, float)):
                        data_summary["data_types"][col] = "numeric"
                    elif isinstance(value, str) and value.isdigit():
                        data_summary["data_types"][col] = "numeric_string"
                    else:
                        data_summary["data_types"][col] = "text"

            analysis_prompt = f"""
You are a senior business intelligence analyst providing executive-level data analysis for a CEO.
Analyze the SQL query results and provide comprehensive insights.

ORIGINAL QUERY: {query}

CONVERSATION CONTEXT:
{conversation_context}

SQL QUERY EXECUTED:
{generated_sql}

DATA SUMMARY:
- Total records: {data_summary['row_count']}
- Columns: {', '.join(col for col in data_summary['columns'] if col not in ['source_type','generated_sql','selected_table'])}

SAMPLE DATA:
{json.dumps([{k: v for k, v in row.items() if k not in ['source_type','generated_sql','selected_table']} for row in results[:5]], indent=2, ensure_ascii=False)}

ANALYSIS INSTRUCTIONS:
1. Directly answer the user's original query using the data
2. Identify key patterns, trends, and insights
3. Highlight top performers, outliers, and patterns
4. Provide quantitative analysis (percentages/ratios) when useful
5. Note data quality issues or limitations
6. Provide business context and implications
7. Structure the response clearly for executives

RESPONSE FORMAT:
- Executive Summary
- Key Findings
- Detailed Analysis
- Business Implications
- Recommendations
"""
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a senior BI analyst specializing in executive reporting and data-driven insights.",
                    },
                    {"role": "user", "content": analysis_prompt},
                ],
                max_tokens=1200,
                temperature=0.1,
            )
            analyzed_response = response.choices[0].message.content.strip()
            metadata_footer = f"""

🔧 **Technical Details:**
• Query returned {data_summary['row_count']} rows
• Data source: {results[0].get('selected_table', 'Unknown table') if results else 'N/A'}
• Analysis model: {self.model_name}
• SQL: {generated_sql[:100]}{'...' if len(generated_sql) > 100 else ''}"""
            return analyzed_response + metadata_footer
        except Exception as e:
            logger.error(f"Error analyzing data results: {str(e)}")
            return f"❌ Error analyzing data results: {str(e)}"

    def synthesize_hybrid_results(
        self,
        query: str,
        vector_chunks: List[Dict[str, Any]],
        sql_results: List[Dict[str, Any]],
        conversation_context: str = "",
    ) -> str:
        try:
            synthesis_prompt = f"""
You are a senior executive advisor analyzing multiple information sources to provide comprehensive insights for a CEO.
Synthesize information from both documents and structured data to answer the query thoroughly.

ORIGINAL QUERY: {query}

CONVERSATION CONTEXT:
{conversation_context}

DOCUMENT SOURCES:
"""
            if vector_chunks:
                for i, chunk in enumerate(vector_chunks[:3], 1):
                    synthesis_prompt += f"""
Document {i}: {chunk.get('file_name', 'Unknown')} (Page {chunk.get('page_number', 1)})
Relevance: {chunk.get('similarity_score', 0):.3f}
Content: {chunk.get('text', '')[:800]}
---
"""
            else:
                synthesis_prompt += "No relevant documents found.\n"

            synthesis_prompt += "STRUCTURED DATA:\n"
            if sql_results:
                synthesis_prompt += f"Results: {len(sql_results)} records found\n"
                synthesis_prompt += f"Sample data: {json.dumps(sql_results[:3], indent=2, ensure_ascii=False)}\n"
            else:
                synthesis_prompt += "No structured data found.\n"

            synthesis_prompt += f"""

COMPREHENSIVE ANALYSIS INSTRUCTIONS:
1. Provide a direct, comprehensive answer
2. Synthesize insights from both sources
3. Cross-reference information where possible
4. Identify correlations, patterns, and discrepancies
5. Provide quantitative backing from data and qualitative context from documents
6. Structure the response for executive consumption
7. Provide actionable recommendations

RESPONSE STRUCTURE:
- Executive Summary
- Document Insights
- Data Analysis
- Cross-Analysis
- Strategic Implications
- Recommendations
"""
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a senior strategy consultant synthesizing multiple information sources.",
                    },
                    {"role": "user", "content": synthesis_prompt},
                ],
                max_tokens=1500,
                temperature=0.2,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error synthesizing hybrid results: {str(e)}")
            return f"❌ Error synthesizing results: {str(e)}"


class ConversationalRAGSystem:
    """Main console-based conversational RAG system."""

    def __init__(self, openai_api_key: str = None, db_config: Dict[str, str] = None) -> None:
        """
        Initialize system components.

        Note: `openai_api_key` is still accepted and passed to `create_retriever`
        in case your retriever uses remote embeddings or other services.
        """
        # LM Studio client and model
        self.client = create_lmstudio_client()
        self.model_name = get_chat_model_name()

        # If your retriever needs an API key (e.g., for embeddings elsewhere), use env or dummy value.
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "not-needed")

        try:
            self.retriever = create_retriever(
                db_config=db_config,
                openai_api_key=self.openai_api_key,  # kept for compatibility with your retriever
                embedding_model=os.getenv("EMBEDDING_MODEL_NAME", "Snowflake/snowflake-arctic-embed-"),
            )
            self.retriever.update_search_parameters(
                similarity_threshold=0.15,
                max_vector_results=10,
                max_sql_results=50,
            )
        except Exception as e:
            logger.error(f"Failed to initialize retriever: {str(e)}")
            raise

        self.email_processor = EmailProcessor(self.client, self.model_name)
        self.query_classifier = QueryClassifier(self.client, self.model_name)
        self.response_analyzer = ResponseAnalyzer(self.client, self.model_name)
        self.context = ConversationContext(history=[])
        self.is_running = True
        self.colors = {
            "user": "\033[94m",
            "assistant": "\033[92m",
            "system": "\033[93m",
            "error": "\033[91m",
            "reset": "\033[0m",
            "bold": "\033[1m",
        }
        logger.info("ConversationalRAGSystem initialized (LM Studio client ready)")

    def print_colored(self, text: str, color_key: str = "reset") -> None:
        color = self.colors.get(color_key, self.colors["reset"])
        print(f"{color}{text}{self.colors['reset']}")

    def display_welcome(self) -> None:
        welcome_msg = f"""
{self.colors['bold']}🤖 CEO RAG Chatbot - Console Interface (LM Studio){self.colors['reset']}
{self.colors['system']}{'='*60}{self.colors['reset']}

Welcome! Capabilities:

📄 {self.colors['assistant']}Document Search{self.colors['reset']}
📊 {self.colors['assistant']}Data Analysis{self.colors['reset']}
✉️  {self.colors['assistant']}Email Drafting{self.colors['reset']}
📧 {self.colors['assistant']}Email Analysis{self.colors['reset']}
💬 {self.colors['assistant']}General Chat{self.colors['reset']}

{self.colors['system']}Special Commands:{self.colors['reset']}
- {self.colors['user']}'help'{self.colors['reset']}   - Detailed help
- {self.colors['user']}'status'{self.colors['reset']} - System status
- {self.colors['user']}'history'{self.colors['reset']}- Conversation history
- {self.colors['user']}'clear'{self.colors['reset']}  - Clear history
- {self.colors['user']}'quit'/'exit'{self.colors['reset']} - Exit

{self.colors['system']}{'='*60}{self.colors['reset']}
"""
        print(welcome_msg)

    def display_help(self) -> None:
        help_text = f"""
{self.colors['bold']}📖 Detailed Help & Examples{self.colors['reset']}
{self.colors['system']}{'='*60}{self.colors['reset']}

{self.colors['bold']}📄 Document Search Examples:{self.colors['reset']}
• "Search for information about quarterly financial results"

{self.colors['bold']}📊 Data Analysis Examples:{self.colors['reset']}
• "Show me the top 10 companies by revenue"

{self.colors['bold']}✉️ Email Drafting Examples:{self.colors['reset']}
• "Draft an email to John about the quarterly review meeting"

{self.colors['bold']}📧 Email Analysis Examples:{self.colors['reset']}
• "Analyze this email: [paste email content]"

{self.colors['system']}{'='*60}{self.colors['reset']}
"""
        print(help_text)

    def display_system_status(self) -> None:
        try:
            self.print_colored("🔍 Checking system status...", "system")
            sources = self.retriever.get_available_sources()
            status_text = f"""
{self.colors['bold']}🖥️ System Status{self.colors['reset']}
{self.colors['system']}{'='*50}{self.colors['reset']}

{self.colors['bold']}✅ Components:{self.colors['reset']}
• Retriever: {self.colors['assistant']}Active{self.colors['reset']}
• LM Studio Client: {self.colors['assistant']}Connected{self.colors['reset']}
• Model: {self.colors['assistant']}{self.model_name}{self.colors['reset']}

{self.colors['bold']}📊 Available Data Sources:{self.colors['reset']}
"""
            print(status_text)
            sources_by_type = sources.get("sources_by_type", {})
            for search_type, info in sources_by_type.items():
                type_display = "📄 Vector Documents" if search_type == "vector" else "🗄️ Data Tables"
                self.print_colored(f"{type_display}:", "bold")
                print(f"  Count: {info['count']}")
                print(f"  Prefixes: {info['prefixes']}")
                if info.get("tables"):
                    print("  Available:")
                    for table in info["tables"][:5]:
                        rows = table.get("rows")
                        rows_str = f"{rows:,} rows" if isinstance(rows, int) else "unknown rows"
                        print(f"    • {table['table_name']} ({rows_str})")
                        if table.get("description"):
                            print(f"      {table['description']}")
                    if len(info["tables"]) > 5:
                        print(f"    ... and {len(info['tables']) - 5} more")
                print()
            search_params = sources.get("search_parameters", {})
            print(f"{self.colors['bold']}⚙️ Search Configuration:{self.colors['reset']}")
            print(f"  Vector Similarity Threshold: {search_params.get('vector_similarity_threshold', 0.7)}")
            print(f"  Max Vector Results: {search_params.get('max_vector_results', 10)}")
            print(f"  Max SQL Results: {search_params.get('max_sql_results', 50)}")
            print(f"\n{self.colors['bold']}💬 Current Session:{self.colors['reset']}")
            print(f"  Session Started: {self.context.session_start.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  Conversation Exchanges: {len(self.context.history)}")
            print(f"  Last Query Type: {self.context.last_query_type.value if self.context.last_query_type else 'None'}")
            print(f"\n{self.colors['system']}{'='*50}{self.colors['reset']}")
        except Exception as e:
            self.print_colored(f"❌ Error checking system status: {str(e)}", "error")

    def display_conversation_history(self) -> None:
        if not self.context.history:
            self.print_colored("📝 No conversation history yet.", "system")
            return
        print(f"\n{self.colors['bold']}📝 Conversation History{self.colors['reset']}")
        print(f"{self.colors['system']}{'='*50}{self.colors['reset']}")
        for i, exchange in enumerate(self.context.history[-5:], 1):
            timestamp = datetime.fromisoformat(exchange["timestamp"]).strftime("%H:%M:%S")
            query_type = exchange["query_type"]
            print(f"\n{self.colors['system']}{i}. [{timestamp}] ({query_type}){self.colors['reset']}")
            print(f"{self.colors['user']}User:{self.colors['reset']} {exchange['user_query']}")
            response = exchange["assistant_response"]
            if len(response) > 300:
                response = response[:300] + "..."
            print(f"{self.colors['assistant']}Assistant:{self.colors['reset']} {response}")
        print(f"\n{self.colors['system']}{'='*50}{self.colors['reset']}")

    def handle_document_search(self, query: str, extracted_info: Dict[str, Any]) -> str:
        try:
            self.print_colored("🔍 Searching through documents...", "system")
            result = self.retriever.hybrid_retrieve(query, force_search_type="vector")
            if not result.vector_chunks:
                return f"""
🔍 Document Search Results

No relevant information found for your query: "{query}"

Possible reasons:
• Not present in uploaded documents
• Try different keywords/phrases
• Documents may not be processed yet
"""
            self.print_colored("🧠 Analyzing retrieved documents...", "system")
            conversation_context = self.context.get_context_string(2)
            analyzed_response = self.response_analyzer.analyze_document_chunks(
                query=query,
                chunks=result.vector_chunks,
                conversation_context=conversation_context,
            )
            unique_docs = {chunk['file_name'] for chunk in result.vector_chunks}
            retrieval_summary = f"""
🔍 **Document Search Summary:**
• Found {len(result.vector_chunks)} relevant sections across {len(unique_docs)} documents
• Search completed in {result.metadata.get('total_retrieval_time', 0):.2f} seconds
• Sources: {', '.join(sorted(unique_docs))}

📊 **AI Analysis:**
{analyzed_response}
"""
            return retrieval_summary
        except Exception as e:
            logger.error(f"Error in document search: {str(e)}")
            return f"❌ Error searching documents: {str(e)}\n\nPlease try again."

    def handle_data_query(self, query: str, extracted_info: Dict[str, Any]) -> str:
        try:
            self.print_colored("📊 Analyzing data...", "system")
            result = self.retriever.hybrid_retrieve(query, force_search_type="sql")
            if not result.sql_results:
                return f"""
📊 Data Analysis Results

No data found for your query: "{query}"

Try rephrasing or check available tables.
"""
            self.print_colored("🧠 Analyzing query results...", "system")
            conversation_context = self.context.get_context_string(2)
            first_result = result.sql_results[0]
            generated_sql = first_result.get("generated_sql", "SQL query not available")
            selected_table = first_result.get("selected_table", "Unknown table")
            analyzed_response = self.response_analyzer.analyze_data_results(
                query=query,
                results=result.sql_results,
                generated_sql=generated_sql,
                conversation_context=conversation_context,
            )
            query_summary = f"""
📊 **Data Query Summary:**
• Source Table: {selected_table}
• Records Found: {len(result.sql_results)}
• Query Time: {result.metadata.get('total_retrieval_time', 0):.2f} seconds

🧠 **AI Analysis:**
{analyzed_response}
"""
            return query_summary
        except Exception as e:
            logger.error(f"Error in data query: {str(e)}")
            return f"❌ Error analyzing data: {str(e)}\n\nPlease try again."

    def handle_hybrid_search(self, query: str, extracted_info: Dict[str, Any]) -> str:
        try:
            self.print_colored("🔍 Performing comprehensive search...", "system")
            result = self.retriever.hybrid_retrieve(query)
            has_vector_results = len(result.vector_chunks) > 0
            has_sql_results = len(result.sql_results) > 0
            if not has_vector_results and not has_sql_results:
                return f"""
🔍 Comprehensive Search Results

No relevant information found for your query: "{query}"
"""
            self.print_colored("🧠 Synthesizing all available information...", "system")
            conversation_context = self.context.get_context_string(2)
            synthesized_response = self.response_analyzer.synthesize_hybrid_results(
                query=query,
                vector_chunks=result.vector_chunks,
                sql_results=result.sql_results,
                conversation_context=conversation_context,
            )
            search_summary = f"""
🔍 **Comprehensive Search Summary:**
• Document chunks found: {len(result.vector_chunks)}
• Data records found: {len(result.sql_results)}
• Total search time: {result.metadata.get('total_retrieval_time', 0):.2f} seconds
• Search strategy: {result.metadata.get('query_type', 'hybrid')}

🧠 **Comprehensive Analysis:**
{synthesized_response}

📊 **Search Metadata:**
• Selected tables: {result.metadata.get('selected_tables', {})}
• Similarity threshold: {result.metadata.get('similarity_threshold', 0.25)}
"""
            return search_summary
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            return f"❌ Error in comprehensive search: {str(e)}\n\nPlease try again."

    def handle_email_draft(self, query: str, extracted_info: Dict[str, Any]) -> str:
        try:
            self.print_colored("✉️ Drafting email...", "system")
            context = ""
            if self.context.history:
                recent_exchanges = self.context.history[-3:]
                for exchange in recent_exchanges:
                    if exchange["query_type"] in [
                        QueryType.DOCUMENT_SEARCH.value,
                        QueryType.DATA_QUERY.value,
                    ]:
                        context += f"Recent context: {exchange['user_query']}\n"
            result = self.email_processor.draft_email(
                recipient=extracted_info.get("recipient", ""),
                subject=extracted_info.get("subject", ""),
                content_request=extracted_info.get("content_request", query),
                tone=extracted_info.get("tone", "professional"),
                context=context,
            )
            if not result["success"]:
                return f"❌ Error drafting email: {result.get('error', 'Unknown error')}"
            email_data = result["email"]
            response_parts = [
                f"✉️ Email Draft Complete",
                f"",
                f"📧 **To:** {email_data['recipient'] or '[Specify recipient]'}",
                f"📧 **Subject:** {email_data['subject'] or '[Specify subject]'}",
                f"📧 **Tone:** {email_data['tone'].title()}",
                f"",
                f"📝 **Email Content:**",
                f"{'='*50}",
                f"{email_data['body']}",
                f"{'='*50}",
                f"",
                f"💡 **Usage Notes:**",
                f"• Review and customize as needed",
                f"• Verify recipient email address before sending",
                f"⏱️ Generated at {result['metadata']['generated_at']}",
            ]
            if context:
                response_parts.append("• Incorporated context from recent conversation")
            return "\n".join(response_parts)
        except Exception as e:
            logger.error(f"Error drafting email: {str(e)}")
            return f"❌ Error drafting email: {str(e)}"

    def handle_email_analysis(self, query: str, extracted_info: Dict[str, Any]) -> str:
        try:
            self.print_colored("📧 Analyzing email...", "system")
            email_content = extracted_info.get("email_content", query)
            sender_match = re.search(r"from:?\s*([^\n]+)", email_content, re.IGNORECASE)
            date_match = re.search(r"(?:date|received):?\s*([^\n]+)", email_content, re.IGNORECASE)
            sender = sender_match.group(1).strip() if sender_match else ""
            received_date = date_match.group(1).strip() if date_match else ""
            result = self.email_processor.analyze_email(
                email_content=email_content,
                sender=sender,
                received_date=received_date,
            )
            if not result["success"]:
                return f"❌ Error analyzing email: {result.get('error', 'Unknown error')}"
            analysis = result["analysis"]
            urgency_icons = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🟠", "URGENT": "🔴"}
            urgency_icon = urgency_icons.get(analysis["urgency_level"], "🟡")
            response_parts = [
                f"📧 Email Analysis Complete",
                f"",
                f"{urgency_icon} **Urgency Level:** {analysis['urgency_level']} (Priority: {analysis['priority_score']}/10)",
                f"📂 **Category:** {analysis['category'].title()}",
                f"👤 **From:** {analysis['sender'] or 'Unknown'}",
                f"📅 **Received:** {analysis['received_date'] or 'Not specified'}",
                f"📊 **Content:** {analysis['word_count']} words, {analysis['content_length']} characters",
                f"",
                f"📋 **Detailed Analysis:**",
                f"{'='*60}",
                f"{analysis['full_analysis']}",
                f"{'='*60}",
                f"",
                f"⏱️ Analysis completed at {datetime.fromisoformat(result['metadata']['analyzed_at']).strftime('%H:%M:%S')}",
            ]
            return "\n".join(response_parts)
        except Exception as e:
            logger.error(f"Error analyzing email: {str(e)}")
            return f"❌ Error analyzing email: {str(e)}"

    def handle_general_chat(self, query: str) -> str:
        try:
            sources = self.retriever.get_available_sources()
            context_info = (
                f"\nAvailable capabilities:\n"
                f"- {sources.get('sources_by_type', {}).get('vector', {}).get('count', 0)} document sources\n"
                f"- {sources.get('sources_by_type', {}).get('sql', {}).get('count', 0)} data tables\n"
                f"- Email drafting and analysis\n"
                f"- Conversation history: {len(self.context.history)} exchanges\n"
            )
            conversation_context = self.context.get_context_string(3)
            system_prompt = (
                "You are a helpful AI assistant for a CEO's RAG chatbot system. "
                "You have access to document search, data analysis, and email processing."
                f"\n\n{context_info}\n\nPrevious conversation context:\n{conversation_context}\n\n"
                "Be professional, helpful, and concise. Guide the user to use capabilities effectively."
            )
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                max_tokens=500,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error in general chat: {str(e)}")
            return (
                "I'm here to help! You can ask me to search documents, analyze data, draft emails, or just chat. "
                "What would you like to do?"
            )

    def _should_attempt_hybrid_search(self, query: str, initial_response: str) -> bool:
        hybrid_indicators = [
            "compare", "analysis", "comprehensive", "complete picture", "full report",
            "detailed", "thorough", "both", "also", "relationship", "correlation",
            "trend", "pattern",
        ]
        query_lower = query.lower()
        has_hybrid_indicators = any(ind in query_lower for ind in hybrid_indicators)
        response_lower = initial_response.lower()
        suggests_limited = any(
            p in response_lower for p in ["couldn't find", "no data", "not available", "no relevant", "didn't find"]
        )
        business_terms = [
            "performance", "revenue", "profit", "financial", "quarterly", "annual",
            "growth", "market", "strategy", "budget", "forecast",
        ]
        has_business_context = any(term in query_lower for term in business_terms)
        return has_hybrid_indicators or suggests_limited or has_business_context

    def process_query(self, query: str) -> str:
        start_time = time.time()
        try:
            context_str = self.context.get_context_string(3)
            query_type, extracted_info = self.query_classifier.classify_query(query, context_str)
            logger.info(f"Query classified as: {query_type.value}")

            if query_type == QueryType.DOCUMENT_SEARCH:
                response = self.handle_document_search(query, extracted_info)
            elif query_type == QueryType.DATA_QUERY:
                response = self.handle_data_query(query, extracted_info)
            elif query_type == QueryType.EMAIL_DRAFT:
                response = self.handle_email_draft(query, extracted_info)
            elif query_type == QueryType.EMAIL_ANALYZE:
                response = self.handle_email_analysis(query, extracted_info)
            elif query_type == QueryType.HELP:
                self.display_help()
                return ""
            elif query_type == QueryType.SYSTEM_STATUS:
                self.display_system_status()
                return ""
            else:
                response = self.handle_general_chat(query)

            if query_type in [QueryType.DOCUMENT_SEARCH, QueryType.DATA_QUERY]:
                if self._should_attempt_hybrid_search(query, response):
                    self.print_colored("🔄 Attempting comprehensive search for better results...", "system")
                    hybrid_response = self.handle_hybrid_search(query, extracted_info)
                    if len(hybrid_response) > len(response):
                        response = hybrid_response

            processing_time = time.time() - start_time
            metadata = {"processing_time": processing_time, "query_type": query_type.value, "extracted_info": extracted_info}
            self.context.add_exchange(query, response, query_type, metadata)
            return response
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            error_response = (
                f"❌ I encountered an error processing your request: {str(e)}\n\n"
                "Please try again or type 'help' for assistance."
            )
            self.context.add_exchange(query, error_response, QueryType.GENERAL_CHAT, {"error": str(e)})
            return error_response

    def run_console_loop(self) -> None:
        print("\n")
        self.display_welcome()
        while self.is_running:
            try:
                print(f"\n{self.colors['user']}You: {self.colors['reset']}", end="")
                user_input = input().strip()
                if not user_input:
                    continue
                user_lower = user_input.lower()
                if user_lower in ["quit", "exit", "q"]:
                    self.print_colored("👋 Goodbye! Thank you for using the CEO RAG Chatbot.", "system")
                    self.is_running = False
                    break
                elif user_lower == "help":
                    self.display_help()
                    continue
                elif user_lower == "status":
                    self.display_system_status()
                    continue
                elif user_lower == "history":
                    self.display_conversation_history()
                    continue
                elif user_lower == "clear":
                    self.context.history.clear()
                    self.print_colored("🗑️ Conversation history cleared.", "system")
                    continue
                print(f"\n{self.colors['assistant']}Assistant: {self.colors['reset']}")
                response = self.process_query(user_input)
                if response:
                    print(response)
            except KeyboardInterrupt:
                self.print_colored("\n\n👋 Goodbye! (Interrupted by user)", "system")
                self.is_running = False
                break
            except EOFError:
                self.print_colored("\n\n👋 Goodbye! (End of input)", "system")
                self.is_running = False
                break
            except Exception as e:
                self.print_colored(f"❌ Unexpected error: {str(e)}", "error")
                logger.error(f"Unexpected error in console loop: {str(e)}")

    def close(self) -> None:
        try:
            self.retriever.close()
        except Exception:
            pass
        logger.info("ConversationalRAGSystem closed")


def create_console_rag_system(
    openai_api_key: str = None, db_config: Dict[str, str] = None
) -> ConversationalRAGSystem:
    return ConversationalRAGSystem(openai_api_key=openai_api_key, db_config=db_config)


def main() -> int:
    print("🚀 Starting CEO RAG Chatbot Console Interface (LM Studio)...")
    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        system = create_console_rag_system()
        system.run_console_loop()
    except KeyboardInterrupt:
        print("\n👋 Application interrupted by user.")
    except Exception as e:
        print(f"❌ Failed to start application: {str(e)}")
        logger.error(f"Application startup failed: {str(e)}")
        return 1
    finally:
        try:
            system.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(main())

