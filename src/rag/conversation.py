#!/usr/bin/env python3
"""
Console-based Conversational RAG System with Email Functionality

This version:
- Uses LM Studio via an OpenAI-compatible API (http://localhost:1234/v1)
- Initializes ONE shared OpenAI client and passes it everywhere
- Forces the retriever to use 'Snowflake/snowflake-arctic-embed-l' for embeddings
- Ensures NO calls go to api.openai.com (sets base_url for the global openai module)
- Automatically uses whichever chat model is currently exposed by LM Studio
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

# ---------------------------------------------------------------------
# LM Studio / OpenAI-compatible client setup (MUST be done BEFORE importing retriever)
# ---------------------------------------------------------------------
LMSTUDIO_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1")
LMSTUDIO_API_KEY = os.getenv("OPENAI_API_KEY", "not-needed")
os.environ.setdefault("OPENAI_BASE_URL", LMSTUDIO_BASE_URL)   # new SDK env
os.environ.setdefault("OPENAI_API_BASE", LMSTUDIO_BASE_URL)   # old SDK env
os.environ.setdefault("OPENAI_API_KEY", LMSTUDIO_API_KEY)

# Set the global module config so ANY code using `import openai` + global namespace
# (e.g., openai.chat.completions.create) will also hit LM Studio instead of api.openai.com.
import openai  # type: ignore
try:
    # For openai>=1.0
    openai.base_url = LMSTUDIO_BASE_URL  # type: ignore[attr-defined]
except Exception:
    pass
try:
    # For openai<1.0
    openai.api_base = LMSTUDIO_BASE_URL  # type: ignore[attr-defined]
except Exception:
    pass
openai.api_key = LMSTUDIO_API_KEY

# Also create a modern client instance to use in THIS file.
from openai import OpenAI  # pip install openai>=1.0.0
client = OpenAI(base_url=LMSTUDIO_BASE_URL, api_key=LMSTUDIO_API_KEY)


def auto_detect_lmstudio_chat_model(lm_client: OpenAI) -> str:
    """
    Pick the model that LM Studio is currently exposing via its OpenAI-compatible API.

    Priority:
    1) If LMSTUDIO_CHAT_MODEL env var is set, use that (manual override).
    2) Otherwise, call /v1/models and take the first model id.
    3) If anything fails, fall back to a known default id.

    In LM Studio, this will usually be "whatever is currently loaded" because
    the server normally exposes a single active model.
    """
    env_model = os.getenv("LMSTUDIO_CHAT_MODEL")
    if env_model:
        return env_model

    logger = logging.getLogger(__name__)

    try:
        models = lm_client.models.list()
        ids = [m.id for m in getattr(models, "data", [])]
        if ids:
            # LM Studio typically returns the active model here.
            return ids[0]
        logger.warning("LM Studio /v1/models returned no model ids; falling back to default model id.")
    except Exception as e:
        logger.warning(f"Could not auto-detect LM Studio model, falling back to default: {e}")

    # Fallback: keep your previous default so behavior is consistent
    return "openai/gpt-oss-20b"


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
# Ensure log directory exists BEFORE configuring logging with FileHandler
LOG_DIR = Path("data/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "conversation.log"),
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
        parts: List[str] = []
        for ex in recent:
            parts.append(f"User: {ex['user_query']}")
            resp = ex["assistant_response"]
            if len(resp) > 200:
                resp = resp[:200] + "..."
            parts.append(f"Assistant: {resp}")
        return "\n".join(parts)


# ---------------------------------------------------------------------
# Import retriever (uses global `openai` module; base_url/api_key already set)
# ---------------------------------------------------------------------
try:
    from retriever import HybridRetriever, RetrievalResult, create_retriever
except ImportError:
    print("Error: Could not import retriever module. Make sure retriever.py is in the same directory.")
    sys.exit(1)


# ---------------- Email + LLM helpers (LM Studio) ----------------

class EmailProcessor:
    """Drafts and analyzes emails using the LM Studio client."""
    def __init__(self, lm_client: OpenAI) -> None:
        self.client = lm_client
        # Automatically detect active LM Studio chat model
        self.chat_model = auto_detect_lmstudio_chat_model(self.client)

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

            resp = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert business email writer specializing "
                            "in executive communications."
                        ),
                    },
                    {"role": "user", "content": full_prompt},
                ],
                max_tokens=1000,
                temperature=0.3,
            )
            email_content = resp.choices[0].message.content.strip()

            subject_match = re.search(r"Subject:\s*(.+?)(?:\n|$)", email_content, re.IGNORECASE)
            recipient_match = re.search(r"(?:To|Recipient):\s*(.+?)(?:\n|$)", email_content, re.IGNORECASE)
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
                    "model_used": self.chat_model,
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
""".strip()

            resp = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert executive assistant specializing in email "
                            "analysis and business communications."
                        ),
                    },
                    {"role": "user", "content": analysis_prompt},
                ],
                max_tokens=800,
                temperature=0.1,
            )
            analysis_content = resp.choices[0].message.content.strip()

            urgency_match = re.search(
                r"URGENCY LEVEL:\s*(LOW|MEDIUM|HIGH|URGENT)",
                analysis_content,
                re.IGNORECASE,
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
                    "model_used": self.chat_model,
                },
            }
        except Exception as e:
            logger.error(f"Error analyzing email: {str(e)}")
            return {"success": False, "error": str(e), "analysis": None}


class QueryClassifier:
    """Classifies queries using LM Studio client (no external calls)."""
    def __init__(self, lm_client: OpenAI) -> None:
        self.client = lm_client
        self.chat_model = auto_detect_lmstudio_chat_model(self.client)

    def classify_query(self, query: str, context: str = "") -> Tuple[QueryType, Dict[str, Any]]:
        try:
            classification_prompt = f"""
Analyze this user query and classify it into one of these categories:

1. DOCUMENT_SEARCH
2. DATA_QUERY
3. EMAIL_DRAFT
4. EMAIL_ANALYZE
5. HELP
6. SYSTEM_STATUS
7. GENERAL_CHAT

CONTEXT:
{context}

USER QUERY:
{query}

Respond with the category name only. If you can, also extract quick params such as recipient/subject for email, or search terms for docs.
""".strip()

            resp = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": "You are a query classifier."},
                    {"role": "user", "content": classification_prompt},
                ],
                max_tokens=100,
                temperature=0.0,
            )
            raw = resp.choices[0].message.content.strip().upper()

            query_type: Optional[QueryType] = None
            for qt in QueryType:
                if qt.value.upper() in raw or qt.name in raw:
                    query_type = qt
                    break

            if not query_type:
                # Simple fallback heuristic
                ql = query.lower()
                if any(w in ql for w in ["draft", "write", "compose", "email to"]):
                    query_type = QueryType.EMAIL_DRAFT
                elif any(w in ql for w in ["analyze email", "read email", "email from"]):
                    query_type = QueryType.EMAIL_ANALYZE
                elif any(w in ql for w in ["search", "find", "document", "report", "pdf"]):
                    query_type = QueryType.DOCUMENT_SEARCH
                elif any(w in ql for w in ["data", "statistics", "ranking", "top"]):
                    query_type = QueryType.DATA_QUERY
                elif any(w in ql for w in ["help", "how to", "what can"]):
                    query_type = QueryType.HELP
                elif any(w in ql for w in ["status", "available", "sources", "tables"]):
                    query_type = QueryType.SYSTEM_STATUS
                else:
                    query_type = QueryType.GENERAL_CHAT

            extracted = self._extract_query_info(query, query_type)
            return query_type, extracted

        except Exception as e:
            logger.error(f"Error classifying query: {str(e)}")
            # Simple fallback
            ql = query.lower()
            if any(w in ql for w in ["draft", "write", "compose", "email to"]):
                qt = QueryType.EMAIL_DRAFT
            elif any(w in ql for w in ["analyze email", "read email", "email from"]):
                qt = QueryType.EMAIL_ANALYZE
            elif any(w in ql for w in ["search", "find", "document", "report", "pdf"]):
                qt = QueryType.DOCUMENT_SEARCH
            elif any(w in ql for w in ["data", "statistics", "ranking", "top"]):
                qt = QueryType.DATA_QUERY
            elif any(w in ql for w in ["help", "how to", "what can"]):
                qt = QueryType.HELP
            elif any(w in ql for w in ["status", "available", "sources", "tables"]):
                qt = QueryType.SYSTEM_STATUS
            else:
                qt = QueryType.GENERAL_CHAT
            return qt, self._extract_query_info(query, qt)

    def _extract_query_info(self, query: str, query_type: QueryType) -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        if query_type == QueryType.EMAIL_DRAFT:
            recipient_match = re.search(
                r"(?:to|email)\s+(.+?)(?:\s+about|\s+regarding|$)",
                query,
                re.IGNORECASE,
            )
            subject_match = re.search(
                r"(?:about|regarding|subject)\s+(.+)",
                query,
                re.IGNORECASE,
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
    """Synthesizes results using LM Studio client."""
    def __init__(self, lm_client: OpenAI) -> None:
        self.client = lm_client
        self.chat_model = auto_detect_lmstudio_chat_model(self.client)

    def analyze_document_chunks(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        conversation_context: str = "",
    ) -> str:
        try:
            chunk_info = []
            for i, chunk in enumerate(chunks[:5], 1):
                sim_val = chunk.get("similarity_score", 0.0)
                try:
                    sim_float = float(sim_val)
                except (TypeError, ValueError):
                    sim_float = 0.0
                chunk_info.append({
                    "index": i,
                    "source": chunk.get("file_name", "Unknown document"),
                    "page": chunk.get("page_number", 1),
                    "similarity": sim_float,
                    "content": chunk.get("text", "")[:1500],
                })

            synthesis_prompt = (
                "You are an expert business analyst reviewing document search results for a CEO.\n"
                f"ORIGINAL QUERY: {query}\n\n"
                f"CONVERSATION CONTEXT:\n{conversation_context}\n\n"
                "RETRIEVED INFORMATION:\n"
            )
            for c in chunk_info:
                sim_display = round(c["similarity"], 3)
                synthesis_prompt += (
                    f"\nDocument {c['index']}: {c['source']} "
                    f"(Page {c['page']}, Relevance: {sim_display})\n"
                    f"Content: {c['content']}\n---\n"
                )
            synthesis_prompt += """
ANALYSIS INSTRUCTIONS:
1) Synthesize information from all sources to answer the query directly.
2) Highlight key findings, themes, and contradictions.
3) Provide specific details with doc references.
4) Use clear sections and bullet points.

RESPONSE FORMAT:
- Direct answer
- Detailed analysis with references
- Key insights and implications
- Recommendations (if applicable)
"""

            resp = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": "You are an executive document analyst."},
                    {"role": "user", "content": synthesis_prompt},
                ],
                max_tokens=1200,
                temperature=0.2,
            )
            synthesized = resp.choices[0].message.content.strip()

            unique_sources = {c["source"] for c in chunk_info}
            if chunks:
                total_sim = 0.0
                for ch in chunks:
                    sim_val = ch.get("similarity_score", 0.0)
                    try:
                        total_sim += float(sim_val)
                    except (TypeError, ValueError):
                        total_sim += 0.0
                avg_sim = total_sim / max(1, len(chunks))
            else:
                avg_sim = 0.0
            avg_sim_display = round(avg_sim, 3)

            footer = (
                "\n\n📊 Analysis Metadata:\n"
                f"• Sources analyzed: {len(unique_sources)} ({', '.join(sorted(unique_sources))})\n"
                f"• Chunks considered: {len(chunks)} total, {len(chunk_info)} in detail\n"
                f"• Avg relevance: {avg_sim_display}\n"
                f"• Model: {self.chat_model}"
            )
            return synthesized + footer
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
                "sample": results[:5],
            }

            filtered_sample = [
                {
                    k: v
                    for k, v in row.items()
                    if k not in ["source_type", "generated_sql", "selected_table"]
                }
                for row in results[:5]
            ]

            prompt = f"""You are a senior business intelligence analyst.
ORIGINAL QUERY: {query}

CONTEXT:
{conversation_context}

SQL EXECUTED:
{generated_sql}

DATA SUMMARY:
- Rows: {data_summary['row_count']}
- Columns: {', '.join([c for c in data_summary['columns'] if c not in ['source_type','generated_sql','selected_table']])}

SAMPLE:
{json.dumps(filtered_sample, indent=2)}

INSTRUCTIONS:
- Give an executive summary
- Key findings with numbers
- Detailed analysis and implications
- Recommendations if useful
"""

            resp = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": "You are a data analyst for executives."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1200,
                temperature=0.1,
            )
            analyzed = resp.choices[0].message.content.strip()

            meta = (
                "\n\n🔧 Technical:\n"
                f"• Records: {data_summary['row_count']}\n"
                f"• Source table: {results[0].get('selected_table', 'Unknown')}\n"
                f"• Model: {self.chat_model}\n"
                f"• SQL (truncated): {generated_sql[:100]}{'...' if len(generated_sql) > 100 else ''}"
            )
            return analyzed + meta
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
            prompt = (
                "You are a senior executive advisor combining documents and data.\n\n"
                f"QUERY: {query}\n\n"
                f"CONTEXT:\n{conversation_context}\n\n"
                "DOCUMENTS:\n"
            )
            if vector_chunks:
                for i, ch in enumerate(vector_chunks[:3], 1):
                    sim_val = ch.get("similarity_score", 0.0)
                    try:
                        sim_float = float(sim_val)
                    except (TypeError, ValueError):
                        sim_float = 0.0
                    sim_display = round(sim_float, 3)
                    prompt += (
                        f"\nDoc {i}: {ch.get('file_name', 'Unknown')} "
                        f"(p{ch.get('page_number', 1)}), rel={sim_display}\n"
                        f"Excerpt: {ch.get('text', '')[:800]}\n---\n"
                    )
            else:
                prompt += "No relevant documents.\n"

            prompt += "\nDATA:\n"
            if sql_results:
                prompt += (
                    f"Rows: {len(sql_results)}\n"
                    f"Sample: {json.dumps(sql_results[:3], indent=2)}\n"
                )
            else:
                prompt += "No structured data found.\n"

            prompt += """
INSTRUCTIONS:
- Executive summary
- Document insights
- Data analysis
- Cross-analysis (how they support/contradict)
- Strategic implications
- Recommended next steps
"""

            resp = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You synthesize multiple sources for executives.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1500,
                temperature=0.2,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error synthesizing hybrid results: {str(e)}")
            return f"❌ Error synthesizing results: {str(e)}"


# ---------------- Main console app ----------------

class ConversationalRAGSystem:
    def __init__(self, lm_client: OpenAI, db_config: Dict[str, str] = None) -> None:
        self.client = lm_client

        # IMPORTANT: Force the retriever to use Snowflake Arctic embeddings
        embedding_model_name = os.getenv("EMBEDDING_MODEL", "Snowflake/snowflake-arctic-embed-l")

        try:
            # NOTE: Do NOT pass unknown kwargs (like openai_client). retriever.create_retriever
            # signature is (db_config=None, openai_api_key=None, embedding_model=..., custom_prefixes=None)
            self.retriever = create_retriever(
                db_config=db_config,
                embedding_model=embedding_model_name,
                # Optional: pass a key (LM Studio ignores it). retriever may set openai.api_key again.
                openai_api_key=LMSTUDIO_API_KEY,
            )
            # Tune search parameters
            self.retriever.update_search_parameters(
                similarity_threshold=0.15,
                max_vector_results=10,
                max_sql_results=50,
            )
        except Exception as e:
            logger.error(f"Failed to initialize retriever: {str(e)}")
            raise

        self.email_processor = EmailProcessor(self.client)
        self.query_classifier = QueryClassifier(self.client)
        self.response_analyzer = ResponseAnalyzer(self.client)
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
        logger.info(
            "ConversationalRAGSystem initialized "
            "(LM Studio + Snowflake embeddings, auto chat model)"
        )

    def print_colored(self, text: str, color_key: str = "reset") -> None:
        color = self.colors.get(color_key, self.colors["reset"])
        print(f"{color}{text}{self.colors['reset']}")

    def display_welcome(self) -> None:
        chat_model = self.query_classifier.chat_model
        welcome_msg = f"""
{self.colors['bold']}🤖 CEO RAG Chatbot - Console Interface{self.colors['reset']}
{self.colors['system']}{'='*60}{self.colors['reset']}

Using LM Studio at {LMSTUDIO_BASE_URL}
Chat model (auto-detected): {chat_model}
Embedding model: Snowflake/snowflake-arctic-embed-l

📄 Document Search  •  📊 Data Analysis  •  ✉️ Email Drafting  •  📧 Email Analysis  •  💬 General Chat

Commands: 'help', 'status', 'history', 'clear', 'quit'
"""
        print(welcome_msg)

    def display_help(self) -> None:
        help_text = f"""
{self.colors['bold']}📖 Help & Examples{self.colors['reset']}
{self.colors['system']}{'='*60}{self.colors['reset']}

Document Search: "Search marketing strategy in reports"
Data Analysis:   "Top 10 companies by revenue"
Email Draft:     "Draft an email to Jane about the Q4 plan"
Email Analyze:   "Analyze this email: ..."

Status:          "status"
History:         "history"
Clear:           "clear"
Quit:            "quit"
"""
        print(help_text)

    def display_system_status(self) -> None:
        try:
            self.print_colored("🔍 Checking system status...", "system")
            sources = self.retriever.get_available_sources()
            status_text = f"""
{self.colors['bold']}🖥️ System Status{self.colors['reset']}
{self.colors['system']}{'='*50}{self.colors['reset']}

✅ Retriever: Active
✅ LM Studio: {LMSTUDIO_BASE_URL}
✅ Chat model: {self.query_classifier.chat_model}
✅ Embeddings: Snowflake/snowflake-arctic-embed-l
"""
            print(status_text)
            sources_by_type = sources.get("sources_by_type", {})
            for search_type, info in sources_by_type.items():
                type_display = "📄 Vector Documents" if search_type == "vector" else "🗄️ Data Tables"
                self.print_colored(f"{type_display}:", "bold")
                print(f"  Count: {info.get('count', 0)}")
                print(f"  Prefixes: {info.get('prefixes')}")
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
            print(
                f"  Vector Similarity Threshold: "
                f"{search_params.get('vector_similarity_threshold', 0.7)}"
            )
            print(f"  Max Vector Results: {search_params.get('max_vector_results', 10)}")
            print(f"  Max SQL Results: {search_params.get('max_sql_results', 50)}")
            print(f"\n{self.colors['bold']}💬 Session:{self.colors['reset']}")
            print(f"  Started: {self.context.session_start.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  Exchanges: {len(self.context.history)}")
            print(
                f"  Last Query: "
                f"{self.context.last_query_type.value if self.context.last_query_type else 'None'}"
            )
            print(f"\n{self.colors['system']}{'='*50}{self.colors['reset']}")
        except Exception as e:
            self.print_colored(f"❌ Error checking system status: {str(e)}", "error")

    def display_conversation_history(self) -> None:
        if not self.context.history:
            self.print_colored("📝 No conversation history yet.", "system")
            return
        print(f"\n{self.colors['bold']}📝 Conversation History{self.colors['reset']}")
        print(f"{self.colors['system']}{'='*50}{self.colors['reset']}")
        for i, ex in enumerate(self.context.history[-5:], 1):
            timestamp = datetime.fromisoformat(ex["timestamp"]).strftime("%H:%M:%S")
            qtype = ex["query_type"]
            print(f"\n{self.colors['system']}{i}. [{timestamp}] ({qtype}){self.colors['reset']}")
            print(f"{self.colors['user']}User:{self.colors['reset']} {ex['user_query']}")
            resp = ex["assistant_response"]
            if len(resp) > 300:
                resp = resp[:300] + "..."
            print(f"{self.colors['assistant']}Assistant:{self.colors['reset']} {resp}")
        print(f"\n{self.colors['system']}{'='*50}{self.colors['reset']}")

    # --- Handlers ---

    def handle_document_search(self, query: str, extracted_info: Dict[str, Any]) -> str:
        try:
            self.print_colored("🔍 Searching documents...", "system")
            result = self.retriever.hybrid_retrieve(query, force_search_type="vector")
            if not result.vector_chunks:
                return (
                    "🔍 No relevant document sections found for your query.\n"
                    "Try different keywords or check if documents were indexed."
                )
            self.print_colored("🧠 Analyzing retrieved documents...", "system")
            analyzed = self.response_analyzer.analyze_document_chunks(
                query=query,
                chunks=result.vector_chunks,
                conversation_context=self.context.get_context_string(2),
            )
            unique_docs = {ch.get("file_name") for ch in result.vector_chunks}
            return (
                "🔍 **Document Search Summary**\n"
                f"• Sections: {len(result.vector_chunks)} across {len(unique_docs)} documents\n"
                f"• Time: {result.metadata.get('total_retrieval_time', 0):.2f}s\n\n"
                f"📊 **AI Analysis**\n{analyzed}"
            )
        except Exception as e:
            logger.error(f"Error in document search: {str(e)}")
            return f"❌ Error searching documents: {str(e)}"

    def handle_data_query(self, query: str, extracted_info: Dict[str, Any]) -> str:
        try:
            self.print_colored("📊 Running data query...", "system")
            result = self.retriever.hybrid_retrieve(query, force_search_type="sql")
            if not result.sql_results:
                return "📊 No data found for your query."
            self.print_colored("🧠 Analyzing data...", "system")
            first = result.sql_results[0]
            generated_sql = first.get("generated_sql", "SQL not available")
            analyzed = self.response_analyzer.analyze_data_results(
                query=query,
                results=result.sql_results,
                generated_sql=generated_sql,
                conversation_context=self.context.get_context_string(2),
            )
            return (
                "📊 **Data Query Summary**\n"
                f"• Rows: {len(result.sql_results)}\n"
                f"• Time: {result.metadata.get('total_retrieval_time', 0):.2f}s\n\n"
                f"🧠 **AI Analysis**\n{analyzed}"
            )
        except Exception as e:
            logger.error(f"Error in data query: {str(e)}")
            return f"❌ Error analyzing data: {str(e)}"

    def handle_hybrid_search(self, query: str, extracted_info: Dict[str, Any]) -> str:
        try:
            self.print_colored("🔍 Comprehensive search (docs + data)...", "system")
            result = self.retriever.hybrid_retrieve(query)
            if not result.vector_chunks and not result.sql_results:
                return "🔍 No results in documents or data."
            self.print_colored("🧠 Synthesizing results...", "system")
            synthesized = self.response_analyzer.synthesize_hybrid_results(
                query=query,
                vector_chunks=result.vector_chunks,
                sql_results=result.sql_results,
                conversation_context=self.context.get_context_string(2),
            )
            return (
                "🔍 **Comprehensive Search Summary**\n"
                f"• Doc chunks: {len(result.vector_chunks)}\n"
                f"• Data rows: {len(result.sql_results)}\n"
                f"• Time: {result.metadata.get('total_retrieval_time', 0):.2f}s\n\n"
                f"{synthesized}"
            )
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            return f"❌ Error in comprehensive search: {str(e)}"

    def handle_email_draft(self, query: str, extracted_info: Dict[str, Any]) -> str:
        try:
            self.print_colored("✉️ Drafting email...", "system")
            context = self.context.get_context_string(2)
            result = self.email_processor.draft_email(
                recipient=extracted_info.get("recipient", ""),
                subject=extracted_info.get("subject", ""),
                content_request=extracted_info.get("content_request", query),
                tone=extracted_info.get("tone", "professional"),
                context=context,
            )
            if not result["success"]:
                return f"❌ Error drafting email: {result.get('error', 'unknown')}"
            email = result["email"]
            return (
                "✉️ Email Draft\n\n"
                f"To: {email['recipient'] or '[Specify]'}\n"
                f"Subject: {email['subject'] or '[Specify]'}\n"
                f"Tone: {email['tone']}\n\n"
                f"{'='*50}\n{email['body']}\n{'='*50}\n"
                "Notes: Review and customize before sending."
            )
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
            result = self.email_processor.analyze_email(email_content, sender, received_date)
            if not result["success"]:
                return f"❌ Error analyzing email: {result.get('error', 'unknown')}"
            a = result["analysis"]
            icon = {
                "LOW": "🟢",
                "MEDIUM": "🟡",
                "HIGH": "🟠",
                "URGENT": "🔴",
            }.get(a["urgency_level"], "🟡")
            return (
                "📧 Email Analysis\n\n"
                f"{icon} Urgency: {a['urgency_level']} (Priority {a['priority_score']}/10)\n"
                f"Category: {a['category'].title()}\n"
                f"From: {a['sender'] or 'Unknown'}\n"
                f"Received: {a['received_date'] or 'Not specified'}\n"
                f"Length: {a['word_count']} words\n\n"
                f"{'='*60}\n{a['full_analysis']}\n{'='*60}\n"
            )
        except Exception as e:
            logger.error(f"Error analyzing email: {str(e)}")
            return f"❌ Error analyzing email: {str(e)}"

    def handle_general_chat(self, query: str) -> str:
        try:
            sources = self.retriever.get_available_sources()
            context_info = (
                "\nCapabilities:\n"
                f"- {sources.get('sources_by_type', {}).get('vector', {}).get('count', 0)} document sources\n"
                f"- {sources.get('sources_by_type', {}).get('sql', {}).get('count', 0)} data tables\n"
                "- Email drafting/analysis\n"
            )
            system_prompt = (
                "You are a helpful assistant for a CEO's RAG system. "
                "Be concise and helpful."
                f"\n{context_info}\n"
                f"Conversation context:\n{self.context.get_context_string(3)}"
            )
            resp = self.client.chat.completions.create(
                model=self.query_classifier.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                max_tokens=500,
                temperature=0.7,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error in general chat: {str(e)}")
            return (
                "I can search documents, analyze data, draft emails, or chat. "
                "What would you like to do?"
            )

    def _should_attempt_hybrid_search(self, query: str, initial_response: str) -> bool:
        indicators = ["compare", "analysis", "comprehensive", "relationship", "trend", "pattern"]
        ql = query.lower()
        has_indicators = any(x in ql for x in indicators)
        limited = any(
            x in initial_response.lower()
            for x in ["no data", "no results", "didn't find"]
        )
        biz_terms = [
            "performance",
            "revenue",
            "profit",
            "financial",
            "quarter",
            "strategy",
            "budget",
        ]
        has_biz = any(x in ql for x in biz_terms)
        return has_indicators or limited or has_biz

    def process_query(self, query: str) -> str:
        start = time.time()
        try:
            qtype, extracted = self.query_classifier.classify_query(
                query,
                self.context.get_context_string(3),
            )
            logger.info(f"Query classified as: {qtype.value}")

            if qtype == QueryType.DOCUMENT_SEARCH:
                resp = self.handle_document_search(query, extracted)
            elif qtype == QueryType.DATA_QUERY:
                resp = self.handle_data_query(query, extracted)
            elif qtype == QueryType.EMAIL_DRAFT:
                resp = self.handle_email_draft(query, extracted)
            elif qtype == QueryType.EMAIL_ANALYZE:
                resp = self.handle_email_analysis(query, extracted)
            elif qtype == QueryType.HELP:
                self.display_help()
                return ""
            elif qtype == QueryType.SYSTEM_STATUS:
                self.display_system_status()
                return ""
            else:
                resp = self.handle_general_chat(query)

            if qtype in [QueryType.DOCUMENT_SEARCH, QueryType.DATA_QUERY]:
                if self._should_attempt_hybrid_search(query, resp):
                    self.print_colored(
                        "🔄 Trying comprehensive (hybrid) search...",
                        "system",
                    )
                    hybrid = self.handle_hybrid_search(query, extracted)
                    if len(hybrid) > len(resp):
                        resp = hybrid

            elapsed = time.time() - start
            self.context.add_exchange(query, resp, qtype, {"processing_time": elapsed})
            return resp
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            err = f"❌ Error: {str(e)}"
            self.context.add_exchange(
                query,
                err,
                QueryType.GENERAL_CHAT,
                {"error": str(e)},
            )
            return err

    def run_console_loop(self) -> None:
        print("\n")
        self.display_welcome()
        while self.is_running:
            try:
                print(f"\n{self.colors['user']}You: {self.colors['reset']}", end="")
                user_input = input().strip()
                if not user_input:
                    continue
                low = user_input.lower()
                if low in ["quit", "exit", "q"]:
                    self.print_colored("👋 Goodbye!", "system")
                    self.is_running = False
                    break
                elif low == "help":
                    self.display_help()
                    continue
                elif low == "status":
                    self.display_system_status()
                    continue
                elif low == "history":
                    self.display_conversation_history()
                    continue
                elif low == "clear":
                    self.context.history.clear()
                    self.print_colored("🗑️ History cleared.", "system")
                    continue

                print(f"\n{self.colors['assistant']}Assistant:{self.colors['reset']}")
                response = self.process_query(user_input)
                if response:
                    print(response)
            except KeyboardInterrupt:
                self.print_colored("\n\n👋 Goodbye! (Interrupted)", "system")
                self.is_running = False
                break
            except EOFError:
                self.print_colored("\n\n👋 Goodbye! (End of input)", "system")
                self.is_running = False
                break
            except Exception as e:
                self.print_colored(f"❌ Unexpected error: {str(e)}", "error")
                logger.error(f"Unexpected error: {str(e)}")

    def close(self) -> None:
        try:
            self.retriever.close()
        except Exception:
            pass
        logger.info("ConversationalRAGSystem closed")


def create_console_rag_system(db_config: Dict[str, str] = None) -> ConversationalRAGSystem:
    return ConversationalRAGSystem(lm_client=client, db_config=db_config)


def main() -> int:
    print("🚀 Starting CEO RAG Chatbot (LM Studio)...")
    # Directory ensured at top, but keeping this is harmless
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
    sys.exit(main())

