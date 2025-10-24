"""
CEO RAG Chatbot Integration Test Script

This script tests the retriever.py and conversation.py components with various query types
to verify correct routing and response generation.

Requirements:
- PostgreSQL database running with processed documents/data
- OpenAI API key configured
- All dependencies installed

Usage:
    python test_rag_system.py
"""

import os
import sys
import time
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

# Add the src directory to Python path for imports
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

# Add specific module paths
rag_path = os.path.join(src_path, 'rag')
llm_path = os.path.join(src_path, 'llm')
sys.path.insert(0, rag_path)
sys.path.insert(0, llm_path)

try:
    # Try different import strategies
    try:
        # Strategy 1: Import from src/rag/ and src/llm/ (expected structure)
        from rag.retriever import HybridRetriever, create_retriever
        from llm.conversation import (
            ConversationalAgent, 
            create_conversational_agent, 
            QueryType,
            format_email_for_analysis,
            create_query_examples
        )
        print("✅ Successfully imported modules from src/rag/ and src/llm/")
    except ImportError:
        # Strategy 2: Try direct imports if files are in current directory
        try:
            from retriever import HybridRetriever, create_retriever
            from conversation import (
                ConversationalAgent, 
                create_conversational_agent, 
                QueryType,
                format_email_for_analysis,
                create_query_examples
            )
            print("✅ Successfully imported modules from current directory")
        except ImportError:
            # Strategy 3: Try importing from src/ directly
            try:
                sys.path.insert(0, os.path.join(project_root, 'src', 'rag'))
                sys.path.insert(0, os.path.join(project_root, 'src', 'llm'))
                
                import retriever
                import conversation
                
                HybridRetriever = retriever.HybridRetriever
                create_retriever = retriever.create_retriever
                ConversationalAgent = conversation.ConversationalAgent
                create_conversational_agent = conversation.create_conversational_agent
                QueryType = conversation.QueryType
                format_email_for_analysis = conversation.format_email_for_analysis
                create_query_examples = conversation.create_query_examples
                
                print("✅ Successfully imported modules using direct import strategy")
            except ImportError as final_error:
                raise final_error

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\n🔍 Checking file structure...")
    
    # Check if files exist
    possible_locations = [
        'src/rag/retriever.py',
        'src/llm/conversation.py', 
        'retriever.py',
        'conversation.py',
        'src/retriever.py',
        'src/conversation.py'
    ]
    
    found_files = []
    for location in possible_locations:
        full_path = os.path.join(project_root, location)
        if os.path.exists(full_path):
            found_files.append(location)
            print(f"✅ Found: {location}")
        else:
            print(f"❌ Not found: {location}")
    
    if found_files:
        print(f"\n💡 Found files at: {found_files}")
        print("The import paths in the test script may need adjustment.")
    
    print(f"\n📁 Current working directory: {os.getcwd()}")
    print(f"📁 Project root: {project_root}")
    print(f"📁 Src path: {src_path}")
    
    # List directories to help debug
    if os.path.exists(src_path):
        print(f"\n📂 Contents of src/:")
        for item in os.listdir(src_path):
            item_path = os.path.join(src_path, item)
            if os.path.isdir(item_path):
                print(f"   📁 {item}/")
                try:
                    for subitem in os.listdir(item_path):
                        if subitem.endswith('.py'):
                            print(f"      📄 {subitem}")
                except PermissionError:
                    print(f"      ❌ Permission denied")
            else:
                print(f"   📄 {item}")
    else:
        print(f"❌ src/ directory not found at {src_path}")
    
    print("\n💡 Possible solutions:")
    print("1. Make sure you're running from the project root directory")
    print("2. Check that retriever.py exists in src/rag/")
    print("3. Check that conversation.py exists in src/llm/")
    print("4. Ensure all __init__.py files exist (run with --validate to auto-create)")
    print("5. Try running: python test_rag_system.py --validate")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_results.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class RAGSystemTester:
    """Comprehensive tester for the RAG system components."""
    
    def __init__(self, openai_api_key: Optional[str] = None, db_config: Optional[Dict] = None):
        """
        Initialize the tester with retriever and conversational agent.
        
        Args:
            openai_api_key: OpenAI API key (gets from env if not provided)
            db_config: Database configuration (uses defaults if not provided)
        """
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.db_config = db_config
        
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        
        print("🚀 Initializing RAG System Components...")
        
        # Initialize retriever
        try:
            self.retriever = create_retriever(
                db_config=self.db_config,
                openai_api_key=self.openai_api_key
            )
            print("✅ Retriever initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize retriever: {e}")
            raise
        
        # Initialize conversational agent
        try:
            self.agent = create_conversational_agent(
                openai_api_key=self.openai_api_key,
                db_config=self.db_config
            )
            print("✅ Conversational agent initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize conversational agent: {e}")
            raise
        
        # Test results storage
        self.test_results = []
        self.start_time = time.time()
        
    def print_section_header(self, title: str):
        """Print formatted section header."""
        print(f"\n{'='*80}")
        print(f"🔍 {title}")
        print(f"{'='*80}")
    
    def print_test_header(self, test_name: str, query: str):
        """Print formatted test header."""
        print(f"\n📋 TEST: {test_name}")
        print(f"Query: '{query}'")
        print("-" * 60)
    
    def record_test_result(self, test_name: str, query: str, success: bool, 
                          response_time: float, details: Dict[str, Any]):
        """Record test result for later analysis."""
        result = {
            'test_name': test_name,
            'query': query,
            'success': success,
            'response_time': response_time,
            'timestamp': datetime.now().isoformat(),
            'details': details
        }
        self.test_results.append(result)
    
    def test_system_status(self):
        """Test system initialization and status."""
        self.print_section_header("SYSTEM STATUS VERIFICATION")
        
        try:
            # Test retriever status
            sources = self.retriever.get_available_sources()
            print(f"📊 Available Data Sources:")
            print(f"   Vector Tables: {sources.get('vector_sources', {}).get('count', 0)}")
            print(f"   Data Tables: {sources.get('data_sources', {}).get('count', 0)}")
            print(f"   Total Documents: {len(sources.get('vector_sources', {}).get('tables', []))}")
            print(f"   Total Chunks: {sources.get('vector_sources', {}).get('total_chunks', 0)}")
            print(f"   Total Data Rows: {sources.get('data_sources', {}).get('total_rows', 0)}")
            
            # Test agent status
            agent_status = self.agent.get_system_status()
            print(f"\n🤖 Agent Status: {agent_status['status']}")
            print(f"   Capabilities: {list(agent_status.get('capabilities', {}).keys())}")
            print(f"   Conversation History: {agent_status.get('conversation_history', 0)}")
            
            # Check if we have data to work with
            has_vector_data = sources.get('vector_sources', {}).get('count', 0) > 0
            has_sql_data = sources.get('data_sources', {}).get('count', 0) > 0
            
            if not has_vector_data and not has_sql_data:
                print("⚠️  WARNING: No data sources found. Upload documents/data for meaningful tests.")
            
            return True, {
                'vector_sources': sources.get('vector_sources', {}),
                'data_sources': sources.get('data_sources', {}),
                'agent_status': agent_status
            }
            
        except Exception as e:
            print(f"❌ System status check failed: {e}")
            return False, {'error': str(e)}
    
    def test_retriever_vector_search(self):
        """Test retriever vector search functionality."""
        self.print_section_header("RETRIEVER - VECTOR SEARCH TESTS")
        
        vector_test_queries = [
            "financial performance and revenue growth",
            "market analysis and competitive positioning", 
            "operational efficiency and cost reduction",
            "strategic planning and business objectives",
            "risk management and compliance"
        ]
        
        for query in vector_test_queries:
            self.print_test_header("Vector Search", query)
            
            start_time = time.time()
            try:
                results = self.retriever.vector_search(query)
                response_time = time.time() - start_time
                
                print(f"✅ Found {len(results)} vector chunks")
                print(f"⏱️  Response time: {response_time:.2f}s")
                
                # Show top results
                for i, chunk in enumerate(results[:3]):
                    print(f"   {i+1}. File: {chunk['file_name']}, Page: {chunk['page_number']}")
                    print(f"      Similarity: {chunk['similarity_score']:.3f}")
                    print(f"      Preview: {chunk['text'][:100]}...")
                
                success = len(results) > 0
                if not success:
                    print("⚠️  No vector results found - check if documents are processed")
                
                self.record_test_result(
                    "Vector Search", query, success, response_time,
                    {'result_count': len(results), 'has_results': success}
                )
                
            except Exception as e:
                response_time = time.time() - start_time
                print(f"❌ Vector search failed: {e}")
                self.record_test_result(
                    "Vector Search", query, False, response_time,
                    {'error': str(e)}
                )
    
    def test_retriever_sql_search(self):
        """Test retriever SQL search functionality."""
        self.print_section_header("RETRIEVER - SQL SEARCH TESTS")
        
        sql_test_queries = [
            "Show me the total sales revenue",
            "What are the top performing products?",
            "Calculate average profit margins by category",
            "Find records with highest values",
            "Show recent financial data"
        ]
        
        for query in sql_test_queries:
            self.print_test_header("SQL Search", query)
            
            start_time = time.time()
            try:
                results = self.retriever.sql_search(query)
                response_time = time.time() - start_time
                
                print(f"✅ Found {len(results)} SQL results")
                print(f"⏱️  Response time: {response_time:.2f}s")
                
                if results:
                    # Show generated SQL
                    if 'generated_sql' in results[0]:
                        print(f"📝 Generated SQL: {results[0]['generated_sql'][:100]}...")
                    
                    # Show sample data
                    print(f"📊 Sample results:")
                    for i, result in enumerate(results[:2]):
                        clean_result = {k: v for k, v in result.items() 
                                     if k not in ['source_type', 'generated_sql']}
                        print(f"   Row {i+1}: {str(clean_result)[:150]}...")
                
                success = len(results) > 0
                if not success:
                    print("⚠️  No SQL results found - check if data tables exist")
                
                self.record_test_result(
                    "SQL Search", query, success, response_time,
                    {'result_count': len(results), 'has_results': success}
                )
                
            except Exception as e:
                response_time = time.time() - start_time
                print(f"❌ SQL search failed: {e}")
                self.record_test_result(
                    "SQL Search", query, False, response_time,
                    {'error': str(e)}
                )
    
    def test_retriever_hybrid_search(self):
        """Test retriever hybrid search (both vector and SQL)."""
        self.print_section_header("RETRIEVER - HYBRID SEARCH TESTS")
        
        hybrid_test_queries = [
            "financial performance metrics and strategic analysis",
            "revenue growth trends and market positioning",
            "operational costs and efficiency improvements"
        ]
        
        for query in hybrid_test_queries:
            self.print_test_header("Hybrid Search", query)
            
            start_time = time.time()
            try:
                result = self.retriever.hybrid_retrieve(query)
                response_time = time.time() - start_time
                
                vector_count = len(result.vector_chunks)
                sql_count = len(result.sql_results)
                total_results = vector_count + sql_count
                
                print(f"✅ Hybrid search completed")
                print(f"📄 Vector chunks: {vector_count}")
                print(f"🗄️  SQL results: {sql_count}")
                print(f"📊 Total results: {total_results}")
                print(f"⏱️  Response time: {response_time:.2f}s")
                
                # Show metadata
                print(f"🔧 Metadata:")
                for key, value in result.metadata.items():
                    if key != 'total_retrieval_time':
                        print(f"   {key}: {value}")
                
                success = total_results > 0
                if not success:
                    print("⚠️  No hybrid results found")
                
                self.record_test_result(
                    "Hybrid Search", query, success, response_time,
                    {
                        'vector_count': vector_count,
                        'sql_count': sql_count,
                        'total_results': total_results,
                        'metadata': result.metadata
                    }
                )
                
            except Exception as e:
                response_time = time.time() - start_time
                print(f"❌ Hybrid search failed: {e}")
                self.record_test_result(
                    "Hybrid Search", query, False, response_time,
                    {'error': str(e)}
                )
    
    def test_conversation_email_drafting(self):
        """Test conversation agent email drafting functionality."""
        self.print_section_header("CONVERSATION - EMAIL DRAFTING TESTS")
        
        email_draft_queries = [
            "Draft an email to John Smith about the quarterly financial review meeting",
            "Write a professional email to the board about budget concerns and cost optimization",
            "Compose an email to Sarah Johnson requesting the Q3 sales performance report",
            "Create a friendly email to the team announcing the successful product launch",
            "Draft a formal email to investors about our expansion plans"
        ]
        
        for query in email_draft_queries:
            self.print_test_header("Email Drafting", query)
            
            start_time = time.time()
            try:
                response = self.agent.process_query(query)
                response_time = time.time() - start_time
                
                print(f"✅ Intent detected: {response.action_taken.value}")
                print(f"✅ Success: {response.success}")
                print(f"⏱️  Response time: {response_time:.2f}s")
                
                if response.action_taken == QueryType.EMAIL_DRAFT:
                    print(f"📧 Email drafted successfully")
                    if response.data and 'subject' in response.data:
                        print(f"   Subject: {response.data['subject']}")
                        print(f"   Recipient: {response.data.get('recipient', 'Not specified')}")
                else:
                    print(f"⚠️  Unexpected intent: {response.action_taken.value}")
                
                print(f"📝 Response preview: {response.response_text[:200]}...")
                
                success = (response.success and 
                          response.action_taken == QueryType.EMAIL_DRAFT)
                
                self.record_test_result(
                    "Email Drafting", query, success, response_time,
                    {
                        'intent': response.action_taken.value,
                        'success': response.success,
                        'has_email_data': bool(response.data)
                    }
                )
                
            except Exception as e:
                response_time = time.time() - start_time
                print(f"❌ Email drafting failed: {e}")
                self.record_test_result(
                    "Email Drafting", query, False, response_time,
                    {'error': str(e)}
                )
    
    def test_conversation_email_analysis(self):
        """Test conversation agent email analysis functionality."""
        self.print_section_header("CONVERSATION - EMAIL ANALYSIS TESTS")
        
        # Sample emails for analysis
        test_emails = [
            {
                "query": "Analyze this email: Subject: URGENT - Server Outage. From: IT Team. The main server is down and requires immediate attention. Expected downtime 2-4 hours.",
                "expected_importance": "HIGH"
            },
            {
                "query": "Read this email: Subject: Team Meeting Tomorrow. From: Manager. Hi everyone, just a reminder about our weekly team meeting tomorrow at 2 PM in conference room A.",
                "expected_importance": "MEDIUM"
            },
            {
                "query": "Summarize this email: Subject: Invoice #12345 Due. From: Accounting. This is a friendly reminder that invoice #12345 for $50,000 is due next week.",
                "expected_importance": "MEDIUM"
            }
        ]
        
        for email_test in test_emails:
            query = email_test["query"]
            self.print_test_header("Email Analysis", query)
            
            start_time = time.time()
            try:
                response = self.agent.process_query(query)
                response_time = time.time() - start_time
                
                print(f"✅ Intent detected: {response.action_taken.value}")
                print(f"✅ Success: {response.success}")
                print(f"⏱️  Response time: {response_time:.2f}s")
                
                if response.action_taken == QueryType.EMAIL_READ:
                    print(f"📧 Email analyzed successfully")
                    if response.data:
                        importance = response.data.get('importance_level', 'Unknown')
                        priority = response.data.get('priority_score', 'Unknown')
                        print(f"   Importance: {importance}")
                        print(f"   Priority Score: {priority}/10")
                        print(f"   Summary: {response.data.get('summary', 'No summary')[:100]}...")
                else:
                    print(f"⚠️  Unexpected intent: {response.action_taken.value}")
                
                success = (response.success and 
                          response.action_taken == QueryType.EMAIL_READ)
                
                self.record_test_result(
                    "Email Analysis", query, success, response_time,
                    {
                        'intent': response.action_taken.value,
                        'success': response.success,
                        'importance_detected': response.data.get('importance_level') if response.data else None
                    }
                )
                
            except Exception as e:
                response_time = time.time() - start_time
                print(f"❌ Email analysis failed: {e}")
                self.record_test_result(
                    "Email Analysis", query, False, response_time,
                    {'error': str(e)}
                )
    
    def test_conversation_sql_queries(self):
        """Test conversation agent SQL query routing."""
        self.print_section_header("CONVERSATION - SQL QUERY TESTS")
        
        sql_queries = [
            "What were our total sales last quarter?",
            "Show me the revenue breakdown by product category",
            "Which departments have the highest costs?",
            "Calculate our profit margins for this year",
            "Find the top 10 customers by revenue"
        ]
        
        for query in sql_queries:
            self.print_test_header("SQL Query Routing", query)
            
            start_time = time.time()
            try:
                response = self.agent.process_query(query)
                response_time = time.time() - start_time
                
                print(f"✅ Intent detected: {response.action_taken.value}")
                print(f"✅ Success: {response.success}")
                print(f"⏱️  Response time: {response_time:.2f}s")
                
                if response.action_taken == QueryType.SQL_QUERY:
                    print(f"🗄️  SQL query executed successfully")
                    if response.data:
                        result_count = response.data.get('result_count', 0)
                        print(f"   Results found: {result_count}")
                        if 'generated_sql' in response.data:
                            print(f"   SQL: {response.data['generated_sql'][:100]}...")
                else:
                    print(f"⚠️  Routed to: {response.action_taken.value}")
                
                print(f"📝 Response preview: {response.response_text[:200]}...")
                
                # Consider success if intent is correct (even if no data found)
                success = response.action_taken == QueryType.SQL_QUERY
                
                self.record_test_result(
                    "SQL Query Routing", query, success, response_time,
                    {
                        'intent': response.action_taken.value,
                        'success': response.success,
                        'result_count': response.data.get('result_count', 0) if response.data else 0
                    }
                )
                
            except Exception as e:
                response_time = time.time() - start_time
                print(f"❌ SQL query routing failed: {e}")
                self.record_test_result(
                    "SQL Query Routing", query, False, response_time,
                    {'error': str(e)}
                )
    
    def test_conversation_document_search(self):
        """Test conversation agent document search routing."""
        self.print_section_header("CONVERSATION - DOCUMENT SEARCH TESTS")
        
        document_queries = [
            "Find information about our strategic business plan",
            "Search for details about market analysis and competitor research",
            "Look for documents mentioning financial projections",
            "Find references to operational efficiency improvements",
            "Search for information about our product development roadmap"
        ]
        
        for query in document_queries:
            self.print_test_header("Document Search Routing", query)
            
            start_time = time.time()
            try:
                response = self.agent.process_query(query)
                response_time = time.time() - start_time
                
                print(f"✅ Intent detected: {response.action_taken.value}")
                print(f"✅ Success: {response.success}")
                print(f"⏱️  Response time: {response_time:.2f}s")
                
                if response.action_taken == QueryType.DOCUMENT_SEARCH:
                    print(f"📄 Document search executed successfully")
                    if response.data:
                        total_chunks = response.data.get('total_chunks', 0)
                        unique_docs = len(response.data.get('unique_documents', []))
                        print(f"   Chunks found: {total_chunks}")
                        print(f"   Documents: {unique_docs}")
                else:
                    print(f"⚠️  Routed to: {response.action_taken.value}")
                
                print(f"📝 Response preview: {response.response_text[:200]}...")
                
                # Consider success if intent is correct
                success = response.action_taken == QueryType.DOCUMENT_SEARCH
                
                self.record_test_result(
                    "Document Search Routing", query, success, response_time,
                    {
                        'intent': response.action_taken.value,
                        'success': response.success,
                        'chunks_found': response.data.get('total_chunks', 0) if response.data else 0
                    }
                )
                
            except Exception as e:
                response_time = time.time() - start_time
                print(f"❌ Document search routing failed: {e}")
                self.record_test_result(
                    "Document Search Routing", query, False, response_time,
                    {'error': str(e)}
                )
    
    def test_conversation_general_chat(self):
        """Test conversation agent general chat functionality."""
        self.print_section_header("CONVERSATION - GENERAL CHAT TESTS")
        
        general_queries = [
            "Hello, how are you today?",
            "What can you help me with?",
            "Show me the system status",
            "What are your capabilities?",
            "Thanks for your help!"
        ]
        
        for query in general_queries:
            self.print_test_header("General Chat", query)
            
            start_time = time.time()
            try:
                response = self.agent.process_query(query)
                response_time = time.time() - start_time
                
                print(f"✅ Intent detected: {response.action_taken.value}")
                print(f"✅ Success: {response.success}")
                print(f"⏱️  Response time: {response_time:.2f}s")
                
                print(f"💬 Response: {response.response_text[:300]}...")
                
                # General chat should be successful
                success = response.success
                
                self.record_test_result(
                    "General Chat", query, success, response_time,
                    {
                        'intent': response.action_taken.value,
                        'success': response.success,
                        'response_length': len(response.response_text)
                    }
                )
                
            except Exception as e:
                response_time = time.time() - start_time
                print(f"❌ General chat failed: {e}")
                self.record_test_result(
                    "General Chat", query, False, response_time,
                    {'error': str(e)}
                )
    
    def test_conversation_context_awareness(self):
        """Test conversation agent context and follow-up handling."""
        self.print_section_header("CONVERSATION - CONTEXT AWARENESS TESTS")
        
        # Test conversation flow
        conversation_flow = [
            "What were our sales figures last quarter?",
            "Can you break that down by region?",
            "Which region performed the best?",
            "Draft an email to the sales team about these results"
        ]
        
        print("🔄 Testing conversational flow with context...")
        
        for i, query in enumerate(conversation_flow, 1):
            self.print_test_header(f"Context Test Step {i}", query)
            
            # Get current context
            context = self.agent.get_conversation_context(3)
            print(f"📚 Context length: {len(context.split()) if context else 0} words")
            
            start_time = time.time()
            try:
                response = self.agent.process_query(query, context)
                response_time = time.time() - start_time
                
                print(f"✅ Intent detected: {response.action_taken.value}")
                print(f"✅ Success: {response.success}")
                print(f"⏱️  Response time: {response_time:.2f}s")
                print(f"📝 Response preview: {response.response_text[:150]}...")
                
                self.record_test_result(
                    f"Context Step {i}", query, response.success, response_time,
                    {
                        'intent': response.action_taken.value,
                        'context_length': len(context.split()) if context else 0,
                        'success': response.success
                    }
                )
                
                # Brief pause between queries
                time.sleep(1)
                
            except Exception as e:
                response_time = time.time() - start_time
                print(f"❌ Context test step {i} failed: {e}")
                self.record_test_result(
                    f"Context Step {i}", query, False, response_time,
                    {'error': str(e)}
                )
    
    def generate_test_report(self):
        """Generate comprehensive test report."""
        self.print_section_header("TEST RESULTS SUMMARY")
        
        total_time = time.time() - self.start_time
        total_tests = len(self.test_results)
        successful_tests = sum(1 for r in self.test_results if r['success'])
        failed_tests = total_tests - successful_tests
        
        print(f"📊 OVERALL RESULTS:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Successful: {successful_tests} ({successful_tests/total_tests*100:.1f}%)")
        print(f"   Failed: {failed_tests} ({failed_tests/total_tests*100:.1f}%)")
        print(f"   Total Time: {total_time:.2f}s")
        print(f"   Average Time: {total_time/total_tests:.2f}s per test")
        
        # Group results by test type
        test_types = {}
        for result in self.test_results:
            test_type = result['test_name']
            if test_type not in test_types:
                test_types[test_type] = {'total': 0, 'successful': 0, 'times': []}
            
            test_types[test_type]['total'] += 1
            if result['success']:
                test_types[test_type]['successful'] += 1
            test_types[test_type]['times'].append(result['response_time'])
        
        print(f"\n📈 RESULTS BY TEST TYPE:")
        for test_type, stats in test_types.items():
            success_rate = stats['successful'] / stats['total'] * 100
            avg_time = sum(stats['times']) / len(stats['times'])
            print(f"   {test_type}:")
            print(f"      Success Rate: {success_rate:.1f}% ({stats['successful']}/{stats['total']})")
            print(f"      Avg Time: {avg_time:.2f}s")
        
        # Show failed tests
        failed_results = [r for r in self.test_results if not r['success']]
        if failed_results:
            print(f"\n❌ FAILED TESTS:")
            for result in failed_results:
                print(f"   {result['test_name']}: {result['query'][:50]}...")
                if 'error' in result['details']:
                    print(f"      Error: {result['details']['error']}")
        
        # Save detailed results
        try:
            with open('test_results_detailed.json', 'w') as f:
                json.dump(self.test_results, f, indent=2)
            print(f"\n💾 Detailed results saved to: test_results_detailed.json")
        except Exception as e:
            print(f"⚠️  Could not save detailed results: {e}")
        
        # Performance insights
        response_times = [r['response_time'] for r in self.test_results if r['success']]
        if response_times:
            print(f"\n⚡ PERFORMANCE INSIGHTS:")
            print(f"   Fastest Response: {min(response_times):.2f}s")
            print(f"   Slowest Response: {max(response_times):.2f}s")
            print(f"   Median Response: {sorted(response_times)[len(response_times)//2]:.2f}s")
        
        return {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'failed_tests': failed_tests,
            'success_rate': successful_tests / total_tests * 100,
            'total_time': total_time,
            'average_time': total_time / total_tests,
            'test_types': test_types,
            'failed_results': failed_results
        }
    
    def run_all_tests(self):
        """Run all test suites."""
        print("🎯 Starting Comprehensive RAG System Testing")
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # System status check
            status_success, status_details = self.test_system_status()
            if not status_success:
                print("❌ System status check failed - aborting tests")
                return False
            
            # Test retriever components
            self.test_retriever_vector_search()
            self.test_retriever_sql_search()
            self.test_retriever_hybrid_search()
            
            # Test conversation components
            self.test_conversation_email_drafting()
            self.test_conversation_email_analysis()
            self.test_conversation_sql_queries()
            self.test_conversation_document_search()
            self.test_conversation_general_chat()
            
            # Test context awareness
            self.test_conversation_context_awareness()
            
            # Generate final report
            report = self.generate_test_report()
            
            print(f"\n🎉 Testing completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Testing failed with error: {e}")
            logger.error(f"Test suite failed: {e}")
            return False
        
        finally:
            # Cleanup
            try:
                self.cleanup()
            except:
                pass
    
    def cleanup(self):
        """Clean up resources."""
        try:
            if hasattr(self, 'retriever') and self.retriever:
                self.retriever.close()
            if hasattr(self, 'agent') and self.agent:
                self.agent.close()
            print("🧹 Cleanup completed")
        except Exception as e:
            print(f"⚠️  Cleanup warning: {e}")


def run_quick_test():
    """Run a quick subset of tests for rapid validation."""
    print("🚀 Running Quick Validation Tests...")
    
    try:
        tester = RAGSystemTester()
        
        # Quick system check
        status_success, _ = tester.test_system_status()
        if not status_success:
            print("❌ Quick test failed - system not ready")
            return False
        
        # Test one query of each type
        quick_tests = [
            ("Vector Search", lambda: tester.retriever.vector_search("financial performance")),
            ("SQL Search", lambda: tester.retriever.sql_search("total revenue")),
            ("Email Draft", lambda: tester.agent.process_query("Draft email to John about meeting")),
            ("Document Search", lambda: tester.agent.process_query("Find strategic planning documents")),
            ("General Chat", lambda: tester.agent.process_query("Hello, what can you do?"))
        ]
        
        results = []
        for test_name, test_func in quick_tests:
            print(f"\n🔍 Quick Test: {test_name}")
            try:
                start_time = time.time()
                result = test_func()
                response_time = time.time() - start_time
                
                success = bool(result)
                if hasattr(result, 'success'):
                    success = result.success
                elif isinstance(result, list):
                    success = len(result) > 0
                
                print(f"   ✅ {test_name}: {'SUCCESS' if success else 'NO RESULTS'} ({response_time:.2f}s)")
                results.append(success)
                
            except Exception as e:
                print(f"   ❌ {test_name}: FAILED - {e}")
                results.append(False)
        
        success_rate = sum(results) / len(results) * 100
        print(f"\n📊 Quick Test Results: {sum(results)}/{len(results)} passed ({success_rate:.1f}%)")
        
        tester.cleanup()
        return success_rate > 60  # Consider success if >60% pass
        
    except Exception as e:
        print(f"❌ Quick test setup failed: {e}")
        return False


def run_performance_test():
    """Run performance-focused tests."""
    print("⚡ Running Performance Tests...")
    
    try:
        tester = RAGSystemTester()
        
        # Performance test queries
        perf_queries = [
            "financial performance metrics",
            "sales revenue analysis", 
            "operational efficiency data"
        ]
        
        print(f"🎯 Testing response times for {len(perf_queries)} queries...")
        
        times = []
        for i, query in enumerate(perf_queries, 1):
            print(f"   Test {i}: '{query}'")
            
            # Vector search timing
            start = time.time()
            vector_results = tester.retriever.vector_search(query)
            vector_time = time.time() - start
            
            # Hybrid search timing
            start = time.time()
            hybrid_result = tester.retriever.hybrid_retrieve(query)
            hybrid_time = time.time() - start
            
            # Conversation timing
            start = time.time()
            conv_response = tester.agent.process_query(f"Find information about {query}")
            conv_time = time.time() - start
            
            times.append({
                'query': query,
                'vector_time': vector_time,
                'hybrid_time': hybrid_time,
                'conversation_time': conv_time
            })
            
            print(f"      Vector: {vector_time:.2f}s, Hybrid: {hybrid_time:.2f}s, Conversation: {conv_time:.2f}s")
        
        # Performance summary
        avg_vector = sum(t['vector_time'] for t in times) / len(times)
        avg_hybrid = sum(t['hybrid_time'] for t in times) / len(times)
        avg_conv = sum(t['conversation_time'] for t in times) / len(times)
        
        print(f"\n📈 Performance Summary:")
        print(f"   Average Vector Search: {avg_vector:.2f}s")
        print(f"   Average Hybrid Search: {avg_hybrid:.2f}s")
        print(f"   Average Conversation: {avg_conv:.2f}s")
        
        # Performance evaluation
        if avg_conv < 5.0:
            print("   🟢 Performance: EXCELLENT")
        elif avg_conv < 10.0:
            print("   🟡 Performance: GOOD")
        else:
            print("   🔴 Performance: NEEDS IMPROVEMENT")
        
        tester.cleanup()
        return times
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return []


def create_missing_init_files():
    """Create missing __init__.py files for proper Python package structure."""
    init_files = [
        'src/__init__.py',
        'src/rag/__init__.py', 
        'src/llm/__init__.py',
        'src/document_processing/__init__.py',
        'src/vector_store/__init__.py',
        'src/utils/__init__.py'
    ]
    
    created_files = []
    for init_file in init_files:
        if not os.path.exists(init_file):
            try:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(init_file), exist_ok=True)
                
                # Create empty __init__.py file
                with open(init_file, 'w') as f:
                    f.write('# Auto-generated __init__.py file\n')
                
                created_files.append(init_file)
                print(f"📝 Created missing {init_file}")
                
            except Exception as e:
                print(f"⚠️  Could not create {init_file}: {e}")
    
    if created_files:
        print(f"✅ Created {len(created_files)} missing __init__.py files")
        return True
    else:
        print("ℹ️  All __init__.py files already exist")
        return False


def validate_environment():
    """Validate that the environment is set up correctly."""
    print("🔍 Validating Environment Setup...")
    
    issues = []
    
    # Create missing __init__.py files first
    print("\n📝 Checking Python package structure...")
    create_missing_init_files()
    
    # Check OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        issues.append("❌ OPENAI_API_KEY environment variable not set")
    else:
        print("✅ OpenAI API key found")
    
    # Check database environment variables
    db_vars = ['POSTGRES_HOST', 'POSTGRES_PORT', 'POSTGRES_DB', 'POSTGRES_USER']
    for var in db_vars:
        if not os.getenv(var):
            issues.append(f"⚠️  {var} not set (using default)")
        else:
            print(f"✅ {var} configured")
    
    # Check if we can import required modules
    try:
        import openai
        print("✅ OpenAI library available")
    except ImportError:
        issues.append("❌ OpenAI library not installed")
    
    try:
        import psycopg2
        print("✅ PostgreSQL driver available")
    except ImportError:
        issues.append("❌ psycopg2 not installed")
    
    try:
        import sqlalchemy
        print("✅ SQLAlchemy available")
    except ImportError:
        issues.append("❌ SQLAlchemy not installed")
    
    # Try to connect to database
    try:
        from sqlalchemy import create_engine, text
        
        db_config = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': os.getenv('POSTGRES_PORT', '5432'),
            'database': os.getenv('POSTGRES_DB', 'ceo_rag_db'),
            'username': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', 'password')
        }
        
        password_part = f":{db_config['password']}" if db_config['password'] else ""
        connection_string = (
            f"postgresql://{db_config['username']}{password_part}@"
            f"{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )
        
        engine = create_engine(connection_string, echo=False)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        print("✅ Database connection successful")
        engine.dispose()
        
    except Exception as e:
        issues.append(f"❌ Database connection failed: {e}")
    
    # Check project structure
    expected_files = [
        ('src/rag/retriever.py', 'Retriever module'),
        ('src/llm/conversation.py', 'Conversation module'),
        ('src/rag/__init__.py', 'RAG package init'),
        ('src/llm/__init__.py', 'LLM package init'),
        ('src/__init__.py', 'Src package init')
    ]
    
    for file_path, description in expected_files:
        if os.path.exists(file_path):
            print(f"✅ {description} found at {file_path}")
        else:
            issues.append(f"❌ {description} not found at {file_path}")
            # Try to suggest where it might be
            filename = os.path.basename(file_path)
            for root, dirs, files in os.walk('.'):
                if filename in files:
                    actual_path = os.path.join(root, filename)
                    print(f"   💡 Found {filename} at {actual_path} instead")
    
    if issues:
        print(f"\n⚠️  Environment Issues Found:")
        for issue in issues:
            print(f"   {issue}")
        return False
    else:
        print(f"\n🎉 Environment validation passed!")
        return True


def main():
    """Main test runner with command line options."""
    import argparse
    
    parser = argparse.ArgumentParser(description='CEO RAG Chatbot System Tester')
    parser.add_argument('--quick', action='store_true', help='Run quick validation tests only')
    parser.add_argument('--performance', action='store_true', help='Run performance tests only')
    parser.add_argument('--validate', action='store_true', help='Validate environment setup only')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--output', type=str, help='Output file for results')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("🤖 CEO RAG Chatbot System Tester")
    print("=" * 50)
    
    # Environment validation
    if args.validate or not validate_environment():
        if args.validate:
            return
        else:
            print("❌ Environment validation failed. Use --validate flag for details.")
            return
    
    try:
        if args.quick:
            # Quick test
            success = run_quick_test()
            print(f"\n{'🎉 Quick tests PASSED' if success else '❌ Quick tests FAILED'}")
        
        elif args.performance:
            # Performance test
            results = run_performance_test()
            if results:
                print(f"\n🎉 Performance tests completed with {len(results)} queries")
            else:
                print(f"\n❌ Performance tests failed")
        
        else:
            # Full test suite
            print("🚀 Running Full Test Suite...")
            print("This may take several minutes depending on data size and API response times.")
            print()
            
            tester = RAGSystemTester()
            success = tester.run_all_tests()
            
            if success:
                print(f"\n🎉 All tests completed! Check test_results.log and test_results_detailed.json for details.")
            else:
                print(f"\n❌ Some tests failed. Check logs for details.")
    
    except KeyboardInterrupt:
        print(f"\n🛑 Testing interrupted by user")
    
    except Exception as e:
        print(f"\n❌ Testing failed with error: {e}")
        logger.error(f"Main test runner failed: {e}")
    
    print(f"\n✅ Testing session completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
