import logging
import json
import re
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import openai
from sqlalchemy import create_engine, text, MetaData
import pandas as pd
import os
import time
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """Container for retrieval results from both vector and SQL searches."""
    vector_chunks: List[Dict[str, Any]]
    sql_results: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    query_info: Dict[str, Any]

class TablePrefixConfig:
    """Configuration for table prefixes and their associated search types."""
    
    def __init__(self):
        self.prefixes = {
            'vectors_': {
                'search_type': 'vector',
                'description': 'Vector search tables containing document embeddings'
            },
            'iso_': {
                'search_type': 'sql',
                'description': 'ISO company data tables for SQL queries'
            }
            # Easy to extend with new prefixes:
            # 'analytics_': {
            #     'search_type': 'sql',
            #     'description': 'Analytics and metrics tables'
            # },
            # 'logs_': {
            #     'search_type': 'sql', 
            #     'description': 'Application and system logs'
            # }
        }
    
    def get_prefixes_by_type(self, search_type: str) -> List[str]:
        """Get all prefixes for a given search type."""
        return [prefix for prefix, config in self.prefixes.items() 
                if config['search_type'] == search_type]
    
    def add_prefix(self, prefix: str, search_type: str, description: str):
        """Add a new prefix configuration."""
        if not prefix.endswith('_'):
            prefix += '_'
        
        self.prefixes[prefix] = {
            'search_type': search_type,
            'description': description
        }

class TableSelector:
    """Handles LLM-based table selection from filtered lists."""
    
    def __init__(self, openai_api_key: str, engine):
        self.openai_api_key = openai_api_key
        self.engine = engine
        openai.api_key = openai_api_key
        
        # Cache for table lists to avoid repeated queries
        self._table_cache = {}
        self._cache_expiry = 300  # 5 minutes
    
    def _get_all_table_names(self) -> List[str]:
        """Get all table names from the database."""
        cache_key = "all_tables"
        now = time.time()
        
        # Check cache
        if (cache_key in self._table_cache and 
            now - self._table_cache[cache_key]['timestamp'] < self._cache_expiry):
            return self._table_cache[cache_key]['tables']
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_type = 'BASE TABLE'
                    ORDER BY table_name
                """))
                
                tables = [row[0] for row in result]
                
                # Update cache
                self._table_cache[cache_key] = {
                    'tables': tables,
                    'timestamp': now
                }
                
                return tables
                
        except Exception as e:
            logger.error(f"Error getting table names: {str(e)}")
            return []
    
    def _filter_tables_by_prefixes(self, tables: List[str], prefixes: List[str]) -> List[str]:
        """Filter table names by given prefixes."""
        filtered_tables = []
        for table in tables:
            for prefix in prefixes:
                if table.startswith(prefix):
                    filtered_tables.append(table)
                    break
        return sorted(filtered_tables)
    
    def _get_table_metadata(self, table_name: str) -> Dict[str, Any]:
        """Get basic metadata about a table for context."""
        try:
            with self.engine.connect() as conn:
                # Get column count and row count
                columns_result = conn.execute(text("""
                    SELECT COUNT(*) as column_count
                    FROM information_schema.columns
                    WHERE table_name = :table_name
                    AND table_schema = 'public'
                """), {"table_name": table_name})
                
                column_count = columns_result.scalar() or 0
                
                # Get row count (with timeout protection)
                try:
                    rows_result = conn.execute(text(f'SELECT COUNT(*) FROM "{table_name}"'))
                    row_count = rows_result.scalar() or 0
                except Exception:
                    row_count = "unknown"
                
                # For vector tables, extract document info from table name
                description = ""
                if table_name.startswith('vectors_'):
                    # Extract document name from table name pattern: vectors_doc_YYYY_document_name
                    try:
                        if 'vectors_doc_' in table_name:
                            # Remove prefix and convert underscores to readable format
                            doc_part = table_name.replace('vectors_doc_', '')
                            # Extract year if present
                            year_match = re.search(r'(\d{4})', doc_part)
                            if year_match:
                                year = year_match.group(1)
                                doc_name = doc_part.replace(year + '_', '').replace('_', ' ').title()
                                description = f"Document: {doc_name} ({year})"
                            else:
                                doc_name = doc_part.replace('_', ' ').title()
                                description = f"Document: {doc_name}"
                    except Exception:
                        description = "Vector document table"
                
                return {
                    'table_name': table_name,
                    'columns': column_count,
                    'rows': row_count,
                    'description': description
                }
                
        except Exception as e:
            logger.error(f"Error getting metadata for table {table_name}: {str(e)}")
            return {
                'table_name': table_name,
                'columns': 0,
                'rows': 'unknown',
                'description': ''
            }
    
    def select_table_with_llm(self, query: str, search_type: str, 
                             available_tables: List[str]) -> Optional[str]:
        """
        Use LLM to select the most appropriate table from the filtered list.
        
        Args:
            query: User's query
            search_type: 'vector' or 'sql'
            available_tables: List of filtered table names
            
        Returns:
            Selected table name or None if selection failed
        """
        if not available_tables:
            logger.warning(f"No tables available for search type: {search_type}")
            return None
        
        if len(available_tables) == 1:
            logger.info(f"Only one table available, selecting: {available_tables[0]}")
            return available_tables[0]
        
        try:
            # Get metadata for context
            table_metadata = []
            for table in available_tables:
                metadata = self._get_table_metadata(table)
                table_info = f"- {table}"
                if metadata['description']:
                    table_info += f" ({metadata['description']})"
                if isinstance(metadata['rows'], int):
                    table_info += f" - {metadata['rows']:,} rows, {metadata['columns']} columns"
                table_metadata.append(table_info)
            
            # Enhanced Turkish/English context
            if search_type == 'vector':
                search_context = """You are selecting from vector search tables that contain document embeddings in Turkish and English.
These tables store text chunks from uploaded documents (PDFs, annual reports, faaliyet raporları, etc.) with their vector representations.

Turkish Business Context:
- "Faaliyet raporu" = Annual activity report
- "Yönetim Kurulu" = Board of Directors
- "2022", "2024" = Year indicators for specific reports
- Look for document names or table descriptions that match the year or document type requested

Consider which document or content type would most likely contain the information requested."""
            else:  # sql
                search_context = """You are selecting from structured data tables for SQL queries about Turkish companies and ISO rankings.
These tables contain business data, company rankings, financial metrics for Turkish companies.

Turkish Business Context:
- "ISO 500" = Turkey's top 500 companies ranking
- "en çok kar eden" = highest profit earning
- "şirket" = company
- "gelir" = revenue
- Look for tables with "iso_" prefix that contain company rankings or financial data
- Consider year indicators (2022, 2024) in table names

Consider which business dataset would best answer the ranking/statistical question."""
            
            system_prompt = f"""{search_context}

CRITICAL INSTRUCTIONS:
1. You must return EXACTLY ONE table name from the provided list
2. Return ONLY the table name, no explanations, no additional text
3. Do not add quotes, prefixes, or suffixes
4. Choose the table most likely to contain relevant information for the query
5. Pay attention to years mentioned in the query (2022, 2024) and match with table names
6. For Turkish queries, understand the business context and document types"""

            user_prompt = f"""USER QUERY: {query}

AVAILABLE TABLES:
{chr(10).join(table_metadata)}

Select the most appropriate table name from the list above that would best answer this query.
Consider the year mentioned in the query and the type of information requested.
Return only the table name."""

            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=50,
                temperature=0.1,
                top_p=0.95
            )
            
            selected_table = response.choices[0].message.content.strip()
            
            # Clean up the response - remove any quotes, spaces, or extra text
            selected_table = selected_table.strip('"\'` \n\r')
            
            # Validate that the selected table is in our list
            if selected_table in available_tables:
                logger.info(f"LLM selected table: {selected_table} for query: '{query[:50]}...'")
                return selected_table
            else:
                # Try to find a partial match in case of minor variations
                for table in available_tables:
                    if table in selected_table or selected_table in table:
                        logger.warning(f"LLM returned '{selected_table}', using closest match: {table}")
                        return table
                
                logger.error(f"LLM returned invalid table: '{selected_table}', not in available list: {available_tables}")
                # Fallback to first table
                return available_tables[0]
            
        except Exception as e:
            logger.error(f"Error in LLM table selection: {str(e)}")
            # Fallback to first available table
            return available_tables[0] if available_tables else None

class HybridRetriever:
    """
    Enhanced hybrid retriever with LLM-based table selection.
    Combines vector similarity search with SQL query generation.
    """
    
    def __init__(self, 
                 db_config: Optional[Dict[str, str]] = None,
                 openai_api_key: Optional[str] = None,
                 embedding_model: str = "text-embedding-3-small",
                 vector_similarity_threshold: float = 0.25,
                 max_vector_results: int = 10,
                 max_sql_results: int = 50,
                 prefix_config: Optional[TablePrefixConfig] = None):
        """
        Initialize the hybrid retriever with LLM table selection.
        
        Args:
            db_config: Database configuration
            openai_api_key: OpenAI API key
            embedding_model: OpenAI embedding model to use
            vector_similarity_threshold: Maximum similarity threshold for vector search
            max_vector_results: Maximum number of vector search results
            max_sql_results: Maximum number of SQL query results
            prefix_config: Table prefix configuration
        """
        self.db_config = db_config or self._get_default_db_config()
        self.engine = self._create_engine()
        self.metadata = MetaData()
        
        # OpenAI configuration
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.embedding_model = embedding_model
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
        
        # Search parameters
        self.vector_similarity_threshold = min(vector_similarity_threshold, 0.25)
        self.max_vector_results = max_vector_results
        self.max_sql_results = max_sql_results
        
        # Table management
        self.prefix_config = prefix_config or TablePrefixConfig()
        self.table_selector = TableSelector(self.openai_api_key, self.engine)
        
        # Cache for table schemas to avoid repeated queries
        self._table_schema_cache = {}
        
        logger.info(f"HybridRetriever initialized with LLM table selection")
    
    def _get_default_db_config(self) -> Dict[str, str]:
        """Get default PostgreSQL configuration."""
        return {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': os.getenv('POSTGRES_PORT', '5432'),
            'database': os.getenv('POSTGRES_DB', 'ceo_rag_db'),
            'username': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', 'password')
        }
    
    def _create_engine(self):
        """Create SQLAlchemy engine for PostgreSQL connection."""
        password_part = f":{self.db_config['password']}" if self.db_config['password'] else ""
        connection_string = (
            f"postgresql://{self.db_config['username']}{password_part}@"
            f"{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
        )
        return create_engine(connection_string, echo=False)
    
    def _generate_embedding(self, text: str, max_retries: int = 3) -> Optional[List[float]]:
        """Generate embedding for text using OpenAI API with retry logic."""
        if not self.openai_api_key:
            logger.warning("No OpenAI API key provided, skipping embedding generation")
            return None
        
        for attempt in range(max_retries):
            try:
                response = openai.embeddings.create(
                    model=self.embedding_model,
                    input=text
                )
                return response.data[0].embedding
                
            except openai.RateLimitError:
                wait_time = (2 ** attempt) + 1
                logger.warning(f"Rate limit hit, waiting {wait_time} seconds before retry {attempt + 1}")
                time.sleep(wait_time)
                
            except Exception as e:
                logger.error(f"Error generating embedding (attempt {attempt + 1}): {str(e)}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(1)
        
        return None
    
    def _get_available_tables_by_type(self, search_type: str) -> List[str]:
        """Get available tables filtered by search type and prefixes."""
        # Get all table names
        all_tables = self.table_selector._get_all_table_names()
        
        # Get prefixes for this search type
        prefixes = self.prefix_config.get_prefixes_by_type(search_type)
        
        # Filter tables by prefixes
        filtered_tables = self.table_selector._filter_tables_by_prefixes(all_tables, prefixes)
        
        logger.info(f"Found {len(filtered_tables)} tables for {search_type} search with prefixes {prefixes}")
        return filtered_tables
    
    def _determine_query_type(self, query: str) -> str:
        """Determine if query should use vector search or SQL search."""
        try:
            system_prompt = """Analyze the user query in Turkish or English and determine if it should use:

1. "vector" - for searching through documents (faaliyet raporları, annual reports, PDFs), finding information in uploaded files, semantic search
   - Keywords: rapor, belge, faaliyet raporu, annual report, dokuman, pdf, bilgi ara, ara, bul
   - Examples: "2022 Faaliyet raporuna göre...", "dokümanlarda ara", "raporlarda bul"

2. "sql" - for structured data analysis, metrics, statistics, company rankings, financial data queries
   - Keywords: en çok, sıralama, liste, top 10, kar, gelir, revenue, profit, şirket listesi, istatistik
   - Examples: "en çok kar eden şirketler", "ISO 500 listesi", "gelir sıralaması"

Turkish Business Context:
- "Faaliyet raporu" → vector (document search)
- "ISO 500 listesi", "en çok kar eden" → sql (structured data)
- "Yönetim Kurulu toplantı" in context of rapor → vector
- "şirket sıralaması", "top 10" → sql

Return only "vector" or "sql" with no other text."""

            user_prompt = f"""Query: {query}

Should this use vector search (documents) or sql search (structured data)?"""

            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=10,
                temperature=0.1
            )
            
            query_type = response.choices[0].message.content.strip().lower()
            
            if query_type not in ['vector', 'sql']:
                # Enhanced fallback logic for Turkish
                query_lower = query.lower()
                # Vector search indicators
                vector_keywords = ['rapor', 'faaliyet', 'belge', 'dokuman', 'document', 'pdf', 'ara', 'bul', 'find', 'search']
                # SQL search indicators  
                sql_keywords = ['en çok', 'sıralama', 'liste', 'top', 'kar', 'gelir', 'revenue', 'profit', 'iso', 'şirket', 'company']
                
                vector_score = sum(1 for word in vector_keywords if word in query_lower)
                sql_score = sum(1 for word in sql_keywords if word in query_lower)
                
                if sql_score > vector_score:
                    query_type = 'sql'
                else:
                    query_type = 'vector'
            
            logger.info(f"Query type determined: {query_type} for query: '{query[:50]}...'")
            return query_type
            
        except Exception as e:
            logger.error(f"Error determining query type: {str(e)}")
            # Default fallback
            return 'vector'
    
    def vector_search(self, query: str, selected_table: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search on the selected table.
        
        Args:
            query: Search query text
            selected_table: Specific vector table to search
            
        Returns:
            List of similar chunks with similarity scores
        """
        if not self.openai_api_key:
            logger.warning("No OpenAI API key provided, cannot perform vector search")
            return []
        
        try:
            # Get query embedding
            query_embedding = self._generate_embedding(query)
            if not query_embedding:
                logger.error("Failed to generate embedding for query")
                return []
            
            # If no specific table provided, use LLM to select one
            if not selected_table:
                available_tables = self._get_available_tables_by_type('vector')
                selected_table = self.table_selector.select_table_with_llm(query, 'vector', available_tables)
                
                if not selected_table:
                    logger.warning("No vector table selected")
                    return []
            
            # Perform vector search on selected table
            try:
                embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
                
                # First, let's check if the table has any data with embeddings
                debug_sql = f"""
                SELECT COUNT(*) as total_rows,
                       COUNT(embedding) as rows_with_embeddings
                FROM "{selected_table}"
                """
                
                with self.engine.connect() as conn:
                    debug_result = conn.execute(text(debug_sql))
                    debug_row = debug_result.fetchone()
                    logger.info(f"Table {selected_table}: {debug_row.total_rows} total rows, {debug_row.rows_with_embeddings} with embeddings")
                    
                    if debug_row.rows_with_embeddings == 0:
                        logger.warning(f"No embeddings found in table {selected_table}")
                        return []
                
                # Check similarity scores without threshold first
                search_sql = f"""
                SELECT v.id, v.chunk_text, v.chunk_tokens, v.page_number, v.chunk_type,
                       v.metadata, v.embedding_model,
                       1 - (v.embedding <=> '{embedding_str}'::vector) AS similarity_score
                FROM "{selected_table}" v
                WHERE v.embedding IS NOT NULL
                ORDER BY v.embedding <=> '{embedding_str}'::vector
                LIMIT :limit
                """
                
                with self.engine.connect() as conn:
                    result = conn.execute(text(search_sql), {
                        'limit': self.max_vector_results
                    })
                    
                    all_chunks = []
                    for row in result:
                        similarity = float(row.similarity_score)
                        logger.info(f"Found chunk with similarity: {similarity:.4f}")
                        
                        chunk_metadata = {}
                        if row.metadata:
                            try:
                                if isinstance(row.metadata, str):
                                    chunk_metadata = json.loads(row.metadata)
                                elif isinstance(row.metadata, dict):
                                    chunk_metadata = row.metadata
                            except (json.JSONDecodeError, TypeError) as e:
                                logger.warning(f"Could not parse metadata for chunk {row.id}: {str(e)}")
                        
                        # Extract document name from table name for display
                        # Table name pattern: vectors_doc_YYYY_document_name
                        file_name = selected_table.replace('vectors_doc_', '').replace('_', ' ').title()
                        
                        chunk = {
                            'id': row.id,
                            'text': row.chunk_text,
                            'tokens': row.chunk_tokens,
                            'page_number': row.page_number or 1,
                            'chunk_type': row.chunk_type or 'content',
                            'file_name': file_name,
                            'file_path': f"documents/{file_name}.pdf",
                            'vector_table_name': selected_table,
                            'similarity_score': similarity,
                            'source_type': 'pdf_vector',
                            'selected_table': selected_table,
                            'embedding_model': row.embedding_model or self.embedding_model,
                            'metadata': chunk_metadata
                        }
                        all_chunks.append(chunk)
                    
                    # Filter by similarity threshold after logging all scores
                    filtered_chunks = [chunk for chunk in all_chunks if chunk['similarity_score'] >= (1.0 - self.vector_similarity_threshold)]
                    
                    logger.info(f"Found {len(all_chunks)} total chunks, {len(filtered_chunks)} above similarity threshold {1.0 - self.vector_similarity_threshold:.3f}")
                    
                    if len(filtered_chunks) == 0 and len(all_chunks) > 0:
                        # If threshold is too strict, return top results anyway
                        logger.warning(f"No chunks above threshold, returning top {min(5, len(all_chunks))} results")
                        return all_chunks[:5]
                
                logger.info(f"Vector search in {selected_table} found {len(filtered_chunks)} chunks")
                return filtered_chunks
                
            except Exception as e:
                logger.error(f"Error searching vector table {selected_table}: {str(e)}")
                return []
                
        except Exception as e:
            logger.error(f"Error in vector search: {str(e)}")
            return []
    
    def _get_table_schema(self, table_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed schema information for a table."""
        cache_key = f"schema_{table_name}"
        
        if cache_key in self._table_schema_cache:
            cached_time = self._table_schema_cache[cache_key].get('cached_at', 0)
            if time.time() - cached_time < 3600:
                return self._table_schema_cache[cache_key]['schema']
        
        try:
            with self.engine.connect() as conn:
                columns_result = conn.execute(text("""
                    SELECT column_name, data_type, is_nullable, column_default
                    FROM information_schema.columns
                    WHERE table_name = :table_name
                    AND table_schema = 'public'
                    ORDER BY ordinal_position
                """), {"table_name": table_name})
                
                columns = []
                for row in columns_result:
                    columns.append({
                        'name': row.column_name,
                        'type': row.data_type,
                        'nullable': row.is_nullable == 'YES',
                        'default': row.column_default
                    })
                
                if not columns:
                    return None
                
                count_result = conn.execute(text(f'SELECT COUNT(*) FROM "{table_name}"'))
                row_count = count_result.scalar()
                
                sample_result = conn.execute(text(f'SELECT * FROM "{table_name}" LIMIT 3'))
                sample_data = [dict(row._mapping) for row in sample_result]
                
                schema_info = {
                    'table_name': table_name,
                    'columns': columns,
                    'row_count': row_count,
                    'sample_data': sample_data
                }
                
                self._table_schema_cache[cache_key] = {
                    'schema': schema_info,
                    'cached_at': time.time()
                }
                
                return schema_info
                
        except Exception as e:
            logger.error(f"Error getting schema for table {table_name}: {str(e)}")
            return None
    
    def _generate_sql_query(self, query: str, table_name: str) -> Optional[str]:
        """
        Generate SQL query for the specific selected table with Turkish language support.
        
        Args:
            query: User's natural language query (Turkish or English)
            table_name: Selected table name
            
        Returns:
            Generated SQL query string or None if failed
        """
        if not self.openai_api_key:
            logger.warning("No OpenAI API key provided, cannot generate SQL query")
            return None
        
        # Get table schema
        schema = self._get_table_schema(table_name)
        if not schema:
            logger.error(f"Could not get schema for table {table_name}")
            return None
        
        # Prepare schema description
        column_descriptions = []
        for col in schema['columns']:
            col_desc = f'"{col["name"]}" ({col["type"]})'
            if not col['nullable']:
                col_desc += ' NOT NULL'
            column_descriptions.append(col_desc)
        
        schema_description = f"""
TABLE: "{table_name}"
- Total records: {schema['row_count']:,}
- Columns: {', '.join(column_descriptions)}
"""
        
        if schema['sample_data']:
            schema_description += f"\nSample data: {schema['sample_data'][:2]}"
        
        system_prompt = """You are an expert SQL analyst specializing in Turkish business data and ISO 500 company rankings.
Generate precise PostgreSQL queries for Turkish business intelligence queries.

CRITICAL DATA HANDLING RULES:
1. Financial columns may contain non-numeric values like "-" representing missing data
2. Use NULLIF and regex to handle non-numeric values safely
3. For ordering by financial amounts, use this pattern:
   ORDER BY NULLIF(regexp_replace(column_name, '[^0-9]', '', 'g'), '')::BIGINT DESC NULLS LAST

Turkish Business Terms:
- "en çok kar eden" = highest profit earning → ORDER BY profit DESC
- "şirket" = company
- "gelir" = revenue  
- "kar" = profit
- "sıralama" = ranking → use ORDER BY
- "top 10" = LIMIT 10
- "hangileri" = which ones → SELECT company names/info

Query Patterns:
- For "en çok X eden Y şirket": SELECT ... ORDER BY (safe_numeric_conversion) DESC LIMIT Y
- For rankings: Always include ORDER BY with NULLS LAST and appropriate LIMIT
- For "hangileri": Focus on company names and relevant metrics

Safe Numeric Conversion Pattern:
NULLIF(regexp_replace(column_name, '[^0-9]', '', 'g'), '')::BIGINT

Return ONLY the SQL query without explanations or formatting."""
        
        user_prompt = f"""DATABASE SCHEMA:
{schema_description}

USER QUERY (Turkish/English): {query}

Generate a PostgreSQL SELECT query using safe numeric conversion for financial columns.

IMPORTANT: Financial columns may contain "-" or other non-numeric values. Use this safe pattern:
ORDER BY NULLIF(regexp_replace(column_name, '[^0-9]', '', 'g'), '')::BIGINT DESC NULLS LAST

Example for "en çok kar eden şirketler":
SELECT kurulus_adi, kar_column 
FROM "{table_name}" 
ORDER BY NULLIF(regexp_replace(kar_column, '[^0-9]', '', 'g'), '')::BIGINT DESC NULLS LAST 
LIMIT 10;

For Turkish queries about rankings, always use the safe numeric conversion pattern above.
Include company names and relevant financial metrics.
LIMIT results to maximum {self.max_sql_results} rows.
Return only the SQL query."""

        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            sql_query = response.choices[0].message.content.strip()
            sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
            
            if not sql_query.upper().startswith('SELECT'):
                logger.warning("Generated query does not start with SELECT")
                return None
            
            # Basic security check
            dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 'TRUNCATE']
            sql_upper = sql_query.upper()
            for keyword in dangerous_keywords:
                if keyword in sql_upper:
                    logger.warning(f"Generated query contains dangerous keyword: {keyword}")
                    return None
            
            logger.info(f"Generated SQL query for table {table_name}: {sql_query[:100]}...")
            return sql_query
            
        except Exception as e:
            logger.error(f"Error generating SQL query: {str(e)}")
            return None
    
    def sql_search(self, query: str, selected_table: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Perform SQL-based search on the selected table.
        
        Args:
            query: Natural language query
            selected_table: Specific table to search
            
        Returns:
            List of query results
        """
        try:
            # If no specific table provided, use LLM to select one
            if not selected_table:
                available_tables = self._get_available_tables_by_type('sql')
                selected_table = self.table_selector.select_table_with_llm(query, 'sql', available_tables)
                
                if not selected_table:
                    logger.warning("No SQL table selected")
                    return []
            
            # Generate SQL query for the selected table
            sql_query = self._generate_sql_query(query, selected_table)
            if not sql_query:
                logger.warning("Failed to generate SQL query")
                return []
            
            # Execute the query
            with self.engine.connect() as conn:
                try:
                    result = conn.execute(text(sql_query))
                    
                    results = []
                    for row in result:
                        row_dict = dict(row._mapping)
                        row_dict['source_type'] = 'sql_query'
                        row_dict['generated_sql'] = sql_query
                        row_dict['selected_table'] = selected_table
                        results.append(row_dict)
                    
                    logger.info(f"SQL search in {selected_table} returned {len(results)} rows")
                    return results[:self.max_sql_results]
                    
                except Exception as e:
                    logger.error(f"Error executing SQL query on {selected_table}: {str(e)}")
                    logger.error(f"Query was: {sql_query}")
                    return []
            
        except Exception as e:
            logger.error(f"Error in SQL search: {str(e)}")
            return []
    
    def hybrid_retrieve(self, query: str,
                       force_search_type: Optional[str] = None,
                       force_table: Optional[str] = None) -> RetrievalResult:
        """
        Perform hybrid retrieval with LLM-based table selection.
        
        Args:
            query: Search query
            force_search_type: Force specific search type ('vector' or 'sql')
            force_table: Force specific table selection
            
        Returns:
            RetrievalResult containing results and metadata
        """
        start_time = time.time()
        
        vector_chunks = []
        sql_results = []
        selected_tables = {}
        
        try:
            # Determine query type unless forced
            if force_search_type:
                query_type = force_search_type
            else:
                query_type = self._determine_query_type(query)
            
            logger.info(f"Processing query as type: {query_type}")
            
            # Perform search based on determined type
            if query_type == 'vector':
                vector_chunks = self.vector_search(query, force_table)
                if vector_chunks:
                    selected_tables['vector'] = vector_chunks[0].get('selected_table')
            
            elif query_type == 'sql':
                sql_results = self.sql_search(query, force_table)
                if sql_results:
                    selected_tables['sql'] = sql_results[0].get('selected_table')
            
            else:
                logger.warning(f"Unknown query type: {query_type}, defaulting to vector search")
                vector_chunks = self.vector_search(query, force_table)
                if vector_chunks:
                    selected_tables['vector'] = vector_chunks[0].get('selected_table')
            
            total_time = time.time() - start_time
            
            # Prepare metadata
            metadata = {
                'total_retrieval_time': total_time,
                'query_type': query_type,
                'vector_results_count': len(vector_chunks),
                'sql_results_count': len(sql_results),
                'selected_tables': selected_tables,
                'table_selection_method': 'llm' if not force_table else 'forced',
                'similarity_threshold': self.vector_similarity_threshold,
                'max_results': {
                    'vector': self.max_vector_results,
                    'sql': self.max_sql_results
                }
            }
            
            # Query information
            query_info = {
                'original_query': query,
                'detected_type': query_type,
                'forced_type': force_search_type,
                'forced_table': force_table,
                'results_found': len(vector_chunks) > 0 or len(sql_results) > 0
            }
            
            return RetrievalResult(
                vector_chunks=vector_chunks,
                sql_results=sql_results,
                metadata=metadata,
                query_info=query_info
            )
            
        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {str(e)}")
            return RetrievalResult(
                vector_chunks=[],
                sql_results=[],
                metadata={'error': str(e), 'total_retrieval_time': time.time() - start_time},
                query_info={'original_query': query, 'error': str(e)}
            )
    
    def get_available_sources(self) -> Dict[str, Any]:
        """Get information about all available data sources organized by type."""
        try:
            all_tables = self.table_selector._get_all_table_names()
            
            sources_by_type = {}
            
            # Organize tables by search type
            for search_type in ['vector', 'sql']:
                prefixes = self.prefix_config.get_prefixes_by_type(search_type)
                filtered_tables = self.table_selector._filter_tables_by_prefixes(all_tables, prefixes)
                
                table_details = []
                for table in filtered_tables:
                    metadata = self.table_selector._get_table_metadata(table)
                    table_details.append(metadata)
                
                sources_by_type[search_type] = {
                    'count': len(filtered_tables),
                    'prefixes': prefixes,
                    'tables': table_details
                }
            
            return {
                'total_tables': len(all_tables),
                'sources_by_type': sources_by_type,
                'prefix_config': {
                    prefix: config for prefix, config in self.prefix_config.prefixes.items()
                },
                'search_parameters': {
                    'vector_similarity_threshold': self.vector_similarity_threshold,
                    'max_vector_results': self.max_vector_results,
                    'max_sql_results': self.max_sql_results
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting available sources: {str(e)}")
            return {}
    
    def add_table_prefix(self, prefix: str, search_type: str, description: str):
        """
        Add a new table prefix configuration.
        
        Args:
            prefix: Table prefix (e.g., 'analytics_')
            search_type: 'vector' or 'sql'
            description: Human-readable description
        """
        self.prefix_config.add_prefix(prefix, search_type, description)
        logger.info(f"Added new prefix: {prefix} for {search_type} search")
    
    def list_tables_by_prefix(self, prefix: str) -> List[str]:
        """
        List all tables matching a specific prefix.
        
        Args:
            prefix: Table prefix to filter by
            
        Returns:
            List of matching table names
        """
        all_tables = self.table_selector._get_all_table_names()
        return [table for table in all_tables if table.startswith(prefix)]
    
    def test_table_selection(self, query: str, search_type: str) -> Dict[str, Any]:
        """
        Test table selection for a given query without performing actual search.
        
        Args:
            query: Test query
            search_type: 'vector' or 'sql'
            
        Returns:
            Dictionary with selection results
        """
        try:
            available_tables = self._get_available_tables_by_type(search_type)
            selected_table = self.table_selector.select_table_with_llm(query, search_type, available_tables)
            
            return {
                'query': query,
                'search_type': search_type,
                'available_tables': available_tables,
                'selected_table': selected_table,
                'selection_successful': selected_table is not None
            }
            
        except Exception as e:
            return {
                'query': query,
                'search_type': search_type,
                'error': str(e),
                'selection_successful': False
            }
    
    def clear_cache(self):
        """Clear all internal caches."""
        self._table_schema_cache.clear()
        self.table_selector._table_cache.clear()
        logger.info("Retriever caches cleared")
    
    def update_search_parameters(self, 
                                similarity_threshold: Optional[float] = None,
                                max_vector_results: Optional[int] = None,
                                max_sql_results: Optional[int] = None):
        """
        Update search parameters.
        
        Args:
            similarity_threshold: New similarity threshold (max 0.25)
            max_vector_results: New max vector results limit
            max_sql_results: New max SQL results limit
        """
        if similarity_threshold is not None:
            self.vector_similarity_threshold = min(similarity_threshold, 0.25)
            logger.info(f"Updated similarity threshold to: {self.vector_similarity_threshold}")
        
        if max_vector_results is not None:
            self.max_vector_results = max_vector_results
            logger.info(f"Updated max vector results to: {self.max_vector_results}")
        
        if max_sql_results is not None:
            self.max_sql_results = max_sql_results
            logger.info(f"Updated max SQL results to: {self.max_sql_results}")
    
    def close(self):
        """Close database connections."""
        if hasattr(self, 'engine'):
            self.engine.dispose()


# Convenience functions for easy integration
def create_retriever(db_config: Optional[Dict[str, str]] = None,
                    openai_api_key: Optional[str] = None,
                    embedding_model: str = "text-embedding-3-small",
                    custom_prefixes: Optional[Dict[str, Dict[str, str]]] = None) -> HybridRetriever:
    """
    Create a HybridRetriever instance with default or custom settings.
    
    Args:
        db_config: Database configuration
        openai_api_key: OpenAI API key
        embedding_model: OpenAI embedding model to use
        custom_prefixes: Custom prefix configuration
            Format: {'prefix_': {'search_type': 'vector|sql', 'description': 'desc'}}
        
    Returns:
        Configured HybridRetriever instance
    """
    # Set up prefix configuration
    prefix_config = TablePrefixConfig()
    
    if custom_prefixes:
        for prefix, config in custom_prefixes.items():
            prefix_config.add_prefix(
                prefix, 
                config['search_type'], 
                config['description']
            )
    
    return HybridRetriever(
        db_config=db_config,
        openai_api_key=openai_api_key,
        embedding_model=embedding_model,
        vector_similarity_threshold=0.25,
        max_vector_results=10,
        max_sql_results=50,
        prefix_config=prefix_config
    )


# Example usage and testing
if __name__ == "__main__":
    import time
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize retriever
    retriever = create_retriever(
        openai_api_key=os.getenv('OPENAI_API_KEY')
    )
    
    try:
        print("Enhanced HybridRetriever with LLM Table Selection initialized!")
        
        # Show available sources
        sources = retriever.get_available_sources()
        print(f"\nAvailable Sources:")
        for search_type, info in sources.get('sources_by_type', {}).items():
            print(f"{search_type.upper()} Search:")
            print(f"  - Prefixes: {info['prefixes']}")
            print(f"  - Tables: {info['count']}")
            for table in info['tables']:
                print(f"    * {table['table_name']} ({table['rows']} rows)")
        
        # Test queries for different types - Turkish business queries
        test_queries = [
            ("2022 Faaliyet raporuna göre Yönetim Kurulu kaç kere toplandı?", "Should use vector search - searching in 2022 annual report document"),
            ("2022 ISO belgesine göre en çok kar eden 10 şirket hangisi?", "Should use SQL search - ranking query for top 10 profitable companies"),
            ("2024 Faaliyet raporuna göre Yönetim Kurulu kaç kere toplandı", "Should use vector search - searching in 2024 annual report document"), 
            ("2024 ISO belgesine göre en çok kar eden 10 şirket hangisi?", "Should use SQL search - ranking query for top 10 profitable companies"),
        ]
        
        print("\n" + "="*70)
        print("TESTING LLM-BASED QUERY TYPE DETECTION AND TABLE SELECTION")
        print("="*70)
        
        for query, expected in test_queries:
            print(f"\nQuery: '{query}'")
            print(f"Expected: {expected}")
            print("-" * 50)
            
            # Test query type determination
            query_type = retriever._determine_query_type(query)
            print(f"Detected Type: {query_type}")
            
            # Test table selection
            selection_test = retriever.test_table_selection(query, query_type)
            print(f"Available Tables: {selection_test.get('available_tables', [])}")
            print(f"Selected Table: {selection_test.get('selected_table', 'None')}")
            print(f"Selection Success: {selection_test.get('selection_successful', False)}")
            
            time.sleep(1)  # Brief pause between tests
        
        # Test full hybrid retrieval with Turkish query
        print("\n" + "="*70)
        print("TESTING FULL HYBRID RETRIEVAL")
        print("="*70)
        
        test_query = "2024 ISO belgesine göre en çok gelir elde eden şirketler hangileri?"
        print(f"\nTesting full retrieval for: '{test_query}'")
        
        result = retriever.hybrid_retrieve(test_query)
        
        print(f"\nResults:")
        print(f"Query Type: {result.metadata.get('query_type')}")
        print(f"Selected Tables: {result.metadata.get('selected_tables')}")
        print(f"Vector Results: {len(result.vector_chunks)}")
        print(f"SQL Results: {len(result.sql_results)}")
        print(f"Processing Time: {result.metadata.get('total_retrieval_time', 0):.2f}s")
        
        if result.vector_chunks:
            print(f"\nSample Vector Result:")
            chunk = result.vector_chunks[0]
            print(f"  File: {chunk.get('file_name')}")
            print(f"  Table: {chunk.get('selected_table')}")
            print(f"  Similarity: {chunk.get('similarity_score', 0):.3f}")
            print(f"  Text Preview: {chunk.get('text', '')[:100]}...")
        
        if result.sql_results:
            print(f"\nSample SQL Result:")
            sql_result = result.sql_results[0]
            print(f"  Table: {sql_result.get('selected_table')}")
            print(f"  Generated SQL: {sql_result.get('generated_sql', '')[:100]}...")
            print(f"  Columns: {list(k for k in sql_result.keys() if k not in ['source_type', 'generated_sql', 'selected_table'])[:5]}")
            # Show sample data values
            sample_values = {k: v for k, v in sql_result.items() if k not in ['source_type', 'generated_sql', 'selected_table']}
            if sample_values:
                first_items = list(sample_values.items())[:3]
                print(f"  Sample Values: {dict(first_items)}")
        
        # Additional test with vector search query
        print(f"\nTesting vector search query:")
        vector_query = "2022 Faaliyet raporuna göre şirketin finansal durumu nasıl?"
        vector_result = retriever.hybrid_retrieve(vector_query)
        
        print(f"Query Type: {vector_result.metadata.get('query_type')}")
        print(f"Selected Tables: {vector_result.metadata.get('selected_tables')}")
        print(f"Vector Results Found: {len(vector_result.vector_chunks)}")
        
        if vector_result.vector_chunks:
            chunk = vector_result.vector_chunks[0]
            print(f"  Best Match - File: {chunk.get('file_name')}")
            print(f"  Similarity: {chunk.get('similarity_score', 0):.3f}")
            print(f"  Page: {chunk.get('page_number')}")
            print(f"  Text Preview: {chunk.get('text', '')[:150]}...")
        
        # Test adding custom prefixes
        print("\n" + "="*70)
        print("TESTING CUSTOM PREFIX ADDITION")
        print("="*70)
        
        retriever.add_table_prefix('analytics_', 'sql', 'Analytics and metrics tables')
        retriever.add_table_prefix('logs_', 'vector', 'Application log documents')
        
        updated_sources = retriever.get_available_sources()
        print("Updated Prefix Configuration:")
        for prefix, config in updated_sources.get('prefix_config', {}).items():
            print(f"  {prefix}: {config['search_type']} - {config['description']}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        retriever.close()
        print("\nRetriever closed.")


# Export main classes and functions
__all__ = [
    'HybridRetriever',
    'RetrievalResult',
    'TablePrefixConfig',
    'TableSelector',
    'create_retriever'
]
