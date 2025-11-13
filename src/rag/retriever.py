import logging
import json
import re
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
from openai import OpenAI  # NOTE: used for default client if none is passed
from sqlalchemy import create_engine, text, MetaData
import pandas as pd
import os
import time
import hashlib
from sentence_transformers import SentenceTransformer

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
    
    def __init__(self, openai_client: Optional[Any], engine):
        # If no client provided, fall back to default OpenAI client (respects env)
        self.client = openai_client or OpenAI()
        self.engine = engine
        
        # Cache for table lists to avoid repeated queries
        self._table_cache = {}
        self._cache_expiry = 300  # 5 minutes
    
    def _get_all_table_names(self) -> List[str]:
        """Get all table names from the database."""
        cache_key = "all_tables"
        now = time.time()
        
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
                columns_result = conn.execute(text("""
                    SELECT COUNT(*) as column_count
                    FROM information_schema.columns
                    WHERE table_name = :table_name
                    AND table_schema = 'public'
                """), {"table_name": table_name})
                
                column_count = columns_result.scalar() or 0
                
                try:
                    rows_result = conn.execute(text(f'SELECT COUNT(*) FROM "{table_name}"'))
                    row_count = rows_result.scalar() or 0
                except Exception:
                    row_count = "unknown"
                
                description = ""
                if table_name.startswith('vectors_'):
                    try:
                        if 'vectors_doc_' in table_name:
                            doc_part = table_name.replace('vectors_doc_', '')
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
        """
        if not available_tables:
            logger.warning(f"No tables available for search type: {search_type}")
            return None
        
        if len(available_tables) == 1:
            logger.info(f"Only one table available, selecting: {available_tables[0]}")
            return available_tables[0]
        
        try:
            table_metadata = []
            for table in available_tables:
                metadata = self._get_table_metadata(table)
                table_info = f"- {table}"
                if metadata['description']:
                    table_info += f" ({metadata['description']})"
                if isinstance(metadata['rows'], int):
                    table_info += f" - {metadata['rows']:,} rows, {metadata['columns']} columns"
                table_metadata.append(table_info)
            
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

            # *** LM Studio / OpenAI-compatible endpoint ***
            response = self.client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=50,
                temperature=0.1,
                top_p=0.95
            )
            
            selected_table = response.choices[0].message.content.strip()
            selected_table = selected_table.strip('"\'` \n\r')
            
            if selected_table in available_tables:
                logger.info(f"LLM selected table: {selected_table} for query: '{query[:50]}...'")
                return selected_table
            else:
                for table in available_tables:
                    if table in selected_table or selected_table in table:
                        logger.warning(f"LLM returned '{selected_table}', using closest match: {table}")
                        return table
                
                logger.error(f"LLM returned invalid table: '{selected_table}', not in available list: {available_tables}")
                return available_tables[0]
            
        except Exception as e:
            logger.error(f"Error in LLM table selection: {str(e)}")
            return available_tables[0] if available_tables else None

class HybridRetriever:
    """
    Enhanced hybrid retriever with LLM-based table selection using Snowflake Arctic Embed.
    Combines vector similarity search with SQL query generation.
    """
    
    def __init__(self, 
                 db_config: Optional[Dict[str, str]] = None,
                 openai_api_key: Optional[str] = None,  # kept for backward compat (unused if client is provided)
                 embedding_model: str = "Snowflake/snowflake-arctic-embed-l",
                 vector_similarity_threshold: float = 0.25,
                 max_vector_results: int = 10,
                 max_sql_results: int = 50,
                 prefix_config: Optional[TablePrefixConfig] = None,
                 openai_client: Optional[Any] = None):
        """
        Initialize the hybrid retriever with LLM table selection and Snowflake embeddings.
        """
        self.db_config = db_config or self._get_default_db_config()
        self.engine = self._create_engine()
        self.metadata = MetaData()
        
        # --- LLM CLIENT ---
        # Prefer explicit client (LM Studio) if provided; otherwise make a default client
        self.client = openai_client or OpenAI()
        
        # Initialize Snowflake Arctic Embed model for embeddings
        logger.info(f"Loading Snowflake embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_model_name = embedding_model
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")
        
        # Search parameters
        self.vector_similarity_threshold = min(vector_similarity_threshold, 0.25)
        self.max_vector_results = max_vector_results
        self.max_sql_results = max_sql_results
        
        # Table management
        self.prefix_config = prefix_config or TablePrefixConfig()
        self.table_selector = TableSelector(self.client, self.engine)
        
        # Cache for table schemas to avoid repeated queries
        self._table_schema_cache = {}
        
        logger.info("HybridRetriever initialized with Snowflake Arctic Embed and LLM table selection (LM Studio client ready).")
    
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
    
    def _generate_embedding(self, text: str, is_query: bool = False) -> Optional[List[float]]:
        """
        Generate embedding for text using Snowflake Arctic Embed model.
        """
        try:
            if is_query:
                embedding = self.embedding_model.encode(
                    text, 
                    prompt_name="query",
                    convert_to_numpy=True
                )
            else:
                embedding = self.embedding_model.encode(
                    text,
                    convert_to_numpy=True
                )
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return None
    
    def _get_available_tables_by_type(self, search_type: str) -> List[str]:
        """Get available tables filtered by search type and prefixes."""
        all_tables = self.table_selector._get_all_table_names()
        prefixes = self.prefix_config.get_prefixes_by_type(search_type)
        filtered_tables = self.table_selector._filter_tables_by_prefixes(all_tables, prefixes)
        logger.info(f"Found {len(filtered_tables)} tables for {search_type} search with prefixes {prefixes}")
        return filtered_tables
    
    def _determine_query_type(self, query: str) -> str:
        """Determine if query should use vector search or SQL search."""
        try:
            system_prompt = """Analyze the user query in Turkish or English and determine if it should use:

1. "vector" - for searching through documents (faaliyet raporları, annual reports, PDFs), finding information in uploaded files, semantic search
2. "sql" - for structured data analysis, metrics, statistics, company rankings, financial data queries

Return only "vector" or "sql"."""
            user_prompt = f"""Query: {query}

Should this use vector search (documents) or sql search (structured data)?"""

            resp = self.client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=10,
                temperature=0.1
            )
            query_type = resp.choices[0].message.content.strip().lower()
            if query_type not in ['vector', 'sql']:
                query_lower = query.lower()
                vector_keywords = ['rapor', 'faaliyet', 'belge', 'dokuman', 'document', 'pdf', 'ara', 'bul', 'find', 'search']
                sql_keywords  = ['en çok', 'sıralama', 'liste', 'top', 'kar', 'gelir', 'revenue', 'profit', 'iso', 'şirket', 'company']
                vector_score = sum(1 for w in vector_keywords if w in query_lower)
                sql_score = sum(1 for w in sql_keywords if w in query_lower)
                query_type = 'sql' if sql_score > vector_score else 'vector'
            logger.info(f"Query type determined: {query_type} for query: '{query[:50]}...'")
            return query_type
        except Exception as e:
            logger.error(f"Error determining query type: {str(e)}")
            return 'vector'
    
    def vector_search(self, query: str, selected_table: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search on the selected table using Snowflake Arctic Embed.
        """
        try:
            logger.info("Generating query embedding with Snowflake Arctic Embed...")
            query_embedding = self._generate_embedding(query, is_query=True)
            if not query_embedding:
                logger.error("Failed to generate embedding for query")
                return []
            
            if not selected_table:
                available_tables = self._get_available_tables_by_type('vector')
                selected_table = self.table_selector.select_table_with_llm(query, 'vector', available_tables)
                if not selected_table:
                    logger.warning("No vector table selected")
                    return []
            
            try:
                embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
                
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
                    result = conn.execute(text(search_sql), {'limit': self.max_vector_results})
                    
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
                            'embedding_model': row.embedding_model or self.embedding_model_name,
                            'metadata': chunk_metadata
                        }
                        all_chunks.append(chunk)
                    
                    filtered_chunks = [c for c in all_chunks if c['similarity_score'] >= (1.0 - self.vector_similarity_threshold)]
                    logger.info(f"Found {len(all_chunks)} total chunks, {len(filtered_chunks)} above similarity threshold {1.0 - self.vector_similarity_threshold:.3f}")
                    
                    if len(filtered_chunks) == 0 and len(all_chunks) > 0:
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
        """
        # If you want to disable LLM SQL generation in LM Studio, you can return None here.
        schema = self._get_table_schema(table_name)
        if not schema:
            logger.error(f"Could not get schema for table {table_name}")
            return None
        
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
4.DO NOT try to CREATE or DROP anythıng.

Return ONLY the SQL query without explanations or formatting."""
        
        user_prompt = f"""DATABASE SCHEMA:
{schema_description}

USER QUERY (Turkish/English): {query}

Generate a PostgreSQL SELECT query using safe numeric conversion for financial columns.
LIMIT results to maximum {self.max_sql_results} rows.
Return only the SQL query."""

        try:
            resp = self.client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            sql_query = resp.choices[0].message.content.strip()
            sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
            
            if not sql_query.upper().startswith('SELECT'):
                logger.warning("Generated query does not start with SELECT")
                return None
            
            dangerous = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 'TRUNCATE']
            upper = sql_query.upper()
            for kw in dangerous:
                if kw in upper:
                    logger.warning(f"Generated query contains dangerous keyword: {kw}")
                    return None
            
            logger.info(f"Generated SQL for table {table_name}: {sql_query[:100]}...")
            return sql_query
        except Exception as e:
            logger.error(f"Error generating SQL query: {str(e)}")
            return None
    
    def sql_search(self, query: str, selected_table: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Perform SQL-based search on the selected table.
        """
        try:
            if not selected_table:
                available_tables = self._get_available_tables_by_type('sql')
                selected_table = self.table_selector.select_table_with_llm(query, 'sql', available_tables)
                if not selected_table:
                    logger.warning("No SQL table selected")
                    return []
            
            sql_query = self._generate_sql_query(query, selected_table)
            if not sql_query:
                logger.warning("Failed to generate SQL query")
                return []
            
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
                    logger.error(f"Error executing SQL on {selected_table}: {str(e)}")
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
        """
        start_time = time.time()
        vector_chunks: List[Dict[str, Any]] = []
        sql_results: List[Dict[str, Any]] = []
        selected_tables: Dict[str, Any] = {}
        
        try:
            if force_search_type:
                query_type = force_search_type
            else:
                query_type = self._determine_query_type(query)
            
            logger.info(f"Processing query as type: {query_type}")
            
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
                },
                'embedding_model': self.embedding_model_name
            }
            
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
                },
                'embedding_model': self.embedding_model_name,
                'embedding_dimension': self.embedding_dim
            }
        except Exception as e:
            logger.error(f"Error getting available sources: {str(e)}")
            return {}
    
    def add_table_prefix(self, prefix: str, search_type: str, description: str):
        """Add a new table prefix configuration."""
        self.prefix_config.add_prefix(prefix, search_type, description)
        logger.info(f"Added new prefix: {prefix} for {search_type} search")
    
    def list_tables_by_prefix(self, prefix: str) -> List[str]:
        """List all tables matching a specific prefix."""
        all_tables = self.table_selector._get_all_table_names()
        return [table for table in all_tables if table.startswith(prefix)]
    
    def test_table_selection(self, query: str, search_type: str) -> Dict[str, Any]:
        """Test table selection for a given query without performing actual search."""
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
        """Update search parameters."""
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
                    embedding_model: str = "Snowflake/snowflake-arctic-embed-l",
                    custom_prefixes: Optional[Dict[str, Dict[str, str]]] = None,
                    openai_client: Optional[Any] = None) -> HybridRetriever:
    """
    Create a HybridRetriever instance with Snowflake Arctic Embed.

    IMPORTANT: Pass `openai_client=OpenAI(base_url=..., api_key=...)` to use LM Studio.
    """
    prefix_config = TablePrefixConfig()
    if custom_prefixes:
        for prefix, config in custom_prefixes.items():
            prefix_config.add_prefix(prefix, config['search_type'], config['description'])
    
    return HybridRetriever(
        db_config=db_config,
        openai_api_key=openai_api_key,          # kept for backward-compat; not used when client is provided
        embedding_model=embedding_model,
        vector_similarity_threshold=0.25,
        max_vector_results=10,
        max_sql_results=50,
        prefix_config=prefix_config,
        openai_client=openai_client             # <-- LM Studio or other OpenAI-compatible client
    )


# Example usage and testing
if __name__ == "__main__":
    import time
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Example LM Studio client (adjust base_url/port as needed)
    lm_client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

    retriever = create_retriever(
        openai_client=lm_client,
        embedding_model="Snowflake/snowflake-arctic-embed-l"
    )
    try:
        print("HybridRetriever ready. Embedding model:", retriever.embedding_model_name)
        sources = retriever.get_available_sources()
        print("Vector tables:", sources.get('sources_by_type', {}).get('vector', {}).get('count'))
        print("SQL tables:",    sources.get('sources_by_type', {}).get('sql', {}).get('count'))
    finally:
        retriever.close()

