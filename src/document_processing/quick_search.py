#!/usr/bin/env python3
"""
Query Search Script
Search vector tables and return chunks with metadata
Usage: python query_search.py "your query here"
"""

import argparse
import sys
import os
import json
from sqlalchemy import create_engine, text
import openai
from typing import List, Dict, Any

class QuerySearcher:
    def __init__(self):
        """Initialize the query searcher."""
        self.db_config = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': os.getenv('POSTGRES_PORT', '5432'),
            'database': os.getenv('POSTGRES_DB', 'ceo_rag_db'),
            'username': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', 'password')
        }
        
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            print("❌ Error: OPENAI_API_KEY environment variable not set")
            sys.exit(1)
        
        openai.api_key = self.openai_api_key
        
        # Create database connection
        password_part = f":{self.db_config['password']}" if self.db_config['password'] else ""
        connection_string = (
            f"postgresql://{self.db_config['username']}{password_part}@"
            f"{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
        )
        self.engine = create_engine(connection_string, echo=False)
    
    def generate_embedding(self, query: str) -> List[float]:
        """Generate embedding for the query."""
        try:
            response = openai.embeddings.create(
                model="text-embedding-3-small",
                input=query
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"❌ Error generating embedding: {e}")
            sys.exit(1)
    
    def get_available_tables(self) -> List[Dict[str, Any]]:
        """Get list of available vector tables."""
        try:
            with self.engine.connect() as conn:
                # Get documents with vector tables
                docs = conn.execute(text("""
                    SELECT 
                        d.vector_table_name,
                        d.file_name,
                        d.total_chunks,
                        d.processed_at
                    FROM pdf_documents d
                    WHERE d.vector_table_created = TRUE 
                    AND d.processing_status = 'completed'
                    ORDER BY d.processed_at DESC
                """)).fetchall()
                
                tables = []
                for doc in docs:
                    # Check if table actually exists and has embeddings
                    try:
                        stats = conn.execute(text(f"""
                            SELECT 
                                COUNT(*) as total_chunks,
                                COUNT(embedding) as chunks_with_embeddings
                            FROM "{doc.vector_table_name}"
                        """)).fetchone()
                        
                        if stats.chunks_with_embeddings > 0:
                            tables.append({
                                'table_name': doc.vector_table_name,
                                'file_name': doc.file_name,
                                'total_chunks': stats.total_chunks,
                                'chunks_with_embeddings': stats.chunks_with_embeddings,
                                'processed_at': doc.processed_at
                            })
                    except Exception:
                        continue
                
                return tables
        except Exception as e:
            print(f"❌ Error getting available tables: {e}")
            return []
    
    def search_table(self, table_name: str, query_embedding: List[float], 
                    limit: int = 5, similarity_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Search a specific table for similar chunks."""
        try:
            with self.engine.connect() as conn:
                # Convert embedding to string format for PostgreSQL
                embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
                
                results = conn.execute(text(f"""
                    SELECT 
                        chunk_index,
                        chunk_text,
                        chunk_tokens,
                        page_number,
                        chunk_type,
                        metadata,
                        embedding_model,
                        created_at,
                        1 - (embedding <=> :embedding_vector) AS similarity_score
                    FROM "{table_name}"
                    WHERE embedding IS NOT NULL
                    AND 1 - (embedding <=> :embedding_vector) >= :threshold
                    ORDER BY embedding <=> :embedding_vector
                    LIMIT :limit
                """), {
                    "embedding_vector": embedding_str,
                    "threshold": similarity_threshold,
                    "limit": limit
                }).fetchall()
                
                chunks = []
                for row in results:
                    # Parse metadata JSON
                    metadata = {}
                    if row.metadata:
                        try:
                            metadata = json.loads(row.metadata)
                        except:
                            metadata = {"raw_metadata": str(row.metadata)}
                    
                    chunk = {
                        'chunk_index': row.chunk_index,
                        'chunk_text': row.chunk_text,
                        'chunk_tokens': row.chunk_tokens,
                        'page_number': row.page_number,
                        'chunk_type': row.chunk_type,
                        'similarity_score': round(float(row.similarity_score), 4),
                        'embedding_model': row.embedding_model,
                        'created_at': row.created_at.isoformat() if row.created_at else None,
                        'metadata': metadata,
                        'table_name': table_name
                    }
                    chunks.append(chunk)
                
                return chunks
                
        except Exception as e:
            print(f"❌ Error searching table {table_name}: {e}")
            return []
    
    def search_all_tables(self, query: str, limit: int = 5, 
                         similarity_threshold: float = 0.5, 
                         specific_table: str = None) -> List[Dict[str, Any]]:
        """Search all available tables or a specific table."""
        
        print(f"🔍 Searching for: '{query}'")
        print(f"📊 Parameters: limit={limit}, threshold={similarity_threshold}")
        
        # Generate embedding
        print("🧮 Generating embedding...")
        query_embedding = self.generate_embedding(query)
        
        # Get available tables
        if specific_table:
            # Check if specific table exists
            available_tables = self.get_available_tables()
            target_tables = [t for t in available_tables if t['table_name'] == specific_table]
            if not target_tables:
                print(f"❌ Table '{specific_table}' not found or has no embeddings")
                return []
            tables_to_search = target_tables
        else:
            tables_to_search = self.get_available_tables()
        
        if not tables_to_search:
            print("❌ No tables with embeddings found")
            return []
        
        print(f"📋 Searching {len(tables_to_search)} table(s):")
        for table in tables_to_search:
            print(f"   - {table['file_name']} ({table['chunks_with_embeddings']} embeddings)")
        
        # Search each table
        all_results = []
        for table_info in tables_to_search:
            table_name = table_info['table_name']
            print(f"\n🔎 Searching {table_info['file_name']}...")
            
            results = self.search_table(table_name, query_embedding, limit, similarity_threshold)
            
            # Add table info to each result
            for result in results:
                result['source_file'] = table_info['file_name']
                result['source_processed_at'] = table_info['processed_at'].isoformat() if table_info['processed_at'] else None
            
            all_results.extend(results)
            print(f"   Found {len(results)} results")
        
        # Sort all results by similarity score
        all_results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Return top results
        top_results = all_results[:limit]
        
        print(f"\n✅ Total results found: {len(all_results)}")
        print(f"📤 Returning top {len(top_results)} results")
        
        return top_results
    
    def format_results(self, results: List[Dict[str, Any]], output_format: str = 'detailed') -> str:
        """Format results for display."""
        if not results:
            return "No results found."
        
        if output_format == 'json':
            return json.dumps(results, indent=2, ensure_ascii=False)
        
        elif output_format == 'brief':
            output = []
            for i, result in enumerate(results, 1):
                output.append(f"{i}. Similarity: {result['similarity_score']:.3f} | "
                            f"Page: {result['page_number']} | "
                            f"File: {result['source_file']}")
                output.append(f"   {result['chunk_text'][:100]}...")
                output.append("")
            return "\n".join(output)
        
        else:  # detailed
            output = []
            output.append("=" * 80)
            output.append("SEARCH RESULTS")
            output.append("=" * 80)
            
            for i, result in enumerate(results, 1):
                output.append(f"\n📄 RESULT {i}")
                output.append("-" * 40)
                output.append(f"Similarity Score: {result['similarity_score']:.4f}")
                output.append(f"Source File: {result['source_file']}")
                output.append(f"Table: {result['table_name']}")
                output.append(f"Page Number: {result['page_number']}")
                output.append(f"Chunk Index: {result['chunk_index']}")
                output.append(f"Chunk Type: {result['chunk_type']}")
                output.append(f"Token Count: {result['chunk_tokens']}")
                output.append(f"Embedding Model: {result['embedding_model']}")
                output.append(f"Created At: {result['created_at']}")
                
                # Metadata
                if result['metadata']:
                    output.append(f"Metadata:")
                    for key, value in result['metadata'].items():
                        output.append(f"  - {key}: {value}")
                
                # Text content
                output.append(f"\nContent:")
                output.append(f"{result['chunk_text']}")
                
                if i < len(results):
                    output.append("\n" + "=" * 80)
            
            return "\n".join(output)


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Search vector tables with a query and return chunks with metadata',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python query_search.py "finansal performans"
  python query_search.py "risk yönetimi" --limit 3 --threshold 0.6
  python query_search.py "annual report" --table vectors_doc_2024_yonetim_kurulu_faaliyet_raporu_sasa
  python query_search.py "yönetim kurulu" --format json --output results.json
        """
    )
    
    parser.add_argument('query', help='Search query')
    parser.add_argument('--limit', '-l', type=int, default=5, 
                       help='Number of results to return (default: 5)')
    parser.add_argument('--threshold', '-t', type=float, default=0.5,
                       help='Similarity threshold (0.0-1.0, default: 0.5)')
    parser.add_argument('--table', help='Search specific table only')
    parser.add_argument('--format', '-f', choices=['detailed', 'brief', 'json'], 
                       default='detailed', help='Output format (default: detailed)')
    parser.add_argument('--output', '-o', help='Output file (default: stdout)')
    
    args = parser.parse_args()
    
    # Initialize searcher
    try:
        searcher = QuerySearcher()
    except Exception as e:
        print(f"❌ Failed to initialize searcher: {e}")
        sys.exit(1)
    
    # Perform search
    results = searcher.search_all_tables(
        query=args.query,
        limit=args.limit,
        similarity_threshold=args.threshold,
        specific_table=args.table
    )
    
    # Format and output results
    formatted_results = searcher.format_results(results, args.format)
    
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(formatted_results)
            print(f"📝 Results saved to: {args.output}")
        except Exception as e:
            print(f"❌ Error saving to file: {e}")
            print(formatted_results)
    else:
        print(formatted_results)


if __name__ == "__main__":
    main()
