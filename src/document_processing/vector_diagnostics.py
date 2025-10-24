#!/usr/bin/env python3
"""
Vector Table Diagnostics
Diagnose why vector search isn't returning results
"""

import psycopg2
from sqlalchemy import create_engine, text
import os
import openai
import json

class VectorTableDiagnostics:
    def __init__(self):
        """Initialize diagnostics."""
        self.db_config = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': os.getenv('POSTGRES_PORT', '5432'),
            'database': os.getenv('POSTGRES_DB', 'ceo_rag_db'),
            'username': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', 'password')
        }
        
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
        
        password_part = f":{self.db_config['password']}" if self.db_config['password'] else ""
        connection_string = (
            f"postgresql://{self.db_config['username']}{password_part}@"
            f"{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
        )
        self.engine = create_engine(connection_string, echo=False)
    
    def check_basic_connection(self):
        """Check basic database connection."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT version()")).scalar()
                print(f"✅ Connected to PostgreSQL: {result}")
                return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def check_pgvector_extension(self):
        """Check if pgvector extension is installed."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT extname, extversion 
                    FROM pg_extension 
                    WHERE extname = 'vector'
                """)).fetchone()
                
                if result:
                    print(f"✅ pgvector extension installed: version {result.extversion}")
                    return True
                else:
                    print("❌ pgvector extension not installed!")
                    print("   Run: CREATE EXTENSION vector;")
                    return False
        except Exception as e:
            print(f"❌ Error checking pgvector: {e}")
            return False
    
    def check_documents_table(self):
        """Check pdf_documents table."""
        try:
            with self.engine.connect() as conn:
                # Check if table exists
                exists = conn.execute(text("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'pdf_documents'
                    )
                """)).scalar()
                
                if not exists:
                    print("❌ pdf_documents table does not exist!")
                    return False
                
                # Get basic stats
                stats = conn.execute(text("""
                    SELECT 
                        COUNT(*) as total_docs,
                        COUNT(CASE WHEN processing_status = 'completed' THEN 1 END) as completed,
                        COUNT(CASE WHEN vector_table_created = TRUE THEN 1 END) as with_vector_tables
                    FROM pdf_documents
                """)).fetchone()
                
                print(f"📊 Documents table:")
                print(f"   Total documents: {stats.total_docs}")
                print(f"   Completed: {stats.completed}")
                print(f"   With vector tables: {stats.with_vector_tables}")
                
                # List documents
                docs = conn.execute(text("""
                    SELECT file_name, processing_status, vector_table_name, vector_table_created
                    FROM pdf_documents
                    ORDER BY processed_at DESC
                """)).fetchall()
                
                print(f"\n📄 Documents list:")
                for doc in docs:
                    status_icon = "✅" if doc.processing_status == 'completed' else "❌"
                    table_icon = "✅" if doc.vector_table_created else "❌"
                    print(f"   {status_icon} {doc.file_name} | Table: {table_icon} {doc.vector_table_name}")
                
                return len(docs) > 0
                
        except Exception as e:
            print(f"❌ Error checking documents table: {e}")
            return False
    
    def discover_vector_tables(self):
        """Discover actual vector tables in database."""
        try:
            with self.engine.connect() as conn:
                # Find all tables that start with 'vectors_'
                actual_tables = conn.execute(text("""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_name LIKE 'vectors_%'
                    ORDER BY table_name
                """)).fetchall()
                
                print(f"\n🗂️  Actual vector tables found: {len(actual_tables)}")
                
                vector_tables = []
                for table in actual_tables:
                    table_name = table.table_name
                    
                    # Get table info with proper error handling
                    try:
                        table_info = conn.execute(text(f"""
                            SELECT 
                                COUNT(*) as total_chunks,
                                COUNT(embedding) as chunks_with_embeddings,
                                COUNT(DISTINCT page_number) as pages,
                                MIN(chunk_tokens) as min_tokens,
                                MAX(chunk_tokens) as max_tokens,
                                AVG(chunk_tokens) as avg_tokens
                            FROM "{table_name}"
                        """)).fetchone()
                        
                        # Check embedding dimensions using a different approach
                        embedding_dim = None
                        embedding_sample = None
                        if table_info.chunks_with_embeddings > 0:
                            try:
                                # Get a sample embedding to check dimensions
                                embedding_result = conn.execute(text(f"""
                                    SELECT embedding
                                    FROM "{table_name}" 
                                    WHERE embedding IS NOT NULL 
                                    LIMIT 1
                                """)).fetchone()
                                
                                if embedding_result and embedding_result.embedding:
                                    embedding_sample = embedding_result.embedding
                                    embedding_dim = len(embedding_sample)
                                    
                            except Exception as e:
                                print(f"   ⚠️  Could not determine embedding dimensions for {table_name}: {e}")
                        
                        # Get sample text
                        sample = conn.execute(text(f"""
                            SELECT chunk_text, page_number, chunk_tokens
                            FROM "{table_name}"
                            ORDER BY chunk_index
                            LIMIT 1
                        """)).fetchone()
                        
                        table_data = {
                            'name': table_name,
                            'total_chunks': table_info.total_chunks,
                            'chunks_with_embeddings': table_info.chunks_with_embeddings,
                            'embedding_coverage': round((table_info.chunks_with_embeddings / table_info.total_chunks * 100), 2) if table_info.total_chunks > 0 else 0,
                            'pages': table_info.pages,
                            'token_range': f"{table_info.min_tokens}-{table_info.max_tokens}" if table_info.min_tokens else "N/A",
                            'avg_tokens': round(table_info.avg_tokens, 1) if table_info.avg_tokens else 0,
                            'embedding_dimensions': embedding_dim,
                            'sample_text': sample.chunk_text[:100] + "..." if sample and sample.chunk_text else "No text",
                            'sample_page': sample.page_number if sample else "N/A",
                            'embedding_sample': embedding_sample
                        }
                        
                        vector_tables.append(table_data)
                        
                        print(f"\n   📋 {table_name}:")
                        print(f"      Total chunks: {table_data['total_chunks']}")
                        print(f"      With embeddings: {table_data['chunks_with_embeddings']} ({table_data['embedding_coverage']}%)")
                        print(f"      Pages covered: {table_data['pages']}")
                        print(f"      Token range: {table_data['token_range']}")
                        print(f"      Embedding dimensions: {table_data['embedding_dimensions']}")
                        print(f"      Sample text: {table_data['sample_text']}")
                        
                    except Exception as e:
                        print(f"   ❌ Error analyzing {table_name}: {e}")
                        # Try to rollback the transaction and continue
                        try:
                            conn.rollback()
                        except:
                            pass
                        continue
                
                return vector_tables
                
        except Exception as e:
            print(f"❌ Error discovering vector tables: {e}")
            return []
    
    def test_embedding_generation(self):
        """Test if we can generate embeddings."""
        if not self.openai_api_key:
            print("❌ No OpenAI API key found!")
            print("   Set OPENAI_API_KEY environment variable")
            return False
        
        try:
            print(f"\n🧪 Testing embedding generation...")
            
            test_text = "finansal performans"
            response = openai.embeddings.create(
                model="text-embedding-3-small",
                input=test_text
            )
            
            embedding = response.data[0].embedding
            print(f"✅ Successfully generated embedding for '{test_text}'")
            print(f"   Embedding dimensions: {len(embedding)}")
            print(f"   First 5 values: {embedding[:5]}")
            
            return embedding
            
        except Exception as e:
            print(f"❌ Error generating embedding: {e}")
            return None
    
    def test_vector_search_manually(self, table_name, test_embedding):
        """Test vector search with a known embedding."""
        try:
            with self.engine.connect() as conn:
                print(f"\n🔍 Testing vector search on {table_name}...")
                
                # First check if table has any embeddings
                embedding_count = conn.execute(text(f"""
                    SELECT COUNT(*) FROM "{table_name}" WHERE embedding IS NOT NULL
                """)).scalar()
                
                if embedding_count == 0:
                    print(f"❌ No embeddings found in {table_name}")
                    return False
                
                print(f"   Found {embedding_count} chunks with embeddings")
                
                # Test basic vector operations
                try:
                    # Test similarity calculation
                    similarity_test = conn.execute(text(f"""
                        SELECT 
                            chunk_index,
                            1 - (embedding <=> :test_embedding::vector) as similarity,
                            chunk_text
                        FROM "{table_name}"
                        WHERE embedding IS NOT NULL
                        ORDER BY embedding <=> :test_embedding::vector
                        LIMIT 3
                    """), {"test_embedding": test_embedding}).fetchall()
                    
                    if similarity_test:
                        print(f"✅ Vector search working! Top results:")
                        for i, result in enumerate(similarity_test):
                            print(f"      {i+1}. Similarity: {result.similarity:.4f} | Preview: {result.chunk_text[:80]}...")
                        return True
                    else:
                        print(f"❌ Vector search returned no results")
                        return False
                        
                except Exception as e:
                    print(f"❌ Vector search error: {e}")
                    return False
                    
        except Exception as e:
            print(f"❌ Error testing vector search: {e}")
            return False
    
    def test_similarity_thresholds(self, table_name, test_embedding):
        """Test different similarity thresholds to see what works."""
        try:
            with self.engine.connect() as conn:
                print(f"\n📊 Testing similarity thresholds on {table_name}...")
                
                thresholds = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
                
                for threshold in thresholds:
                    count = conn.execute(text(f"""
                        SELECT COUNT(*)
                        FROM "{table_name}"
                        WHERE embedding IS NOT NULL
                        AND 1 - (embedding <=> :test_embedding::vector) >= :threshold
                    """), {"test_embedding": test_embedding, "threshold": threshold}).scalar()
                    
                    print(f"   Threshold {threshold}: {count} results")
                    
                    if count > 0 and count <= 10:
                        # Show sample results for reasonable thresholds
                        samples = conn.execute(text(f"""
                            SELECT 
                                1 - (embedding <=> :test_embedding::vector) as similarity,
                                LEFT(chunk_text, 60) as preview
                            FROM "{table_name}"
                            WHERE embedding IS NOT NULL
                            AND 1 - (embedding <=> :test_embedding::vector) >= :threshold
                            ORDER BY embedding <=> :test_embedding::vector
                            LIMIT 3
                        """), {"test_embedding": test_embedding, "threshold": threshold}).fetchall()
                        
                        for sample in samples:
                            print(f"      → {sample.similarity:.3f}: {sample.preview}...")
                
        except Exception as e:
            print(f"❌ Error testing thresholds: {e}")
    
    def check_chunk_content(self, table_name, limit=5):
        """Check actual chunk content to see what's stored."""
        try:
            with self.engine.connect() as conn:
                print(f"\n📖 Sample chunk content from {table_name}:")
                
                chunks = conn.execute(text(f"""
                    SELECT 
                        chunk_index,
                        page_number,
                        chunk_tokens,
                        embedding IS NOT NULL as has_embedding,
                        chunk_text
                    FROM "{table_name}"
                    ORDER BY chunk_index
                    LIMIT :limit
                """), {"limit": limit}).fetchall()
                
                for chunk in chunks:
                    print(f"\n   Chunk {chunk.chunk_index} (Page {chunk.page_number}):")
                    print(f"      Tokens: {chunk.chunk_tokens}")
                    print(f"      Has embedding: {'✅' if chunk.has_embedding else '❌'}")
                    print(f"      Content: {chunk.chunk_text[:200]}...")
                    
        except Exception as e:
            print(f"❌ Error checking chunk content: {e}")
    
    def run_full_diagnostics(self):
        """Run complete diagnostics."""
        print("🔍 VECTOR TABLE DIAGNOSTICS")
        print("=" * 50)
        
        # Basic checks
        if not self.check_basic_connection():
            return
        
        if not self.check_pgvector_extension():
            return
        
        if not self.check_documents_table():
            return
        
        # Discover vector tables
        vector_tables = self.discover_vector_tables()
        if not vector_tables:
            print("\n❌ No vector tables found!")
            print("   Make sure you've processed some PDFs first")
            return
        
        # Test embedding generation
        test_embedding = self.test_embedding_generation()
        if not test_embedding:
            return
        
        # Test each vector table
        for table_data in vector_tables:
            table_name = table_data['name']
            
            if table_data['chunks_with_embeddings'] > 0:
                print(f"\n" + "="*60)
                print(f"TESTING TABLE: {table_name}")
                print("="*60)
                
                # Check chunk content
                self.check_chunk_content(table_name, limit=3)
                
                # Test vector search
                if self.test_vector_search_manually(table_name, test_embedding):
                    # Test thresholds
                    self.test_similarity_thresholds(table_name, test_embedding)
                
            else:
                print(f"\n⚠️  Skipping {table_name} - no embeddings found")
        
        print(f"\n" + "="*50)
        print("DIAGNOSTICS COMPLETE")
        print("="*50)
        
        # Summary and recommendations
        tables_with_embeddings = [t for t in vector_tables if t['chunks_with_embeddings'] > 0]
        
        if tables_with_embeddings:
            print(f"\n✅ Found {len(tables_with_embeddings)} tables with embeddings")
            print("\n💡 RECOMMENDATIONS:")
            
            avg_coverage = sum(t['embedding_coverage'] for t in tables_with_embeddings) / len(tables_with_embeddings)
            if avg_coverage < 50:
                print("   - Low embedding coverage detected. Consider regenerating embeddings.")
            
            print("   - Try lower similarity thresholds (0.6 or 0.5) in your queries")
            print("   - Test with simple queries first: 'financial', 'performance', etc.")
            print("   - Make sure your OpenAI API key is working")
            
        else:
            print("\n❌ No tables with embeddings found!")
            print("\n💡 TO FIX:")
            print("   1. Make sure PDFs are processed with generate_embeddings=True")
            print("   2. Check your OpenAI API key is valid")
            print("   3. Re-run PDF processing to generate embeddings")


def main():
    """Run diagnostics."""
    diagnostics = VectorTableDiagnostics()
    diagnostics.run_full_diagnostics()


if __name__ == "__main__":
    main()
