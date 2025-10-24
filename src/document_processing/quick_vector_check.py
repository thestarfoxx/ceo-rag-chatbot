#!/usr/bin/env python3
"""
Quick check of your specific vector tables
"""

import psycopg2
from sqlalchemy import create_engine, text
import os
import openai

def quick_check():
    """Quick check of vector tables."""
    
    # Database connection
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
    
    # Your specific tables
    tables = [
        "vectors_doc_2022_yonetim_kurulu_faaliyet_raporu_sasa",
        "vectors_doc_2023_yonetim_kurulu_faaliyet_raporu_sasa", 
        "vectors_doc_2024_yonetim_kurulu_faaliyet_raporu_sasa"
    ]
    
    print("🔍 Quick Vector Table Check")
    print("=" * 50)
    
    try:
        with engine.connect() as conn:
            for table_name in tables:
                print(f"\n📋 Checking: {table_name}")
                
                try:
                    # Basic stats
                    stats = conn.execute(text(f"""
                        SELECT 
                            COUNT(*) as total_chunks,
                            COUNT(embedding) as with_embeddings,
                            MIN(chunk_tokens) as min_tokens,
                            MAX(chunk_tokens) as max_tokens
                        FROM "{table_name}"
                    """)).fetchone()
                    
                    print(f"   Total chunks: {stats.total_chunks}")
                    print(f"   With embeddings: {stats.with_embeddings}")
                    print(f"   Token range: {stats.min_tokens}-{stats.max_tokens}")
                    
                    # Sample content
                    sample = conn.execute(text(f"""
                        SELECT chunk_text, page_number, embedding IS NOT NULL as has_emb
                        FROM "{table_name}"
                        ORDER BY chunk_index
                        LIMIT 2
                    """)).fetchall()
                    
                    for i, chunk in enumerate(sample):
                        emb_status = "✅" if chunk.has_emb else "❌"
                        print(f"   Sample {i+1}: Page {chunk.page_number} {emb_status}")
                        print(f"      {chunk.chunk_text[:80]}...")
                    
                    # If has embeddings, test a simple search
                    if stats.with_embeddings > 0:
                        print(f"   🧪 Testing vector search capability...")
                        
                        # Test cosine similarity between first two embeddings
                        similarity_test = conn.execute(text(f"""
                            SELECT 
                                a.chunk_index as chunk_a,
                                b.chunk_index as chunk_b,
                                1 - (a.embedding <=> b.embedding) as similarity
                            FROM "{table_name}" a, "{table_name}" b
                            WHERE a.embedding IS NOT NULL 
                            AND b.embedding IS NOT NULL
                            AND a.chunk_index != b.chunk_index
                            LIMIT 1
                        """)).fetchone()
                        
                        if similarity_test:
                            print(f"   ✅ Vector operations working! Similarity between chunks: {similarity_test.similarity:.3f}")
                        else:
                            print(f"   ⚠️  Could not test similarity (need at least 2 embeddings)")
                    
                except Exception as e:
                    print(f"   ❌ Error: {e}")
                    continue
    
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return
    
    print(f"\n" + "=" * 50)
    
    # Test OpenAI connection
    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key:
        print("🧪 Testing OpenAI connection...")
        try:
            openai.api_key = openai_key
            response = openai.embeddings.create(
                model="text-embedding-3-small",
                input="test query"
            )
            print("✅ OpenAI API working!")
            print(f"   Embedding dimensions: {len(response.data[0].embedding)}")
        except Exception as e:
            print(f"❌ OpenAI API error: {e}")
    else:
        print("❌ No OpenAI API key found")
    
    print(f"\n💡 NEXT STEPS:")
    print("1. If tables have embeddings, try lowering similarity threshold to 0.5")
    print("2. If no embeddings, re-run PDF processing with generate_embeddings=True")
    print("3. Test with simple queries like 'finansal' or 'yönetim'")

if __name__ == "__main__":
    quick_check()
