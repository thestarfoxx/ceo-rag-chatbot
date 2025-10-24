#!/usr/bin/env python3
"""
Vector Tables Inspector Script
Comprehensive inspection tool for separate PDF vector tables system
"""

import psycopg2
from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine, text
import pandas as pd
import json
from typing import Dict, Any, List, Optional
import os
from datetime import datetime
import argparse
import sys

class VectorTablesInspector:
    """Automated inspector for separate vector tables system."""
    
    def __init__(self, db_config: Dict[str, str] = None):
        """Initialize inspector with database configuration."""
        self.db_config = db_config or self._get_default_db_config()
        self.engine = self._create_engine()
    
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
        """Create SQLAlchemy engine."""
        password_part = f":{self.db_config['password']}" if self.db_config['password'] else "password"
        connection_string = (
            f"postgresql://{self.db_config['username']}{password_part}@"
            f"{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
        )
        return create_engine(connection_string, echo=False)
    
    def check_connection(self) -> bool:
        """Test database connection."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            print("✅ Database connection successful")
            return True
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            return False
    
    def system_overview(self) -> Dict[str, Any]:
        """Get system overview statistics."""
        try:
            with self.engine.connect() as conn:
                # Basic document counts
                doc_stats = conn.execute(text("""
                    SELECT 
                        COUNT(*) as total_documents,
                        COUNT(CASE WHEN processing_status = 'completed' THEN 1 END) as completed_documents,
                        COUNT(CASE WHEN processing_status = 'failed' THEN 1 END) as failed_documents,
                        COUNT(CASE WHEN processing_status = 'processing' THEN 1 END) as processing_documents,
                        COUNT(CASE WHEN vector_table_created = TRUE THEN 1 END) as vector_tables_created,
                        SUM(total_chunks) as total_chunks,
                        SUM(total_tokens) as total_tokens
                    FROM pdf_documents
                """)).fetchone()
                
                # Vector table counts from information_schema
                vector_table_count = conn.execute(text("""
                    SELECT COUNT(*) as actual_vector_tables
                    FROM information_schema.tables 
                    WHERE table_name LIKE 'vectors_%'
                """)).fetchone()
                
                return {
                    'total_documents': doc_stats.total_documents,
                    'completed_documents': doc_stats.completed_documents,
                    'failed_documents': doc_stats.failed_documents,
                    'processing_documents': doc_stats.processing_documents,
                    'vector_tables_created': doc_stats.vector_tables_created,
                    'actual_vector_tables': vector_table_count.actual_vector_tables,
                    'total_chunks': doc_stats.total_chunks or 0,
                    'total_tokens': doc_stats.total_tokens or 0
                }
        except Exception as e:
            print(f"❌ Error getting system overview: {e}")
            return {}
    
    def list_all_documents(self) -> List[Dict[str, Any]]:
        """List all documents with their status."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT 
                        id,
                        file_name,
                        processing_status,
                        vector_table_name,
                        vector_table_created,
                        total_chunks,
                        total_tokens,
                        extraction_method,
                        embedding_model,
                        processed_at,
                        error_message
                    FROM pdf_documents
                    ORDER BY processed_at DESC
                """))
                
                return [dict(row._mapping) for row in result]
        except Exception as e:
            print(f"❌ Error listing documents: {e}")
            return []
    
    def discover_vector_tables(self) -> List[Dict[str, Any]]:
        """Discover all vector tables and their properties."""
        try:
            with self.engine.connect() as conn:
                # Get all vector tables from information_schema
                result = conn.execute(text("""
                    SELECT 
                        t.table_name,
                        pg_size_pretty(pg_total_relation_size('public.' || t.table_name)) as total_size,
                        pg_size_pretty(pg_relation_size('public.' || t.table_name)) as table_size,
                        pg_size_pretty(pg_indexes_size('public.' || t.table_name)) as index_size
                    FROM information_schema.tables t
                    WHERE t.table_name LIKE 'vectors_%'
                    ORDER BY pg_total_relation_size('public.' || t.table_name) DESC
                """))
                
                vector_tables = []
                for row in result:
                    # Get corresponding document info
                    doc_info = conn.execute(text("""
                        SELECT file_name, id, processing_status
                        FROM pdf_documents 
                        WHERE vector_table_name = :table_name
                    """), {"table_name": row.table_name}).fetchone()
                    
                    table_info = {
                        'table_name': row.table_name,
                        'total_size': row.total_size,
                        'table_size': row.table_size,
                        'index_size': row.index_size,
                        'file_name': doc_info.file_name if doc_info else 'ORPHANED',
                        'document_id': doc_info.id if doc_info else None,
                        'processing_status': doc_info.processing_status if doc_info else 'UNKNOWN'
                    }
                    vector_tables.append(table_info)
                
                return vector_tables
        except Exception as e:
            print(f"❌ Error discovering vector tables: {e}")
            return []
    
    def inspect_vector_table(self, table_name: str) -> Dict[str, Any]:
        """Inspect a specific vector table in detail."""
        try:
            with self.engine.connect() as conn:
                # Check if table exists
                table_exists = conn.execute(text("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = :table_name
                    )
                """), {"table_name": table_name}).scalar()
                
                if not table_exists:
                    return {'error': f'Table {table_name} does not exist'}
                
                # Get basic statistics
                stats = conn.execute(text(f"""
                    SELECT 
                        COUNT(*) as total_chunks,
                        COUNT(embedding) as chunks_with_embeddings,
                        AVG(chunk_tokens) as avg_tokens,
                        MIN(chunk_tokens) as min_tokens,
                        MAX(chunk_tokens) as max_tokens,
                        MIN(page_number) as first_page,
                        MAX(page_number) as last_page,
                        COUNT(DISTINCT page_number) as pages_covered,
                        COUNT(DISTINCT chunk_type) as chunk_types,
                        MIN(created_at) as first_chunk_created,
                        MAX(created_at) as last_chunk_created
                    FROM "{table_name}"
                """)).fetchone()
                
                # Get embedding dimensions
                embedding_info = conn.execute(text(f"""
                    SELECT array_length(embedding, 1) as embedding_dimensions
                    FROM "{table_name}" 
                    WHERE embedding IS NOT NULL 
                    LIMIT 1
                """)).fetchone()
                
                # Get chunk type breakdown
                chunk_types = conn.execute(text(f"""
                    SELECT chunk_type, COUNT(*) as count
                    FROM "{table_name}"
                    GROUP BY chunk_type
                    ORDER BY count DESC
                """)).fetchall()
                
                # Get sample chunks
                samples = conn.execute(text(f"""
                    SELECT 
                        chunk_index,
                        page_number,
                        chunk_tokens,
                        embedding IS NOT NULL as has_embedding,
                        LEFT(chunk_text, 100) as text_preview
                    FROM "{table_name}"
                    ORDER BY chunk_index
                    LIMIT 3
                """)).fetchall()
                
                return {
                    'table_name': table_name,
                    'total_chunks': stats.total_chunks,
                    'chunks_with_embeddings': stats.chunks_with_embeddings,
                    'embedding_coverage': round((stats.chunks_with_embeddings / stats.total_chunks * 100), 2) if stats.total_chunks > 0 else 0,
                    'avg_tokens': round(stats.avg_tokens, 2) if stats.avg_tokens else 0,
                    'min_tokens': stats.min_tokens,
                    'max_tokens': stats.max_tokens,
                    'page_range': f"{stats.first_page}-{stats.last_page}" if stats.first_page else "N/A",
                    'pages_covered': stats.pages_covered,
                    'chunk_types': len(chunk_types),
                    'embedding_dimensions': embedding_info.embedding_dimensions if embedding_info else None,
                    'first_chunk_created': stats.first_chunk_created,
                    'last_chunk_created': stats.last_chunk_created,
                    'chunk_type_breakdown': [dict(row._mapping) for row in chunk_types],
                    'sample_chunks': [dict(row._mapping) for row in samples]
                }
                
        except Exception as e:
            return {'error': f'Error inspecting table {table_name}: {e}'}
    
    def find_orphaned_tables(self) -> List[str]:
        """Find vector tables without corresponding document records."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT t.table_name
                    FROM information_schema.tables t
                    WHERE t.table_name LIKE 'vectors_%'
                    AND t.table_name NOT IN (
                        SELECT vector_table_name 
                        FROM pdf_documents 
                        WHERE vector_table_name IS NOT NULL
                    )
                """))
                
                return [row.table_name for row in result]
        except Exception as e:
            print(f"❌ Error finding orphaned tables: {e}")
            return []
    
    def find_missing_tables(self) -> List[Dict[str, Any]]:
        """Find documents with missing vector tables."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT 
                        d.id,
                        d.file_name,
                        d.vector_table_name,
                        d.vector_table_created
                    FROM pdf_documents d
                    WHERE d.vector_table_created = TRUE
                    AND d.vector_table_name NOT IN (
                        SELECT table_name 
                        FROM information_schema.tables 
                        WHERE table_name LIKE 'vectors_%'
                    )
                """))
                
                return [dict(row._mapping) for row in result]
        except Exception as e:
            print(f"❌ Error finding missing tables: {e}")
            return []
    
    def get_processing_errors(self) -> List[Dict[str, Any]]:
        """Get documents with processing errors."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT 
                        id,
                        file_name,
                        processing_status,
                        error_message,
                        processed_at,
                        vector_table_name
                    FROM pdf_documents
                    WHERE processing_status = 'failed'
                       OR error_message IS NOT NULL
                    ORDER BY processed_at DESC
                """))
                
                return [dict(row._mapping) for row in result]
        except Exception as e:
            print(f"❌ Error getting processing errors: {e}")
            return []
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get embedding statistics across all vector tables."""
        try:
            vector_tables = self.discover_vector_tables()
            total_chunks = 0
            total_embeddings = 0
            embedding_models = set()
            
            with self.engine.connect() as conn:
                for table_info in vector_tables:
                    table_name = table_info['table_name']
                    try:
                        stats = conn.execute(text(f"""
                            SELECT 
                                COUNT(*) as chunks,
                                COUNT(embedding) as embeddings,
                                embedding_model
                            FROM "{table_name}"
                            WHERE embedding_model IS NOT NULL
                            GROUP BY embedding_model
                        """)).fetchall()
                        
                        for stat in stats:
                            total_chunks += stat.chunks
                            total_embeddings += stat.embeddings
                            if stat.embedding_model:
                                embedding_models.add(stat.embedding_model)
                                
                    except Exception as e:
                        print(f"⚠️  Error checking table {table_name}: {e}")
                        continue
            
            return {
                'total_chunks_across_tables': total_chunks,
                'total_embeddings_across_tables': total_embeddings,
                'embedding_coverage': round((total_embeddings / total_chunks * 100), 2) if total_chunks > 0 else 0,
                'embedding_models_used': list(embedding_models)
            }
        except Exception as e:
            print(f"❌ Error getting embedding stats: {e}")
            return {}
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """Generate comprehensive inspection report."""
        report_lines = []
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report_lines.append("=" * 60)
        report_lines.append("VECTOR TABLES INSPECTION REPORT")
        report_lines.append(f"Generated: {timestamp}")
        report_lines.append("=" * 60)
        
        # System Overview
        report_lines.append("\n1. SYSTEM OVERVIEW")
        report_lines.append("-" * 30)
        overview = self.system_overview()
        if overview:
            for key, value in overview.items():
                report_lines.append(f"{key.replace('_', ' ').title()}: {value}")
        
        # Documents List
        report_lines.append("\n2. DOCUMENTS STATUS")
        report_lines.append("-" * 30)
        documents = self.list_all_documents()
        if documents:
            report_lines.append(f"{'ID':<4} {'File Name':<30} {'Status':<12} {'Table Created':<13} {'Chunks':<8} {'Tokens':<8}")
            report_lines.append("-" * 80)
            for doc in documents:
                status_icon = "✅" if doc['processing_status'] == 'completed' else ("❌" if doc['processing_status'] == 'failed' else "⏳")
                table_icon = "✅" if doc['vector_table_created'] else "❌"
                report_lines.append(f"{doc['id']:<4} {doc['file_name'][:29]:<30} {status_icon} {doc['processing_status']:<10} {table_icon} {doc['vector_table_created']:<10} {doc['total_chunks'] or 0:<8} {doc['total_tokens'] or 0:<8}")
        
        # Vector Tables Discovery
        report_lines.append("\n3. VECTOR TABLES")
        report_lines.append("-" * 30)
        vector_tables = self.discover_vector_tables()
        if vector_tables:
            report_lines.append(f"{'Table Name':<35} {'File Name':<25} {'Total Size':<12} {'Status':<10}")
            report_lines.append("-" * 85)
            for table in vector_tables:
                status_icon = "✅" if table['processing_status'] == 'completed' else ("❌" if table['file_name'] == 'ORPHANED' else "⚠️")
                report_lines.append(f"{table['table_name']:<35} {table['file_name'][:24]:<25} {table['total_size']:<12} {status_icon} {table['processing_status']:<8}")
        
        # Embedding Statistics
        report_lines.append("\n4. EMBEDDING STATISTICS")
        report_lines.append("-" * 30)
        embedding_stats = self.get_embedding_stats()
        if embedding_stats:
            for key, value in embedding_stats.items():
                report_lines.append(f"{key.replace('_', ' ').title()}: {value}")
        
        # Issues and Warnings
        report_lines.append("\n5. ISSUES AND WARNINGS")
        report_lines.append("-" * 30)
        
        orphaned = self.find_orphaned_tables()
        if orphaned:
            report_lines.append("⚠️  Orphaned Vector Tables (no document record):")
            for table in orphaned:
                report_lines.append(f"   - {table}")
        
        missing = self.find_missing_tables()
        if missing:
            report_lines.append("⚠️  Missing Vector Tables (document record exists):")
            for doc in missing:
                report_lines.append(f"   - {doc['file_name']} -> {doc['vector_table_name']}")
        
        errors = self.get_processing_errors()
        if errors:
            report_lines.append("❌ Processing Errors:")
            for error in errors:
                report_lines.append(f"   - {error['file_name']}: {error['error_message']}")
        
        if not orphaned and not missing and not errors:
            report_lines.append("✅ No issues found!")
        
        # Table Details (for top 3 largest tables)
        if vector_tables:
            report_lines.append("\n6. DETAILED TABLE INSPECTION (Top 3)")
            report_lines.append("-" * 30)
            for table in vector_tables[:3]:
                table_name = table['table_name']
                details = self.inspect_vector_table(table_name)
                if 'error' not in details:
                    report_lines.append(f"\nTable: {table_name}")
                    report_lines.append(f"  File: {table['file_name']}")
                    report_lines.append(f"  Total Chunks: {details['total_chunks']}")
                    report_lines.append(f"  Chunks with Embeddings: {details['chunks_with_embeddings']}")
                    report_lines.append(f"  Embedding Coverage: {details['embedding_coverage']}%")
                    report_lines.append(f"  Avg Tokens per Chunk: {details['avg_tokens']}")
                    report_lines.append(f"  Page Range: {details['page_range']}")
                    report_lines.append(f"  Embedding Dimensions: {details['embedding_dimensions']}")
        
        report_lines.append("\n" + "=" * 60)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 60)
        
        report_text = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"📝 Report saved to: {output_file}")
        
        return report_text
    
    def interactive_mode(self):
        """Run inspector in interactive mode."""
        print("\n🔍 Vector Tables Inspector - Interactive Mode")
        print("=" * 50)
        
        while True:
            print("\nOptions:")
            print("1. System Overview")
            print("2. List All Documents")
            print("3. List Vector Tables")
            print("4. Inspect Specific Table")
            print("5. Find Issues")
            print("6. Generate Full Report")
            print("7. Exit")
            
            choice = input("\nEnter your choice (1-7): ").strip()
            
            if choice == '1':
                print("\n📊 System Overview:")
                overview = self.system_overview()
                for key, value in overview.items():
                    print(f"  {key.replace('_', ' ').title()}: {value}")
            
            elif choice == '2':
                print("\n📄 Documents List:")
                documents = self.list_all_documents()
                for doc in documents:
                    status = "✅" if doc['processing_status'] == 'completed' else ("❌" if doc['processing_status'] == 'failed' else "⏳")
                    print(f"  {status} {doc['file_name']} (ID: {doc['id']}) - {doc['processing_status']}")
            
            elif choice == '3':
                print("\n🗂️  Vector Tables:")
                tables = self.discover_vector_tables()
                for table in tables:
                    status = "✅" if table['processing_status'] == 'completed' else "❌"
                    print(f"  {status} {table['table_name']} -> {table['file_name']} ({table['total_size']})")
            
            elif choice == '4':
                tables = self.discover_vector_tables()
                if not tables:
                    print("No vector tables found!")
                    continue
                
                print("\nAvailable tables:")
                for i, table in enumerate(tables):
                    print(f"  {i+1}. {table['table_name']} -> {table['file_name']}")
                
                try:
                    table_idx = int(input("Enter table number to inspect: ")) - 1
                    if 0 <= table_idx < len(tables):
                        table_name = tables[table_idx]['table_name']
                        details = self.inspect_vector_table(table_name)
                        print(f"\n🔍 Inspecting: {table_name}")
                        for key, value in details.items():
                            if key not in ['chunk_type_breakdown', 'sample_chunks']:
                                print(f"  {key.replace('_', ' ').title()}: {value}")
                    else:
                        print("Invalid table number!")
                except ValueError:
                    print("Please enter a valid number!")
            
            elif choice == '5':
                print("\n⚠️  Issues Check:")
                orphaned = self.find_orphaned_tables()
                missing = self.find_missing_tables()
                errors = self.get_processing_errors()
                
                if orphaned:
                    print(f"  Orphaned tables: {len(orphaned)}")
                    for table in orphaned:
                        print(f"    - {table}")
                
                if missing:
                    print(f"  Missing tables: {len(missing)}")
                    for doc in missing:
                        print(f"    - {doc['file_name']}")
                
                if errors:
                    print(f"  Processing errors: {len(errors)}")
                    for error in errors:
                        print(f"    - {error['file_name']}: {error['error_message']}")
                
                if not orphaned and not missing and not errors:
                    print("  ✅ No issues found!")
            
            elif choice == '6':
                output_file = input("Enter output file name (or press Enter for console only): ").strip()
                if not output_file:
                    output_file = None
                
                report = self.generate_report(output_file)
                if not output_file:
                    print("\n" + report)
            
            elif choice == '7':
                print("👋 Goodbye!")
                break
            
            else:
                print("Invalid choice! Please enter 1-7.")


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description='Vector Tables Inspector')
    parser.add_argument('--host', default='localhost', help='Database host')
    parser.add_argument('--port', default='5432', help='Database port')
    parser.add_argument('--database', default='ceo_rag_db', help='Database name')
    parser.add_argument('--username', default='postgres', help='Database username')
    parser.add_argument('--password', default='', help='Database password')
    parser.add_argument('--output', '-o', help='Output file for report')
    parser.add_argument('--interactive', '-i', action='store_true', help='Run in interactive mode')
    parser.add_argument('--table', '-t', help='Inspect specific table')
    
    args = parser.parse_args()
    
    # Set up database configuration
    db_config = {
        'host': args.host,
        'port': args.port,
        'database': args.database,
        'username': args.username,
        'password': args.password
    }
    
    # Initialize inspector
    inspector = VectorTablesInspector(db_config)
    
    # Check connection
    if not inspector.check_connection():
        sys.exit(1)
    
    # Run based on arguments
    if args.interactive:
        inspector.interactive_mode()
    elif args.table:
        details = inspector.inspect_vector_table(args.table)
        print(f"\n🔍 Inspecting table: {args.table}")
        print("=" * 50)
        for key, value in details.items():
            if key not in ['chunk_type_breakdown', 'sample_chunks']:
                print(f"{key.replace('_', ' ').title()}: {value}")
    else:
        # Generate full report
        report = inspector.generate_report(args.output)
        if not args.output:
            print(report)


if __name__ == "__main__":
    main()
