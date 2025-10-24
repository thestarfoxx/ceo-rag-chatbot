#!/usr/bin/env python3
"""
Simple Query Executor for CEO RAG Excel Database
Execute queries across all tables and create useful views.
"""

import psycopg2
from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine, text
import pandas as pd
from typing import Dict, List, Any, Optional
import os
from tabulate import tabulate
import argparse
import json

class SimpleQueryExecutor:
    def __init__(self, db_config: Optional[Dict[str, str]] = None):
        """Initialize query executor."""
        self.db_config = db_config or self._get_default_db_config()
        self.engine = self._create_engine()
        
    def _get_default_db_config(self) -> Dict[str, str]:
        """Get default PostgreSQL configuration."""
        return {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': os.getenv('POSTGRES_PORT', '5432'),
            'database': os.getenv('POSTGRES_DB', 'ceo_rag_db'),
            'username': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', 'your_password')
        }
    
    def _create_engine(self):
        """Create SQLAlchemy engine."""
        password_part = f":{self.db_config['password']}" if self.db_config['password'] else ""
        connection_string = (
            f"postgresql://{self.db_config['username']}{password_part}@"
            f"{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
        )
        return create_engine(connection_string, echo=False)
    
    def get_all_tables(self) -> List[str]:
        """Get list of all tables in the database."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                    ORDER BY table_name
                """))
                return [row.table_name for row in result]
        except Exception as e:
            print(f"Error getting tables: {e}")
            return []
    
    def execute_query(self, query: str, params: dict = None) -> List[Dict[str, Any]]:
        """Execute a single query and return results."""
        try:
            with self.engine.connect() as conn:
                if params:
                    result = conn.execute(text(query), params)
                else:
                    result = conn.execute(text(query))
                return [dict(row._mapping) for row in result]
        except Exception as e:
            return [{'error': f"Query failed: {str(e)}"}]
    
    def create_views(self):
        """Create useful database views."""
        print("🔧 Creating Database Views...")
        
        # Get all tables with metadata first
        tables_with_metadata = []
        for table in self.get_all_tables():
            try:
                with self.engine.connect() as conn:
                    columns_check = conn.execute(text("""
                        SELECT column_name 
                        FROM information_schema.columns 
                        WHERE table_name = :table_name 
                        AND column_name IN ('file_name', 'sheet_name', 'created_at')
                    """), {"table_name": table})
                    
                    required_cols = [row.column_name for row in columns_check]
                    if len(required_cols) >= 3:
                        tables_with_metadata.append(f"""
                            SELECT '{table}' as table_name, file_name, sheet_name, created_at 
                            FROM "{table}"
                            WHERE file_name IS NOT NULL
                        """)
            except:
                continue
        
        if tables_with_metadata:
            union_query = " UNION ALL ".join(tables_with_metadata)
            
            # Create Excel files summary view
            try:
                with self.engine.connect() as conn:
                    excel_view_sql = f"""
                        CREATE OR REPLACE VIEW v_excel_files_summary AS
                        SELECT 
                            table_name,
                            file_name,
                            sheet_name,
                            COUNT(*) as total_rows,
                            MIN(created_at) as first_imported,
                            MAX(created_at) as last_imported
                        FROM (
                            {union_query}
                        ) all_data
                        GROUP BY table_name, file_name, sheet_name
                        ORDER BY last_imported DESC
                    """
                    conn.execute(text(excel_view_sql))
                    conn.commit()
                    print("✅ Created view: v_excel_files_summary")
            except Exception as e:
                print(f"❌ Error creating Excel files view: {e}")
            
            # Create recent imports view
            try:
                with self.engine.connect() as conn:
                    recent_view_sql = f"""
                        CREATE OR REPLACE VIEW v_recent_imports AS
                        SELECT 
                            table_name,
                            file_name,
                            COUNT(*) as records,
                            MAX(created_at) as import_time,
                            EXTRACT(EPOCH FROM (NOW() - MAX(created_at)))/3600 as hours_ago
                        FROM (
                            {union_query}
                        ) all_recent
                        WHERE created_at >= NOW() - INTERVAL '7 days'
                        GROUP BY table_name, file_name
                        ORDER BY import_time DESC
                    """
                    conn.execute(text(recent_view_sql))
                    conn.commit()
                    print("✅ Created view: v_recent_imports")
            except Exception as e:
                print(f"❌ Error creating recent imports view: {e}")
        
        # Create table stats view
        try:
            with self.engine.connect() as conn:
                table_stats_sql = """
                    CREATE OR REPLACE VIEW v_table_stats AS
                    SELECT 
                        schemaname,
                        relname as table_name,
                        n_live_tup as live_rows,
                        n_dead_tup as dead_rows,
                        last_vacuum,
                        last_analyze,
                        pg_size_pretty(pg_total_relation_size(schemaname||'.'||relname)) as table_size
                    FROM pg_stat_user_tables
                    ORDER BY n_live_tup DESC
                """
                conn.execute(text(table_stats_sql))
                conn.commit()
                print("✅ Created view: v_table_stats")
        except Exception as e:
            print(f"❌ Error creating table stats view: {e}")
    
    def run_predefined_queries(self):
        """Run a set of predefined useful queries."""
        queries = {
            "Database Size": "SELECT pg_size_pretty(pg_database_size(current_database())) as database_size",
            
            "All Tables Overview": """
                SELECT 
                    table_name,
                    (SELECT COUNT(*) FROM information_schema.columns WHERE table_name = t.table_name) as columns,
                    pg_size_pretty(pg_total_relation_size(('public.' || table_name)::regclass)) as size
                FROM information_schema.tables t
                WHERE table_schema = 'public'
                ORDER BY pg_total_relation_size(('public.' || table_name)::regclass) DESC
            """,
            
            "Excel Files Summary": """
                SELECT * FROM v_excel_files_summary
                ORDER BY last_imported DESC
                LIMIT 10
            """,
            
            "Recent Activity": """
                SELECT * FROM v_recent_imports
                WHERE hours_ago <= 168  -- Last week
                ORDER BY import_time DESC
            """,
            
            "Table Statistics": """
                SELECT * FROM v_table_stats
                ORDER BY live_rows DESC
            """,
            
            "Column Types Summary": """
                SELECT 
                    data_type,
                    COUNT(*) as column_count,
                    COUNT(DISTINCT table_name) as tables_using
                FROM information_schema.columns
                WHERE table_schema = 'public'
                GROUP BY data_type
                ORDER BY column_count DESC
            """
        }
        
        print("🔍 Running Predefined Queries")
        print("=" * 60)
        
        for query_name, query in queries.items():
            print(f"\n📊 {query_name}:")
            print("-" * 40)
            
            try:
                results = self.execute_query(query)
                if results and 'error' not in results[0]:
                    if len(results) > 0:
                        df = pd.DataFrame(results)
                        print(df.to_string(index=False, max_rows=20))
                        if len(results) > 20:
                            print(f"... and {len(results) - 20} more rows")
                    else:
                        print("No results found")
                else:
                    print(f"❌ {results[0].get('error', 'Unknown error')}")
            except Exception as e:
                print(f"❌ Error: {e}")
        
        # Show all tables with sample data
        self.show_all_tables_with_samples()
    
    def show_all_tables_with_samples(self):
        """Show all table names with first 5 records from each."""
        print(f"\n" + "=" * 80)
        print("📋 ALL TABLES WITH SAMPLE DATA")
        print("=" * 80)
        
        tables = self.get_all_tables()
        
        if not tables:
            print("❌ No tables found in database")
            return
        
        for i, table in enumerate(tables, 1):
            print(f"\n📊 Table {i}/{len(tables)}: '{table}'")
            print("-" * 60)
            
            try:
                # Get table info
                with self.engine.connect() as conn:
                    count_result = conn.execute(text(f'SELECT COUNT(*) as count FROM "{table}"'))
                    row_count = count_result.scalar()
                    
                    col_result = conn.execute(text("""
                        SELECT COUNT(*) as columns 
                        FROM information_schema.columns 
                        WHERE table_name = :table_name
                    """), {"table_name": table})
                    col_count = col_result.scalar()
                
                print(f"📈 Rows: {row_count:,} | Columns: {col_count}")
                
                if row_count == 0:
                    print("   (Empty table)")
                    continue
                
                # Get sample data
                sample_results = self.execute_query(f'SELECT * FROM "{table}" LIMIT 5')
                
                if sample_results and 'error' not in sample_results[0]:
                    df = pd.DataFrame(sample_results)
                    
                    print(f"📝 Columns: {', '.join(df.columns.tolist())}")
                    print()
                    
                    # Show data with column limit for readability
                    if len(df.columns) > 8:
                        display_df = df.iloc[:, :8].copy()
                        print("📋 Sample Data (first 8 columns):")
                        print(display_df.to_string(index=False, max_colwidth=20))
                        print(f"   ... and {len(df.columns) - 8} more columns")
                    else:
                        print("📋 Sample Data:")
                        print(df.to_string(index=False, max_colwidth=30))
                    
                else:
                    print(f"   ❌ Could not retrieve sample data")
                    
            except Exception as e:
                print(f"   ❌ Error analyzing table: {str(e)}")
        
        print(f"\n" + "=" * 80)
        print(f"✅ Completed analysis of {len(tables)} tables")
        print("=" * 80)
    
    def show_samples_only(self):
        """Show just table names and sample data (compact version)."""
        print("📋 TABLE SAMPLES")
        print("=" * 50)
        
        tables = self.get_all_tables()
        
        if not tables:
            print("❌ No tables found")
            return
        
        for table in tables:
            try:
                with self.engine.connect() as conn:
                    count_result = conn.execute(text(f'SELECT COUNT(*) FROM "{table}"'))
                    row_count = count_result.scalar()
                
                print(f"\n🗂️  {table} ({row_count:,} rows)")
                print("-" * 30)
                
                if row_count > 0:
                    sample_results = self.execute_query(f'SELECT * FROM "{table}" LIMIT 5')
                    if sample_results and 'error' not in sample_results[0]:
                        df = pd.DataFrame(sample_results)
                        # Show only first 6 columns for compact view
                        display_cols = df.columns[:6] if len(df.columns) > 6 else df.columns
                        print(df[display_cols].to_string(index=False, max_colwidth=15))
                        if len(df.columns) > 6:
                            print(f"... +{len(df.columns) - 6} more columns")
                else:
                    print("(Empty)")
                    
            except Exception as e:
                print(f"❌ Error: {str(e)}")
        
        print(f"\n✅ Showed samples from {len(tables)} tables")
    
    def search_data(self, search_term: str, limit: int = 5):
        """Search for data across all text columns in all tables."""
        print(f"🔍 Searching for '{search_term}' across all tables...")
        print("=" * 60)
        
        tables = self.get_all_tables()
        found_results = False
        
        for table in tables:
            try:
                with self.engine.connect() as conn:
                    text_columns_result = conn.execute(text("""
                        SELECT column_name 
                        FROM information_schema.columns 
                        WHERE table_name = :table_name 
                        AND data_type IN ('text', 'character varying', 'varchar')
                        AND column_name NOT IN ('id', 'original_columns')
                    """), {"table_name": table})
                    
                    text_columns = [row.column_name for row in text_columns_result]
                    
                    if text_columns:
                        search_conditions = []
                        for col in text_columns:
                            search_conditions.append(f'"{col}"::text ILIKE :search_term')
                        
                        if search_conditions:
                            search_query = f"""
                                SELECT * FROM "{table}"
                                WHERE {' OR '.join(search_conditions)}
                                LIMIT {limit}
                            """
                            
                            results = conn.execute(text(search_query), 
                                                 {"search_term": f"%{search_term}%"})
                            results_list = [dict(row._mapping) for row in results]
                            
                            if results_list:
                                found_results = True
                                print(f"\n📋 Found in table '{table}':")
                                df = pd.DataFrame(results_list)
                                cols_to_show = df.columns[:6] if len(df.columns) > 6 else df.columns
                                print(df[cols_to_show].to_string(index=False))
                                if len(df.columns) > 6:
                                    print(f"... and {len(df.columns) - 6} more columns")
            
            except Exception as e:
                continue
        
        if not found_results:
            print(f"❌ No results found for '{search_term}'")
    
    def run_custom_query(self, query: str):
        """Run a custom SQL query."""
        print("🔍 Executing Custom Query...")
        print("=" * 60)
        print(f"Query: {query}")
        print("-" * 40)
        
        try:
            results = self.execute_query(query)
            if results and 'error' not in results[0]:
                if len(results) > 0:
                    df = pd.DataFrame(results)
                    print(df.to_string(index=False))
                else:
                    print("✅ Query executed successfully (no results)")
            else:
                print(f"❌ {results[0].get('error', 'Unknown error')}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def count_all_tables(self):
        """Count rows in all tables."""
        print("📊 Row Counts for All Tables")
        print("=" * 40)
        
        tables = self.get_all_tables()
        table_counts = []
        
        for table in tables:
            try:
                results = self.execute_query(f'SELECT COUNT(*) as count FROM "{table}"')
                count = results[0]['count'] if results and 'error' not in results[0] else 0
                table_counts.append({'Table': table, 'Rows': f"{count:,}"})
            except:
                table_counts.append({'Table': table, 'Rows': 'Error'})
        
        if table_counts:
            df = pd.DataFrame(table_counts)
            print(df.to_string(index=False))
    
    def show_table_structure(self, table_name: str):
        """Show structure of a specific table."""
        print(f"📋 Structure of table '{table_name}'")
        print("=" * 50)
        
        query = """
            SELECT 
                column_name,
                data_type,
                is_nullable,
                column_default,
                character_maximum_length
            FROM information_schema.columns
            WHERE table_name = :table_name
            ORDER BY ordinal_position
        """
        
        results = self.execute_query(query, {"table_name": table_name})
        if results and 'error' not in results[0]:
            df = pd.DataFrame(results)
            print(df.to_string(index=False))
            
            print(f"\n📝 Sample data from '{table_name}':")
            sample_results = self.execute_query(f'SELECT * FROM "{table_name}" LIMIT 3')
            if sample_results and 'error' not in sample_results[0]:
                sample_df = pd.DataFrame(sample_results)
                print(sample_df.to_string(index=False))
        else:
            print(f"❌ Table '{table_name}' not found or error occurred")


def main():
    """CLI interface for the query executor."""
    parser = argparse.ArgumentParser(description='Simple Query Executor for CEO RAG Database')
    parser.add_argument('--create-views', action='store_true', help='Create useful database views')
    parser.add_argument('--queries', action='store_true', help='Run predefined queries')
    parser.add_argument('--search', type=str, help='Search for text across all tables')
    parser.add_argument('--custom', type=str, help='Execute custom SQL query')
    parser.add_argument('--count', action='store_true', help='Count rows in all tables')
    parser.add_argument('--structure', type=str, help='Show structure of specific table')
    parser.add_argument('--samples', action='store_true', help='Show table names and sample data only')
    parser.add_argument('--all', action='store_true', help='Run all operations')
    
    args = parser.parse_args()
    
    executor = SimpleQueryExecutor()
    
    if args.all:
        print("🚀 Running All Operations...\n")
        executor.create_views()
        print("\n")
        executor.run_predefined_queries()
        print("\n")
        executor.count_all_tables()
    elif args.create_views:
        executor.create_views()
    elif args.queries:
        executor.run_predefined_queries()
    elif args.search:
        executor.search_data(args.search)
    elif args.custom:
        executor.run_custom_query(args.custom)
    elif args.count:
        executor.count_all_tables()
    elif args.samples:
        executor.show_samples_only()
    elif args.structure:
        executor.show_table_structure(args.structure)
    else:
        print("No operation specified. Use --help for options.")
        print("\nQuick examples:")
        print("  python simple_query_executor.py --all")
        print("  python simple_query_executor.py --queries")
        print("  python simple_query_executor.py --samples")
        print("  python simple_query_executor.py --search 'revenue'")
        print("  python simple_query_executor.py --custom 'SELECT COUNT(*) FROM information_schema.tables'")


if __name__ == "__main__":
    main()
