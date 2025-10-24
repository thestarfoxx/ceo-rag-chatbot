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
            'password': os.getenv('POSTGRES_PASSWORD', '')
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
    
    def execute_on_all_tables(self, query_template: str, table_placeholder: str = "{table}") -> Dict[str, List[Dict[str, Any]]]:
        """Execute a query template on all tables."""
        tables = self.get_all_tables()
        results = {}
        
        for table in tables:
            query = query_template.replace(table_placeholder, f'"{table}"')
            try:
                results[table] = self.execute_query(query)
            except Exception as e:
                results[table] = [{'error': f"Failed on {table}: {str(e)}"}]
        
        return results
    
    def create_views(self):
        """Create useful database views."""
        views = {
            'v_excel_files_summary': """
                CREATE OR REPLACE VIEW v_excel_files_summary AS
                SELECT 
                    table_name,
                    file_name,
                    sheet_name,
                    COUNT(*) as total_rows,
                    MIN(created_at) as first_imported,
                    MAX(created_at) as last_imported,
                    COUNT(DISTINCT file_name) as unique_files
                FROM (
                    {}
                ) all_data
                GROUP BY table_name, file_name, sheet_name
                ORDER BY last_imported DESC
            """,
            
            'v_table_stats': """
                CREATE OR REPLACE VIEW v_table_stats AS
                SELECT 
                    schemaname,
                    tablename,
                    n_live_tup as live_rows,
                    n_dead_tup as dead_rows,
                    last_vacuum,
                    last_analyze,
                    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as table_size
                FROM pg_stat_user_tables
                ORDER BY n_live_tup DESC
            """,
            
            'v_recent_imports': """
                CREATE OR REPLACE VIEW v_recent_imports AS
                SELECT 
                    table_name,
                    file_name,
                    COUNT(*) as records,
                    MAX(created_at) as import_time,
                    EXTRACT(EPOCH FROM (NOW() - MAX(created_at)))/3600 as hours_ago
                FROM (
                    {}
                ) all_recent
                WHERE created_at >= NOW() - INTERVAL '7 days'
                GROUP BY table_name, file_name
                ORDER BY import_time DESC
            """
        }
        
        # Get all tables with required columns
        tables_with_metadata = []
        for table in self.get_all_tables():
            try:
                # Check if table has the required columns
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
            
            # Create views that need UNION
            try:
                with self.engine.connect() as conn:
                    # Excel files summary view
                    conn.execute(text(views['v_excel_files_summary'].format(union_query)))
                    conn.commit()
                    print("✅ Created view: v_excel_files_summary")
                    
                    # Recent imports view
                    conn.execute(text(views['v_recent_imports'].format(union_query)))
                    conn.commit()
                    print("✅ Created view: v_recent_imports")
                    
            except Exception as e:
                print(f"❌ Error creating Excel views: {e}")
        
        # Create table stats view (doesn't need UNION)
        try:
            with self.engine.connect() as conn:
                conn.execute(text(views['v_table_stats']))
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
                    pg_size_pretty(pg_total_relation_size(table_name)) as size
                FROM information_schema.tables t
                WHERE table_schema = 'public'
                ORDER BY pg_total_relation_size(table_name) DESC
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
                        # Convert to DataFrame for better formatting
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
    
    def search_data(self, search_term: str, limit: int = 5):
        """Search for data across all text columns in all tables."""
        print(f"🔍 Searching for '{search_term}' across all tables...")
        print("=" * 60)
        
        tables = self.get_all_tables()
        found_results = False
        
        for table in tables:
            try:
                with self.engine.connect() as conn:
                    # Get text columns for this table
                    text_columns_result = conn.execute(text("""
                        SELECT column_name 
                        FROM information_schema.columns 
                        WHERE table_name = :table_name 
                        AND data_type IN ('text', 'character varying', 'varchar')
                        AND column_name NOT IN ('id', 'original_columns')
                    """), {"table_name": table})
                    
                    text_columns = [row.column_name for row in text_columns_result]
                    
                    if text_columns:
                        # Build search query
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
                                # Show only first few columns to avoid wide output
                                cols_to_show = df.columns[:6] if len(df.columns) > 6 else df.columns
                                print(df[cols_to_show].to_string(index=False))
                                if len(df.columns) > 6:
                                    print(f"... and {len(df.columns) - 6} more columns")
            
            except Exception as e:
                continue  # Skip problematic tables
        
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
            
            # Show sample data
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
    elif args.structure:
        executor.show_table_structure(args.structure)
    else:
        print("No operation specified. Use --help for options.")
        print("\nQuick examples:")
        print("  python simple_query_executor.py --all")
        print("  python simple_query_executor.py --queries")
        print("  python simple_query_executor.py --search 'revenue'")
        print("  python simple_query_executor.py --custom 'SELECT COUNT(*) FROM information_schema.tables'")


if __name__ == "__main__":
    main()
