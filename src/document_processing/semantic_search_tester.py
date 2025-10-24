#!/usr/bin/env python3
"""
Faaliyet Raporu Query Tester
Tests vector search capabilities with Turkish corporate activity report queries
"""

import psycopg2
from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine, text
import pandas as pd
import json
from typing import Dict, Any, List, Optional
import os
from datetime import datetime
import openai
import time
import argparse

class FaaliyetRaporuQueryTester:
    """Test vector search with Turkish corporate activity report queries."""
    
    def __init__(self, db_config: Dict[str, str] = None, openai_api_key: str = None, embedding_model: str = "text-embedding-3-small"):
        """Initialize query tester."""
        self.db_config = db_config or self._get_default_db_config()
        self.engine = self._create_engine()
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.embedding_model = embedding_model
        
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
        else:
            print("⚠️  No OpenAI API key provided. Vector search will not work.")
    
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
    
    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for query text."""
        if not self.openai_api_key:
            return None
        
        try:
            response = openai.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"❌ Error generating embedding: {e}")
            return None
    
    def get_available_vector_tables(self) -> List[Dict[str, str]]:
        """Get list of available vector tables."""
        try:
            with self.engine.connect() as conn:
                # First get the basic document info
                result = conn.execute(text("""
                    SELECT 
                        d.vector_table_name,
                        d.file_name,
                        d.total_chunks
                    FROM pdf_documents d
                    WHERE d.vector_table_created = TRUE 
                    AND d.processing_status = 'completed'
                    ORDER BY d.file_name
                """))
                
                tables = []
                for row in result:
                    # Get actual chunk count by querying the table directly
                    try:
                        count_result = conn.execute(text(f'SELECT COUNT(*) FROM "{row.vector_table_name}"')).scalar()
                        tables.append({
                            'table_name': row.vector_table_name,
                            'file_name': row.file_name,
                            'chunk_count': count_result
                        })
                    except Exception as e:
                        # If table doesn't exist or can't be queried, use recorded count
                        tables.append({
                            'table_name': row.vector_table_name,
                            'file_name': row.file_name,
                            'chunk_count': row.total_chunks or 0
                        })
                
                return tables
        except Exception as e:
            print(f"❌ Error getting vector tables: {e}")
            return []
    
    def search_vector_table(self, table_name: str, query_text: str, limit: int = 5, similarity_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Search a specific vector table."""
        if not self.openai_api_key:
            print("❌ No OpenAI API key provided")
            return []
        
        # Generate embedding for query
        query_embedding = self._generate_embedding(query_text)
        if not query_embedding:
            return []
        
        try:
            with self.engine.connect() as conn:
                # Check if table has embeddings
                has_embeddings = conn.execute(text(f"""
                    SELECT COUNT(*) FROM "{table_name}" WHERE embedding IS NOT NULL
                """)).scalar()
                
                if has_embeddings == 0:
                    return [{'error': f'No embeddings found in table {table_name}'}]
                
                # Perform vector search
                search_sql = f"""
                SELECT 
                    chunk_index,
                    page_number,
                    chunk_tokens,
                    chunk_text,
                    chunk_type,
                    1 - (embedding <=> :query_embedding::vector) AS similarity_score
                FROM "{table_name}"
                WHERE embedding IS NOT NULL
                AND 1 - (embedding <=> :query_embedding::vector) >= :similarity_threshold
                ORDER BY embedding <=> :query_embedding::vector
                LIMIT :limit
                """
                
                result = conn.execute(text(search_sql), {
                    'query_embedding': query_embedding,
                    'similarity_threshold': similarity_threshold,
                    'limit': limit
                })
                
                results = []
                for row in result:
                    results.append({
                        'chunk_index': row.chunk_index,
                        'page_number': row.page_number,
                        'chunk_tokens': row.chunk_tokens,
                        'chunk_text': row.chunk_text,
                        'chunk_type': row.chunk_type,
                        'similarity_score': round(float(row.similarity_score), 4)
                    })
                
                return results
                
        except Exception as e:
            return [{'error': f'Error searching table {table_name}: {e}'}]
    
    def test_faaliyet_raporu_queries(self, table_name: str = None) -> Dict[str, Any]:
        """Test various faaliyet raporu (activity report) related queries."""
        
        # Define test queries in Turkish and English
        test_queries = {
            # Financial Performance Queries
            "finansal_performans": [
                "finansal performans",
                "financial performance",
                "gelir ve kâr",
                "revenue and profit",
                "satış gelirleri",
                "sales revenue",
                "net kâr",
                "net profit"
            ],
            
            # Risk Management Queries  
            "risk_yonetimi": [
                "risk yönetimi",
                "risk management", 
                "risklerin değerlendirilmesi",
                "risk assessment",
                "iç kontrol sistemi",
                "internal control system",
                "risk faktörleri",
                "risk factors"
            ],
            
            # Corporate Governance Queries
            "kurumsal_yonetim": [
                "kurumsal yönetim",
                "corporate governance",
                "yönetim kurulu",
                "board of directors",
                "şeffaflık",
                "transparency",
                "hesap verebilirlik",
                "accountability"
            ],
            
            # Operations Queries
            "operasyonlar": [
                "operasyonel faaliyetler",
                "operational activities",
                "üretim kapasitesi",
                "production capacity",
                "operasyonel verimlilik",
                "operational efficiency",
                "iş süreçleri",
                "business processes"
            ],
            
            # Investment Queries
            "yatirimlar": [
                "yatırım harcamaları",
                "investment expenditures",
                "sermaye yatırımları", 
                "capital investments",
                "ar-ge yatırımları",
                "R&D investments",
                "teknoloji yatırımları",
                "technology investments"
            ],
            
            # Sustainability Queries
            "surdurulebilirlik": [
                "sürdürülebilirlik",
                "sustainability",
                "çevresel sorumluluk",
                "environmental responsibility",
                "sosyal sorumluluk",
                "social responsibility",
                "sürdürülebilir kalkınma",
                "sustainable development"
            ],
            
            # Human Resources Queries
            "insan_kaynaklari": [
                "insan kaynakları",
                "human resources",
                "personel sayısı",
                "number of employees",
                "eğitim ve geliştirme",
                "training and development",
                "çalışan memnuniyeti",
                "employee satisfaction"
            ],
            
            # Market and Competition Queries
            "pazar_rekabet": [
                "pazar payı",
                "market share",
                "rekabet ortamı",
                "competitive environment",
                "sektör analizi",
                "sector analysis",
                "müşteri memnuniyeti",
                "customer satisfaction"
            ],
            
            # Future Plans Queries
            "gelecek_planlar": [
                "gelecek dönem planları",
                "future period plans",
                "stratejik hedefler",
                "strategic targets",
                "büyüme stratejisi",
                "growth strategy",
                "önümüzdeki dönem",
                "upcoming period"
            ],
            
            # Specific Turkish Corporate Terms
            "turkiye_ozel": [
                "faaliyet raporu",
                "activity report",
                "yıllık rapor", 
                "annual report",
                "genel kurul",
                "general assembly",
                "SPK düzenlemeleri",
                "CMB regulations",
                "TFRS",
                "Turkish Financial Reporting Standards",
                "Borsa İstanbul",
                "Borsa Istanbul"
            ]
        }
        
        # Get available tables
        available_tables = self.get_available_vector_tables()
        if not available_tables:
            return {'error': 'No vector tables available'}
        
        # Determine which table to test
        if table_name:
            target_tables = [t for t in available_tables if t['table_name'] == table_name]
            if not target_tables:
                return {'error': f'Table {table_name} not found'}
        else:
            # Use the table with most chunks
            if available_tables:
                target_tables = [max(available_tables, key=lambda x: x['chunk_count'])]
            else:
                return {'error': 'No available tables found'}
        
        results = {}
        
        for table_info in target_tables:
            table_name = table_info['table_name']
            file_name = table_info['file_name']
            
            print(f"\n🔍 Testing queries on: {file_name} ({table_name})")
            print("=" * 60)
            
            table_results = {}
            
            for category, queries in test_queries.items():
                print(f"\n📂 Category: {category.replace('_', ' ').title()}")
                category_results = {}
                
                for query in queries:
                    print(f"  🔎 Searching: '{query}'")
                    
                    search_results = self.search_vector_table(
                        table_name, 
                        query, 
                        limit=3, 
                        similarity_threshold=0.6
                    )
                    
                    if search_results and len(search_results) > 0 and 'error' not in search_results[0]:
                        print(f"    ✅ Found {len(search_results)} results")
                        for i, result in enumerate(search_results):
                            similarity = result['similarity_score']
                            preview = result['chunk_text'][:100].replace('\n', ' ')
                            print(f"      {i+1}. Similarity: {similarity:.3f} | Page: {result['page_number']} | Preview: {preview}...")
                    else:
                        print(f"    ❌ No results found")
                    
                    category_results[query] = search_results
                    time.sleep(0.1)  # Rate limiting
                
                table_results[category] = category_results
            
            results[table_name] = {
                'file_name': file_name,
                'chunk_count': table_info['chunk_count'],
                'query_results': table_results
            }
        
        return results
    
    def analyze_search_patterns(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze search result patterns."""
        analysis = {
            'total_queries': 0,
            'successful_queries': 0,
            'categories_with_results': {},
            'best_similarity_scores': [],
            'query_performance': {}
        }
        
        # Handle case where results might contain error
        if isinstance(results, dict) and 'error' in results:
            return analysis
        
        for table_name, table_data in results.items():
            if not isinstance(table_data, dict) or 'error' in table_data:
                continue
                
            if 'query_results' not in table_data:
                continue
                
            query_results = table_data['query_results']
            
            for category, category_queries in query_results.items():
                category_success = 0
                category_total = 0
                
                for query, search_results in category_queries.items():
                    analysis['total_queries'] += 1
                    category_total += 1
                    
                    if (search_results and 
                        isinstance(search_results, list) and 
                        len(search_results) > 0 and 
                        isinstance(search_results[0], dict) and 
                        'error' not in search_results[0]):
                        
                        analysis['successful_queries'] += 1
                        category_success += 1
                        
                        # Collect similarity scores
                        for result in search_results:
                            if isinstance(result, dict) and 'similarity_score' in result:
                                analysis['best_similarity_scores'].append(result['similarity_score'])
                
                analysis['categories_with_results'][category] = {
                    'success_rate': round((category_success / category_total * 100), 2) if category_total > 0 else 0,
                    'successful_queries': category_success,
                    'total_queries': category_total
                }
        
        # Calculate overall metrics
        analysis['overall_success_rate'] = round((analysis['successful_queries'] / analysis['total_queries'] * 100), 2) if analysis['total_queries'] > 0 else 0
        analysis['avg_similarity_score'] = round(sum(analysis['best_similarity_scores']) / len(analysis['best_similarity_scores']), 3) if analysis['best_similarity_scores'] else 0
        analysis['max_similarity_score'] = max(analysis['best_similarity_scores']) if analysis['best_similarity_scores'] else 0
        
        return analysis
    
    def generate_test_report(self, results: Dict[str, Any], output_file: str = None) -> str:
        """Generate comprehensive test report."""
        analysis = self.analyze_search_patterns(results)
        
        report_lines = []
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report_lines.append("=" * 80)
        report_lines.append("FAALİYET RAPORU QUERY TEST REPORT")
        report_lines.append(f"Generated: {timestamp}")
        report_lines.append("=" * 80)
        
        # Summary
        report_lines.append("\n📊 SUMMARY")
        report_lines.append("-" * 30)
        report_lines.append(f"Total Queries Tested: {analysis['total_queries']}")
        report_lines.append(f"Successful Queries: {analysis['successful_queries']}")
        report_lines.append(f"Overall Success Rate: {analysis['overall_success_rate']}%")
        report_lines.append(f"Average Similarity Score: {analysis['avg_similarity_score']}")
        report_lines.append(f"Best Similarity Score: {analysis['max_similarity_score']}")
        
        # Category Performance
        report_lines.append("\n📈 CATEGORY PERFORMANCE")
        report_lines.append("-" * 30)
        for category, stats in analysis['categories_with_results'].items():
            report_lines.append(f"{category.replace('_', ' ').title()}: {stats['success_rate']}% ({stats['successful_queries']}/{stats['total_queries']})")
        
        # Detailed Results
        report_lines.append("\n🔍 DETAILED RESULTS")
        report_lines.append("-" * 30)
        
        for table_name, table_data in results.items():
            if 'error' in table_data:
                report_lines.append(f"❌ Error in table {table_name}: {table_data['error']}")
                continue
            
            report_lines.append(f"\nTable: {table_name}")
            report_lines.append(f"File: {table_data['file_name']}")
            report_lines.append(f"Chunks: {table_data['chunk_count']}")
            
            for category, category_queries in table_data['query_results'].items():
                successful_in_category = 0
                total_in_category = len(category_queries)
                
                for query, search_results in category_queries.items():
                    if search_results and 'error' not in search_results[0]:
                        successful_in_category += 1
                
                success_rate = (successful_in_category / total_in_category * 100) if total_in_category > 0 else 0
                icon = "✅" if success_rate > 50 else ("⚠️" if success_rate > 0 else "❌")
                report_lines.append(f"  {icon} {category.replace('_', ' ').title()}: {success_rate:.1f}% ({successful_in_category}/{total_in_category})")
        
        # Best Performing Queries
        report_lines.append("\n🏆 TOP PERFORMING QUERIES")
        report_lines.append("-" * 30)
        
        all_query_results = []
        for table_name, table_data in results.items():
            if isinstance(table_data, dict) and 'query_results' in table_data:
                for category_queries in table_data['query_results'].values():
                    for query, search_results in category_queries.items():
                        if (search_results and 
                            isinstance(search_results, list) and 
                            len(search_results) > 0 and 
                            isinstance(search_results[0], dict) and 
                            'error' not in search_results[0]):
                            
                            # Get best similarity score from results
                            best_score = 0
                            for r in search_results:
                                if isinstance(r, dict) and 'similarity_score' in r:
                                    best_score = max(best_score, r['similarity_score'])
                            
                            if best_score > 0:
                                all_query_results.append((query, best_score))
        
        # Sort by similarity score
        all_query_results.sort(key=lambda x: x[1], reverse=True)
        
        for i, (query, score) in enumerate(all_query_results[:10]):
            report_lines.append(f"  {i+1}. '{query}' - Similarity: {score:.3f}")
        
        if not all_query_results:
            report_lines.append("  No successful queries found")
        
        report_lines.append("\n" + "=" * 80)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 80)
        
        report_text = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"📝 Report saved to: {output_file}")
        
        return report_text
    
    def interactive_query_test(self):
        """Run interactive query testing."""
        print("\n🔍 Faaliyet Raporu Query Tester - Interactive Mode")
        print("=" * 60)
        
        # Get available tables
        tables = self.get_available_vector_tables()
        if not tables:
            print("❌ No vector tables found!")
            return
        
        print(f"\nFound {len(tables)} vector tables:")
        for i, table in enumerate(tables):
            print(f"  {i+1}. {table['file_name']} ({table['chunk_count']} chunks)")
        
        while True:
            print("\nOptions:")
            print("1. Test all predefined queries")
            print("2. Test custom query")
            print("3. Test specific category")
            print("4. Show available tables")
            print("5. Exit")
            
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == '1':
                table_choice = input(f"Enter table number (1-{len(tables)}) or 'all': ").strip()
                
                if table_choice.lower() == 'all':
                    table_name = None
                else:
                    try:
                        table_idx = int(table_choice) - 1
                        if 0 <= table_idx < len(tables):
                            table_name = tables[table_idx]['table_name']
                        else:
                            print("Invalid table number!")
                            continue
                    except ValueError:
                        print("Please enter a valid number!")
                        continue
                
                print("\n🚀 Running all predefined queries...")
                results = self.test_faaliyet_raporu_queries(table_name)
                
                save_report = input("\nSave detailed report? (y/n): ").strip().lower()
                if save_report == 'y':
                    filename = f"faaliyet_raporu_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                    self.generate_test_report(results, filename)
                else:
                    print(self.generate_test_report(results))
            
            elif choice == '2':
                table_choice = input(f"Enter table number (1-{len(tables)}): ").strip()
                try:
                    table_idx = int(table_choice) - 1
                    if 0 <= table_idx < len(tables):
                        table_name = tables[table_idx]['table_name']
                        query = input("Enter your query: ").strip()
                        
                        if query:
                            print(f"\n🔎 Searching '{query}' in {tables[table_idx]['file_name']}...")
                            results = self.search_vector_table(table_name, query, limit=5)
                            
                            if results and 'error' not in results[0]:
                                print(f"✅ Found {len(results)} results:")
                                for i, result in enumerate(results):
                                    print(f"\n  Result {i+1}:")
                                    print(f"    Similarity: {result['similarity_score']:.3f}")
                                    print(f"    Page: {result['page_number']}")
                                    print(f"    Tokens: {result['chunk_tokens']}")
                                    print(f"    Preview: {result['chunk_text'][:200]}...")
                            else:
                                print("❌ No results found or error occurred")
                    else:
                        print("Invalid table number!")
                except ValueError:
                    print("Please enter a valid number!")
            
            elif choice == '4':
                print(f"\nAvailable tables:")
                for i, table in enumerate(tables):
                    print(f"  {i+1}. {table['file_name']} -> {table['table_name']} ({table['chunk_count']} chunks)")
            
            elif choice == '5':
                print("👋 Goodbye!")
                break
            
            else:
                print("Invalid choice! Please enter 1-5.")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Faaliyet Raporu Query Tester')
    parser.add_argument('--host', default='localhost', help='Database host')
    parser.add_argument('--port', default='5432', help='Database port')
    parser.add_argument('--database', default='ceo_rag_db', help='Database name')
    parser.add_argument('--username', default='postgres', help='Database username')
    parser.add_argument('--password', default='', help='Database password')
    parser.add_argument('--openai-key', help='OpenAI API key')
    parser.add_argument('--table', '-t', help='Test specific table')
    parser.add_argument('--query', '-q', help='Test specific query')
    parser.add_argument('--output', '-o', help='Output file for report')
    parser.add_argument('--interactive', '-i', action='store_true', help='Run in interactive mode')
    
    args = parser.parse_args()
    
    # Set up database configuration
    db_config = {
        'host': args.host,
        'port': args.port,
        'database': args.database,
        'username': args.username,
        'password': args.password
    }
    
    # Initialize tester
    tester = FaaliyetRaporuQueryTester(db_config, args.openai_key)
    
    if args.interactive:
        tester.interactive_query_test()
    elif args.query:
        # Test specific query
        tables = tester.get_available_vector_tables()
        if not tables:
            print("❌ No vector tables found!")
            return
        
        target_table = args.table if args.table else tables[0]['table_name']
        results = tester.search_vector_table(target_table, args.query)
        
        print(f"\n🔎 Query: '{args.query}'")
        print(f"📂 Table: {target_table}")
        print("=" * 50)
        
        if results and 'error' not in results[0]:
            for i, result in enumerate(results):
                print(f"\nResult {i+1}:")
                print(f"  Similarity: {result['similarity_score']:.3f}")
                print(f"  Page: {result['page_number']}")
                print(f"  Preview: {result['chunk_text'][:150]}...")
        else:
            print("❌ No results found")
    else:
        # Run full test suite
        print("🚀 Running comprehensive faaliyet raporu query tests...")
        results = tester.test_faaliyet_raporu_queries(args.table)
        report = tester.generate_test_report(results, args.output)
        
        if not args.output:
            print(report)


if __name__ == "__main__":
    main()
