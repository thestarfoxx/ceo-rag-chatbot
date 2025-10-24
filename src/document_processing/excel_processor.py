import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine, text, MetaData, Table, Column, String, Float, Integer, DateTime, Boolean
from sqlalchemy.dialects.postgresql import TEXT, JSONB
import logging
from pathlib import Path
import hashlib
import json
from typing import Dict, List, Optional, Any
import re
from datetime import datetime
import os
import unicodedata
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logger = logging.getLogger(__name__)

class ExcelFileHandler(FileSystemEventHandler):
    """File system event handler for Excel files."""
    
    def __init__(self, processor):
        self.processor = processor
        self.excel_extensions = {'.xlsx', '.xls', '.xlsm'}
    
    def on_created(self, event):
        """Handle file creation events."""
        if not event.is_directory:
            file_path = Path(event.src_path)
            if file_path.suffix.lower() in self.excel_extensions:
                logger.info(f"New Excel file detected: {file_path}")
                self.processor.auto_process_file(str(file_path))
    
    def on_moved(self, event):
        """Handle file move events (like file uploads)."""
        if not event.is_directory:
            file_path = Path(event.dest_path)
            if file_path.suffix.lower() in self.excel_extensions:
                logger.info(f"Excel file moved/uploaded: {file_path}")
                self.processor.auto_process_file(str(file_path))

class ExcelProcessor:
    def __init__(self, db_config: Optional[Dict[str, str]] = None, watch_directory: str = None):
        """
        Initialize Excel processor with PostgreSQL connection.
        
        Args:
            db_config: Database configuration. If None, uses defaults.
            watch_directory: Directory to watch for new Excel files
        """
        self.db_config = db_config or self._get_default_db_config()
        self.engine = self._create_engine()
        self.metadata = MetaData()
        self.watch_directory = watch_directory
        self.observer = None
        
        # Create database if it doesn't exist
        self._ensure_database_exists()
        
        # Initialize Turkish character mapping
        self._init_turkish_char_mapping()
        
    def _init_turkish_char_mapping(self):
        """Initialize Turkish character to ASCII mapping."""
        self.turkish_char_map = {
            # Turkish lowercase
            'ç': 'c', 'ğ': 'g', 'ı': 'i', 'ö': 'o', 'ş': 's', 'ü': 'u',
            # Turkish uppercase
            'Ç': 'C', 'Ğ': 'G', 'İ': 'I', 'Ö': 'O', 'Ş': 'S', 'Ü': 'U',
            # Other common international characters
            'á': 'a', 'à': 'a', 'â': 'a', 'ä': 'a', 'ã': 'a', 'å': 'a',
            'Á': 'A', 'À': 'A', 'Â': 'A', 'Ä': 'A', 'Ã': 'A', 'Å': 'A',
            'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
            'É': 'E', 'È': 'E', 'Ê': 'E', 'Ë': 'E',
            'í': 'i', 'ì': 'i', 'î': 'i', 'ï': 'i',
            'Í': 'I', 'Ì': 'I', 'Î': 'I', 'Ï': 'I',
            'ó': 'o', 'ò': 'o', 'ô': 'o', 'õ': 'o',
            'Ó': 'O', 'Ò': 'O', 'Ô': 'O', 'Õ': 'O',
            'ú': 'u', 'ù': 'u', 'û': 'u',
            'Ú': 'U', 'Ù': 'U', 'Û': 'U',
            'ñ': 'n', 'Ñ': 'N',
            'ý': 'y', 'Ý': 'Y'
        }
    
    def _get_default_db_config(self) -> Dict[str, str]:
        """Get default PostgreSQL configuration."""
        return {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': os.getenv('POSTGRES_PORT', '5432'),
            'database': os.getenv('POSTGRES_DB', 'ceo_rag_db'),
            'username': os.getenv('POSTGRES_USER', 'postgres'),  # Default PostgreSQL user
            'password': os.getenv('POSTGRES_PASSWORD', 'password')       # Default is empty password
        }
    
    def _ensure_database_exists(self):
        """Create database if it doesn't exist."""
        try:
            # Connect to default 'postgres' database to create our database
            temp_config = self.db_config.copy()
            temp_config['database'] = 'postgres'
            
            temp_engine = self._create_engine_from_config(temp_config)
            
            with temp_engine.connect() as conn:
                # Check if database exists
                result = conn.execute(text("""
                    SELECT 1 FROM pg_database WHERE datname = :db_name
                """), {"db_name": self.db_config['database']})
                
                if not result.fetchone():
                    # Create database
                    conn.execute(text("COMMIT"))  # End current transaction
                    conn.execute(text(f"CREATE DATABASE {self.db_config['database']}"))
                    logger.info(f"Created database: {self.db_config['database']}")
            
            temp_engine.dispose()
            
        except Exception as e:
            logger.warning(f"Could not create database (may already exist): {str(e)}")
    
    def _create_engine_from_config(self, config: Dict[str, str]):
        """Create SQLAlchemy engine from config."""
        password_part = f":{config['password']}" if config['password'] else ""
        connection_string = (
            f"postgresql://{config['username']}{password_part}@"
            f"{config['host']}:{config['port']}/{config['database']}"
        )
        return create_engine(connection_string, echo=False)
        
    def _create_engine(self):
        """Create SQLAlchemy engine for PostgreSQL connection."""
        return self._create_engine_from_config(self.db_config)
    
    def _sanitize_table_name(self, filename: str) -> str:
        """
        Convert filename to valid PostgreSQL table name.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized table name
        """
        # Remove file extension and convert to lowercase
        name = Path(filename).stem.lower()
        
        # Normalize Turkish and international characters
        name = self._normalize_text(name)
        
        # Replace special characters with underscores
        name = re.sub(r'[^a-z0-9_]', '_', name)
        
        # Remove multiple underscores
        name = re.sub(r'_+', '_', name)
        
        # Remove leading/trailing underscores
        name = name.strip('_')
        
        # Ensure it starts with a letter or underscore
        if name and name[0].isdigit():
            name = f"excel_{name}"
        
        # Ensure name is not empty
        if not name:
            name = "excel_table"
        
        # Limit length to 63 characters (PostgreSQL limit)
        if len(name) > 63:
            # Create hash suffix to ensure uniqueness
            hash_suffix = hashlib.md5(filename.encode()).hexdigest()[:8]
            name = f"{name[:54]}_{hash_suffix}"
        
        return name
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text by replacing Turkish and international characters with ASCII equivalents.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text with ASCII characters
        """
        if not isinstance(text, str):
            text = str(text)
        
        # First, apply our custom character mapping for Turkish and common international chars
        normalized = ""
        for char in text:
            if char in self.turkish_char_map:
                normalized += self.turkish_char_map[char]
            else:
                normalized += char
        
        # Apply Unicode normalization as fallback for any remaining characters
        # NFD decomposes characters, then we keep only ASCII
        try:
            normalized = unicodedata.normalize('NFD', normalized)
            normalized = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
            # Keep only ASCII characters
            normalized = ''.join(c for c in normalized if ord(c) < 128)
        except Exception as e:
            logger.warning(f"Unicode normalization failed for '{text}': {str(e)}")
            # If normalization fails, just remove non-ASCII characters
            normalized = ''.join(c for c in normalized if ord(c) < 128)
        
        return normalized
    
    def _sanitize_column_name(self, column_name: str) -> str:
        """
        Convert column name to valid PostgreSQL column name with Turkish character normalization.
        
        Args:
            column_name: Original column name
            
        Returns:
            Sanitized column name
        """
        # Convert to string and strip whitespace
        name = str(column_name).strip()
        
        # Normalize Turkish and international characters
        name = self._normalize_text(name)
        
        # Convert to lowercase
        name = name.lower()
        
        # Replace spaces and special characters with underscores
        name = re.sub(r'[^a-z0-9_]', '_', name)
        
        # Remove multiple underscores
        name = re.sub(r'_+', '_', name)
        
        # Remove leading/trailing underscores
        name = name.strip('_')
        
        # Ensure it's not empty and doesn't start with a number
        if not name or name[0].isdigit():
            name = f"col_{name}" if name else "col_unnamed"
        
        # Limit length
        if len(name) > 63:
            # Create a hash suffix to maintain uniqueness
            hash_suffix = hashlib.md5(str(column_name).encode()).hexdigest()[:8]
            name = f"{name[:54]}_{hash_suffix}"
        
        return name or "unnamed_column"
    
    def _table_exists(self, table_name: str) -> bool:
        """
        Check if table already exists in database.
        
        Args:
            table_name: Name of the table to check
            
        Returns:
            True if table exists, False otherwise
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = :table_name
                    )
                """), {"table_name": table_name})
                
                return result.scalar()
        except Exception as e:
            logger.error(f"Error checking if table exists: {str(e)}")
            return False
    
    def _get_table_row_count(self, table_name: str) -> int:
        """
        Get the number of rows in a table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Number of rows in the table
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(f'SELECT COUNT(*) FROM "{table_name}"'))
                return result.scalar()
        except Exception as e:
            logger.error(f"Error getting row count for table {table_name}: {str(e)}")
            return 0
    
    def _should_process_file(self, file_path: str, table_name: str) -> bool:
        """
        Check if file should be processed based on table existence.
        
        Args:
            file_path: Path to the Excel file
            table_name: Name of the corresponding table
            
        Returns:
            True if file should be processed (table doesn't exist)
        """
        try:
            # Simple check: if table doesn't exist, process the file
            if not self._table_exists(table_name):
                logger.info(f"Table '{table_name}' does not exist, will create for file: {Path(file_path).name}")
                return True
            
            logger.info(f"Table '{table_name}' already exists, skipping file: {Path(file_path).name}")
            return False
                
        except Exception as e:
            logger.warning(f"Error checking table existence for {table_name}, will process: {str(e)}")
            return True
    
    def _infer_column_type(self, series: pd.Series) -> str:
        """
        Infer PostgreSQL column type from pandas Series.
        
        Args:
            series: Pandas Series to analyze
            
        Returns:
            PostgreSQL column type as string
        """
        # Drop null values for type inference
        non_null_series = series.dropna()
        
        if len(non_null_series) == 0:
            return "TEXT"
        
        # Check for datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            return "TIMESTAMP"
        
        # Check for boolean
        if pd.api.types.is_bool_dtype(series):
            return "BOOLEAN"
        
        # Check for integer
        if pd.api.types.is_integer_dtype(series):
            return "BIGINT"
        
        # Check for float
        if pd.api.types.is_numeric_dtype(series):
            return "DOUBLE PRECISION"
        
        # Default to TEXT for strings and mixed types
        return "TEXT"
    
    def _create_table_schema(self, df: pd.DataFrame, table_name: str) -> str:
        """
        Generate CREATE TABLE SQL statement for DataFrame.
        
        Args:
            df: DataFrame to analyze
            table_name: Name of the table to create
            
        Returns:
            SQL CREATE TABLE statement
        """
        columns = []
        
        # Add metadata columns
        columns.append("id SERIAL PRIMARY KEY")
        columns.append("file_name TEXT")
        columns.append("sheet_name TEXT")
        columns.append("row_number INTEGER")
        columns.append("created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        
        # Add data columns
        for col in df.columns:
            sanitized_col = self._sanitize_column_name(col)
            col_type = self._infer_column_type(df[col])
            columns.append(f'"{sanitized_col}" {col_type}')
        
        # Add original column names as JSONB for reference
        columns.append("original_columns JSONB")
        
        return f"""
        CREATE TABLE IF NOT EXISTS "{table_name}" (
            {','.join(columns)}
        )
        """
    
    def _prepare_dataframe(self, df: pd.DataFrame, file_path: str) -> pd.DataFrame:
        """
        Prepare DataFrame for database insertion.
        
        Args:
            df: Original DataFrame
            file_path: Path to the source file
            
        Returns:
            Processed DataFrame ready for insertion
        """
        # Create a copy to avoid modifying original
        processed_df = df.copy()
        
        # Store original column names mapping before sanitization
        original_columns = {}
        sanitized_columns = []
        
        for col in df.columns:
            sanitized_col = self._sanitize_column_name(col)
            original_columns[sanitized_col] = str(col)
            sanitized_columns.append(sanitized_col)
        
        # Log the column mapping for debugging
        logger.info(f"Column name mapping for {Path(file_path).name}:")
        for sanitized, original in original_columns.items():
            if sanitized != original.lower().replace(' ', '_'):
                logger.info(f"  '{original}' -> '{sanitized}'")
        
        # Sanitize column names
        processed_df.columns = sanitized_columns
        
        # Handle NaN values
        processed_df = processed_df.replace({pd.NaT: None, pd.NA: None})
        processed_df = processed_df.where(pd.notnull(processed_df), None)
        
        # Add metadata
        processed_df['row_number'] = range(1, len(processed_df) + 1)
        processed_df['original_columns'] = json.dumps(original_columns)
        
        return processed_df
    
    def auto_process_file(self, file_path: str) -> Dict[str, Any]:
        """
        Automatically process Excel file only if table doesn't exist.
        
        Args:
            file_path: Path to Excel file
            
        Returns:
            Processing result dictionary
        """
        try:
            file_path = Path(file_path)
            
            # Generate table name to check
            base_table_name = self._sanitize_table_name(file_path.name)
            
            # Check if we should process this file (table doesn't exist)
            if not self._should_process_file(str(file_path), base_table_name):
                return {
                    'success': True,
                    'file_path': str(file_path),
                    'action': 'skipped',
                    'reason': 'table_already_exists',
                    'table_name': base_table_name
                }
            
            # Process the file
            logger.info(f"Processing Excel file: {file_path}")
            return self.process_excel_file(str(file_path))
            
        except Exception as e:
            logger.error(f"Error in auto-processing {file_path}: {str(e)}")
            return {
                'success': False,
                'file_path': str(file_path),
                'error': str(e)
            }
    
    def process_excel_file(self, file_path: str, sheet_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Process Excel file and store in PostgreSQL.
        
        Args:
            file_path: Path to Excel file
            sheet_name: Specific sheet to process (if None, processes all sheets)
            
        Returns:
            Dictionary with processing results
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Excel file not found: {file_path}")
            
            # Read Excel file
            if sheet_name:
                sheets = {sheet_name: pd.read_excel(file_path, sheet_name=sheet_name)}
            else:
                sheets = pd.read_excel(file_path, sheet_name=None)
            
            results = {}
            
            for current_sheet_name, df in sheets.items():
                if df.empty:
                    logger.warning(f"Empty sheet '{current_sheet_name}' in {file_path}")
                    continue
                
                # Create table name
                base_table_name = self._sanitize_table_name(file_path.name)
                if len(sheets) > 1:
                    sheet_suffix = self._sanitize_table_name(current_sheet_name)
                    table_name = f"{base_table_name}_{sheet_suffix}"
                else:
                    table_name = base_table_name
                
                # Check if table exists - if yes, skip processing entirely
                if self._table_exists(table_name):
                    logger.info(f"Table '{table_name}' already exists, skipping sheet '{current_sheet_name}'")
                    results[current_sheet_name] = {
                        'table_name': table_name,
                        'action': 'skipped',
                        'reason': 'table_exists',
                        'existing_row_count': self._get_table_row_count(table_name)
                    }
                    continue
                
                # Create new table since it doesn't exist
                create_sql = self._create_table_schema(df, table_name)
                with self.engine.connect() as conn:
                    conn.execute(text(create_sql))
                    conn.commit()
                    logger.info(f"Created new table: {table_name}")
                
                # Prepare data
                processed_df = self._prepare_dataframe(df, str(file_path))
                processed_df['file_name'] = file_path.name
                processed_df['sheet_name'] = current_sheet_name
                
                # Insert data
                processed_df.to_sql(
                    table_name,
                    self.engine,
                    if_exists='append',
                    index=False,
                    method='multi'
                )
                
                results[current_sheet_name] = {
                    'table_name': table_name,
                    'rows_inserted': len(processed_df),
                    'columns': list(df.columns),
                    'sanitized_columns': list(processed_df.columns),
                    'action': 'created_and_populated'
                }
                
                logger.info(f"Successfully processed sheet '{current_sheet_name}' "
                           f"from {file_path} into table '{table_name}' "
                           f"({len(processed_df)} rows)")
            
            return {
                'success': True,
                'file_path': str(file_path),
                'action': 'processed',
                'sheets_processed': results,
                'total_sheets': len(results)
            }
            
        except Exception as e:
            logger.error(f"Error processing Excel file {file_path}: {str(e)}")
            return {
                'success': False,
                'file_path': str(file_path),
                'error': str(e)
            }
    
    def start_watching(self, directory: str = None):
        """
        Start watching directory for new Excel files.
        
        Args:
            directory: Directory to watch. If None, uses self.watch_directory
        """
        watch_dir = directory or self.watch_directory
        if not watch_dir:
            raise ValueError("No directory specified to watch")
        
        watch_path = Path(watch_dir)
        if not watch_path.exists():
            watch_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created watch directory: {watch_path}")
        
        # Process existing files first
        self.process_existing_files(str(watch_path))
        
        # Start watching for new files
        event_handler = ExcelFileHandler(self)
        self.observer = Observer()
        self.observer.schedule(event_handler, str(watch_path), recursive=True)
        self.observer.start()
        
        logger.info(f"Started watching directory: {watch_path}")
    
    def stop_watching(self):
        """Stop watching for file changes."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            logger.info("Stopped watching for file changes")
    
    def process_existing_files(self, directory: str):
        """
        Process all existing Excel files in directory.
        
        Args:
            directory: Directory to scan for Excel files
        """
        directory = Path(directory)
        excel_extensions = {'.xlsx', '.xls', '.xlsm'}
        
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in excel_extensions:
                logger.info(f"Processing existing file: {file_path}")
                self.auto_process_file(str(file_path))
    
    def get_table_info(self, table_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Table information dictionary or None if table doesn't exist
        """
        try:
            with self.engine.connect() as conn:
                # Check if table exists
                result = conn.execute(text("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = :table_name
                    )
                """), {"table_name": table_name})
                
                if not result.scalar():
                    return None
                
                # Get column information
                columns_result = conn.execute(text("""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_name = :table_name
                    ORDER BY ordinal_position
                """), {"table_name": table_name})
                
                columns = [dict(row._mapping) for row in columns_result]
                
                # Get row count
                count_result = conn.execute(text(f'SELECT COUNT(*) FROM "{table_name}"'))
                row_count = count_result.scalar()
                
                # Get sample data
                sample_result = conn.execute(text(f'SELECT * FROM "{table_name}" LIMIT 5'))
                sample_data = [dict(row._mapping) for row in sample_result]
                
                return {
                    'table_name': table_name,
                    'columns': columns,
                    'row_count': row_count,
                    'sample_data': sample_data
                }
                
        except Exception as e:
            logger.error(f"Error getting table info for {table_name}: {str(e)}")
            return None
    
    def list_all_tables(self) -> List[Dict[str, Any]]:
        """
        List all Excel-related tables in the database.
        
        Returns:
            List of table information dictionaries
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT table_name, 
                           (SELECT COUNT(*) FROM information_schema.columns 
                            WHERE table_name = t.table_name) as column_count
                    FROM information_schema.tables t
                    WHERE table_schema = 'public' 
                    AND table_type = 'BASE TABLE'
                    ORDER BY table_name
                """))
                
                tables = []
                for row in result:
                    table_info = self.get_table_info(row.table_name)
                    if table_info:
                        tables.append(table_info)
                
                return tables
                
        except Exception as e:
            logger.error(f"Error listing tables: {str(e)}")
            return []
    
    def query_table(self, table_name: str, query: str = None, limit: int = 100) -> Optional[List[Dict[str, Any]]]:
        """
        Query a specific table.
        
        Args:
            table_name: Name of the table to query
            query: Optional WHERE clause (without 'WHERE' keyword)
            limit: Maximum number of rows to return
            
        Returns:
            Query results or None if error
        """
        try:
            with self.engine.connect() as conn:
                if query:
                    sql = f'SELECT * FROM "{table_name}" WHERE {query} LIMIT {limit}'
                else:
                    sql = f'SELECT * FROM "{table_name}" LIMIT {limit}'
                
                result = conn.execute(text(sql))
                return [dict(row._mapping) for row in result]
                
        except Exception as e:
            logger.error(f"Error querying table {table_name}: {str(e)}")
            return None
    
    def get_original_column_mapping(self, table_name: str) -> Optional[Dict[str, str]]:
        """
        Get the original column names mapping for a table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Dictionary mapping sanitized column names to original names
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(f'SELECT original_columns FROM "{table_name}" LIMIT 1'))
                row = result.fetchone()
                
                if row and row.original_columns:
                    return json.loads(row.original_columns)
                
                return None
                
        except Exception as e:
            logger.error(f"Error getting original column mapping for {table_name}: {str(e)}")
            return None
    
    def generate_summary(self, table_name: str) -> str:
        """
        Generate a text summary of the table for vector storage.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Text summary of the table
        """
        table_info = self.get_table_info(table_name)
        if not table_info:
            return ""
        
        # Extract metadata
        file_name = ""
        sheet_name = ""
        if table_info['sample_data']:
            file_name = table_info['sample_data'][0].get('file_name', '')
            sheet_name = table_info['sample_data'][0].get('sheet_name', '')
        
        # Get original column names if available
        original_mapping = self.get_original_column_mapping(table_name)
        
        # Build summary
        summary_parts = [
            f"Excel Data Table: {table_name}",
            f"Source File: {file_name}",
            f"Sheet: {sheet_name}",
            f"Total Records: {table_info['row_count']}",
        ]
        
        # Add column information
        data_columns = [col for col in table_info['columns'] 
                       if col['column_name'] not in ['id', 'file_name', 'sheet_name', 'row_number', 'created_at', 'original_columns']]
        
        if data_columns:
            summary_parts.append(f"Data Columns ({len(data_columns)}):")
            for col in data_columns[:15]:  # Limit to first 15 columns
                col_name = col['column_name']
                # Show original column name if available
                if original_mapping and col_name in original_mapping:
                    original_name = original_mapping[col_name]
                    if original_name != col_name:
                        col_display = f"{original_name} ({col_name})"
                    else:
                        col_display = col_name
                else:
                    col_display = col_name
                
                summary_parts.append(f"- {col_display} ({col['data_type']})")
                
            if len(data_columns) > 15:
                summary_parts.append(f"... and {len(data_columns) - 15} more columns")
        
        # Add sample data context if available
        if table_info['sample_data'] and len(table_info['sample_data']) > 0:
            summary_parts.append("Contains structured financial/business data suitable for analysis and reporting.")
        
        return "\n".join(summary_parts)
    
    def close(self):
        """Close database connections and stop watching."""
        self.stop_watching()
        if hasattr(self, 'engine'):
            self.engine.dispose()


# Example usage for integration
def setup_excel_processor(excel_directory: str = "data/excel/") -> ExcelProcessor:
    """
    Set up Excel processor with automatic file watching.
    
    Args:
        excel_directory: Directory containing Excel files
        
    Returns:
        Configured ExcelProcessor instance
    """
    # Default PostgreSQL configuration
    db_config = {
        'host': 'localhost',
        'port': '5432',
        'database': 'ceo_rag_db',
        'username': 'postgres',  # Default PostgreSQL superuser
        'password': 'password'           # Default is no password
    }
    
    # Initialize processor
    processor = ExcelProcessor(db_config, excel_directory)
    
    # Start watching for new files
    processor.start_watching()
    
    return processor


if __name__ == "__main__":
    import time
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize processor and start watching
    processor = setup_excel_processor("data/excel/")
    
    try:
        print("Excel processor started. Watching for new files...")
        print("Press Ctrl+C to stop")
        
        # Keep the script running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping Excel processor...")
        processor.close()
        print("Excel processor stopped.")
