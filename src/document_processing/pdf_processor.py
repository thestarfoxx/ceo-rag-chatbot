import PyPDF2
import pdfplumber
import fitz  # PyMuPDF
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine, text, MetaData, Table, Column, String, Float, Integer, DateTime, Boolean
from sqlalchemy.dialects.postgresql import TEXT, JSONB
import logging
from pathlib import Path
import hashlib
import json
from typing import Dict, List, Optional, Any, Tuple
import re
from datetime import datetime
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import tiktoken
import numpy as np
from pgvector.sqlalchemy import Vector
import time
from sentence_transformers import SentenceTransformer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

logger = logging.getLogger(__name__)

class PDFFileHandler(FileSystemEventHandler):
    """File system event handler for PDF files."""
    
    def __init__(self, processor):
        self.processor = processor
        self.pdf_extensions = {'.pdf'}
    
    def on_created(self, event):
        """Handle file creation events."""
        if not event.is_directory:
            file_path = Path(event.src_path)
            if file_path.suffix.lower() in self.pdf_extensions:
                logger.info(f"New PDF file detected: {file_path}")
                self.processor.auto_process_file(str(file_path))
    
    def on_moved(self, event):
        """Handle file move events (like file uploads)."""
        if not event.is_directory:
            file_path = Path(event.dest_path)
            if file_path.suffix.lower() in self.pdf_extensions:
                logger.info(f"PDF file moved/uploaded: {file_path}")
                self.processor.auto_process_file(str(file_path))


class PDFProcessorSeparateTables:
    def __init__(self, db_config: Optional[Dict[str, str]] = None, 
                 embedding_model_name: str = "Snowflake/snowflake-arctic-embed-l",
                 watch_directory: str = None):
        """
        Initialize PDF processor with Snowflake Arctic Embed model for separate vector tables per PDF.
        
        Args:
            db_config: Database configuration. If None, uses defaults.
            embedding_model_name: Snowflake embedding model to use
            watch_directory: Directory to watch for new PDF files
        """
        self.db_config = db_config or self._get_default_db_config()
        self.engine = self._create_engine()
        self.metadata = MetaData()
        self.watch_directory = watch_directory
        self.observer = None
        
        # Initialize Snowflake Arctic Embed model
        logger.info(f"Loading Snowflake embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_model_name = embedding_model_name
        
        # Get embedding dimensions from the model
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")
        
        # Initialize text processing tools
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
        # Create database if it doesn't exist
        self._ensure_database_exists()
        
        # Enable pgvector extension and initialize database tables
        self._enable_pgvector()
        self._create_documents_table()
        
    def _get_default_db_config(self) -> Dict[str, str]:
        """Get default PostgreSQL configuration."""
        return {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': os.getenv('POSTGRES_PORT', '5432'),
            'database': os.getenv('POSTGRES_DB', 'ceo_rag_db'),
            'username': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', '')
        }
    
    def _ensure_database_exists(self):
        """Create database if it doesn't exist."""
        try:
            temp_config = self.db_config.copy()
            temp_config['database'] = 'postgres'
            
            temp_engine = self._create_engine_from_config(temp_config)
            
            with temp_engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT 1 FROM pg_database WHERE datname = :db_name
                """), {"db_name": self.db_config['database']})
                
                if not result.fetchone():
                    conn.execute(text("COMMIT"))
                    conn.execute(text(f"CREATE DATABASE {self.db_config['database']}"))
                    logger.info(f"Created database: {self.db_config['database']}")
            
            temp_engine.dispose()
            
        except Exception as e:
            logger.warning(f"Could not create database (may already exist): {str(e)}")
    
    def _enable_pgvector(self):
        """Enable pgvector extension in the database."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.commit()
                logger.info("pgvector extension enabled")
        except Exception as e:
            logger.error(f"Could not enable pgvector extension: {str(e)}")
            raise
    
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
        Convert filename to valid PostgreSQL table name for vectors.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized table name with 'vectors_' prefix
        """
        # Remove file extension and convert to lowercase
        name = Path(filename).stem.lower()
        
        # Replace special characters with underscores
        name = re.sub(r'[^a-z0-9_]', '_', name)
        
        # Ensure it starts with a letter or underscore
        if name[0].isdigit():
            name = f"doc_{name}"
        
        # Add vectors prefix
        table_name = f"vectors_{name}"
        
        # Limit length to 63 characters (PostgreSQL limit)
        if len(table_name) > 63:
            # Create hash suffix to ensure uniqueness
            hash_suffix = hashlib.md5(filename.encode()).hexdigest()[:8]
            table_name = f"vectors_{name[:46]}_{hash_suffix}"
        
        return table_name
    
    def _create_documents_table(self):
        """Create table to store document metadata and vector table references."""
        create_sql = """
        CREATE TABLE IF NOT EXISTS pdf_documents (
            id SERIAL PRIMARY KEY,
            file_name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_hash TEXT UNIQUE NOT NULL,
            file_size BIGINT,
            total_pages INTEGER,
            extraction_method TEXT,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata JSONB,
            processing_status TEXT DEFAULT 'processing',
            error_message TEXT,
            embedding_model TEXT,
            total_chunks INTEGER DEFAULT 0,
            total_tokens INTEGER DEFAULT 0,
            vector_table_name TEXT NOT NULL,
            vector_table_created BOOLEAN DEFAULT FALSE
        )
        """
        
        with self.engine.connect() as conn:
            conn.execute(text(create_sql))
            conn.commit()
    
    def _create_vector_table_for_pdf(self, vector_table_name: str, document_id: int) -> bool:
        """
        Create a separate vector table for a specific PDF.
        
        Args:
            vector_table_name: Name of the vector table to create
            document_id: Document ID for reference
            
        Returns:
            True if successful, False otherwise
        """
        try:
            create_sql = f"""
            CREATE TABLE IF NOT EXISTS "{vector_table_name}" (
                id SERIAL PRIMARY KEY,
                document_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                chunk_tokens INTEGER,
                chunk_hash TEXT NOT NULL,
                page_number INTEGER,
                chunk_type TEXT DEFAULT 'content',
                metadata JSONB,
                embedding vector({self.embedding_dim}),
                embedding_model TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(document_id, chunk_hash)
            );
            
            -- Create index for vector similarity search
            CREATE INDEX IF NOT EXISTS "{vector_table_name}_embedding_idx" 
            ON "{vector_table_name}" USING ivfflat (embedding vector_cosine_ops) 
            WITH (lists = 100);
            
            -- Create other useful indexes
            CREATE INDEX IF NOT EXISTS "{vector_table_name}_document_id_idx" 
            ON "{vector_table_name}" (document_id);
            
            CREATE INDEX IF NOT EXISTS "{vector_table_name}_page_number_idx" 
            ON "{vector_table_name}" (page_number);
            
            CREATE INDEX IF NOT EXISTS "{vector_table_name}_chunk_index_idx" 
            ON "{vector_table_name}" (chunk_index);
            """
            
            with self.engine.connect() as conn:
                conn.execute(text(create_sql))
                conn.commit()
                
            logger.info(f"Created vector table: {vector_table_name}")
            
            # Update document record to mark vector table as created
            with self.engine.connect() as conn:
                conn.execute(text("""
                    UPDATE pdf_documents 
                    SET vector_table_created = TRUE 
                    WHERE id = :doc_id
                """), {"doc_id": document_id})
                conn.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating vector table {vector_table_name}: {str(e)}")
            return False
    
    def _get_file_hash(self, file_path: str) -> str:
        """Generate hash for file to detect duplicates."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _document_exists(self, file_hash: str) -> Optional[Tuple[int, str]]:
        """Check if document already exists in database."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT id, vector_table_name FROM pdf_documents 
                    WHERE file_hash = :file_hash AND processing_status = 'completed'
                """), {"file_hash": file_hash})
                
                row = result.fetchone()
                return (row[0], row[1]) if row else None
        except Exception as e:
            logger.error(f"Error checking document existence: {str(e)}")
            return None
    
    def _extract_text_pymupdf(self, file_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Extract text using PyMuPDF (most reliable for complex PDFs)."""
        pages_data = []
        metadata = {}
        
        try:
            doc = fitz.open(file_path)
            metadata = {
                'total_pages': doc.page_count,
                'title': doc.metadata.get('title', ''),
                'author': doc.metadata.get('author', ''),
                'subject': doc.metadata.get('subject', ''),
                'creator': doc.metadata.get('creator', ''),
                'producer': doc.metadata.get('producer', ''),
                'creation_date': doc.metadata.get('creationDate', ''),
                'modification_date': doc.metadata.get('modDate', ''),
            }
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                text = page.get_text()
                
                page_data = {
                    'page_number': page_num + 1,
                    'text': text,
                    'char_count': len(text),
                    'word_count': len(text.split()) if text else 0,
                }
                
                # Try to extract table data if present
                try:
                    tables = page.find_tables()
                    if tables:
                        page_data['has_tables'] = True
                        page_data['table_count'] = len(tables)
                except:
                    pass
                
                pages_data.append(page_data)
            
            doc.close()
            
        except Exception as e:
            logger.error(f"PyMuPDF extraction failed: {str(e)}")
            raise
        
        return pages_data, metadata
    
    def _extract_text_pdfplumber(self, file_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Extract text using pdfplumber (good for tables and layout)."""
        pages_data = []
        metadata = {}
        
        try:
            with pdfplumber.open(file_path) as pdf:
                metadata = {
                    'total_pages': len(pdf.pages),
                    'title': pdf.metadata.get('Title', ''),
                    'author': pdf.metadata.get('Author', ''),
                    'subject': pdf.metadata.get('Subject', ''),
                    'creator': pdf.metadata.get('Creator', ''),
                    'producer': pdf.metadata.get('Producer', ''),
                    'creation_date': pdf.metadata.get('CreationDate', ''),
                }
                
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    tables = page.extract_tables()
                    table_text = ""
                    
                    if tables:
                        for table in tables:
                            for row in table:
                                if row:
                                    table_text += " | ".join([str(cell) if cell else "" for cell in row]) + "\n"
                    
                    combined_text = text
                    if table_text:
                        combined_text += "\n\nTABLE DATA:\n" + table_text
                    
                    page_data = {
                        'page_number': page_num + 1,
                        'text': combined_text,
                        'raw_text': text,
                        'table_text': table_text,
                        'char_count': len(combined_text),
                        'word_count': len(combined_text.split()) if combined_text else 0,
                        'has_tables': len(tables) > 0,
                        'table_count': len(tables),
                    }
                    
                    pages_data.append(page_data)
                    
        except Exception as e:
            logger.error(f"pdfplumber extraction failed: {str(e)}")
            raise
        
        return pages_data, metadata
    
    def _extract_text_pypdf2(self, file_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Extract text using PyPDF2 (fallback method)."""
        pages_data = []
        metadata = {}
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                if pdf_reader.metadata:
                    metadata = {
                        'total_pages': len(pdf_reader.pages),
                        'title': pdf_reader.metadata.get('/Title', ''),
                        'author': pdf_reader.metadata.get('/Author', ''),
                        'subject': pdf_reader.metadata.get('/Subject', ''),
                        'creator': pdf_reader.metadata.get('/Creator', ''),
                        'producer': pdf_reader.metadata.get('/Producer', ''),
                    }
                else:
                    metadata = {'total_pages': len(pdf_reader.pages)}
                
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    
                    page_data = {
                        'page_number': page_num + 1,
                        'text': text,
                        'char_count': len(text),
                        'word_count': len(text.split()) if text else 0,
                    }
                    
                    pages_data.append(page_data)
                    
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed: {str(e)}")
            raise
        
        return pages_data, metadata
    
    def extract_text_from_pdf(self, file_path: str, method: str = 'auto') -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Extract text from PDF using specified or best available method."""
        if method == 'auto':
            methods = ['pdfplumber', 'pymupdf', 'pypdf2']
        else:
            methods = [method]
        
        last_error = None
        
        for extraction_method in methods:
            try:
                if extraction_method == 'pymupdf':
                    pages_data, metadata = self._extract_text_pymupdf(file_path)
                elif extraction_method == 'pdfplumber':
                    pages_data, metadata = self._extract_text_pdfplumber(file_path)
                elif extraction_method == 'pypdf2':
                    pages_data, metadata = self._extract_text_pypdf2(file_path)
                else:
                    raise ValueError(f"Unknown extraction method: {extraction_method}")
                
                total_text = sum(len(page['text']) for page in pages_data)
                if total_text > 0:
                    logger.info(f"Successfully extracted text using {extraction_method}: {total_text} characters")
                    return pages_data, metadata, extraction_method
                else:
                    logger.warning(f"No text extracted using {extraction_method}")
                    
            except Exception as e:
                logger.warning(f"Failed to extract using {extraction_method}: {str(e)}")
                last_error = e
                continue
        
        raise Exception(f"All extraction methods failed. Last error: {last_error}")
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        if not text:
            return ""
        
        # Remove excessive whitespace and normalize line breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n ', '\n', text)
        
        # Remove common PDF artifacts
        text = re.sub(r'\f', '\n', text)
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)
        
        # Fix common OCR errors and formatting issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
        text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
        
        # Fix hyphenated words split across lines
        text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
        
        # Clean up extra spaces
        text = re.sub(r' +', ' ', text)
        text = text.strip()
        
        return text
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken encoding."""
        return len(self.encoding.encode(text))
    
    def _chunk_text_by_sentences(self, text: str, max_tokens: int = 500, overlap_tokens: int = 50) -> List[str]:
        """Chunk text by sentences with token limits."""
        if not text.strip():
            return []
        
        sentences = sent_tokenize(text)
        if not sentences:
            return [text]
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)
            
            if sentence_tokens > max_tokens:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                    current_tokens = 0
                
                words = sentence.split()
                temp_chunk = ""
                temp_tokens = 0
                
                for word in words:
                    word_tokens = self._count_tokens(word + " ")
                    if temp_tokens + word_tokens > max_tokens and temp_chunk:
                        chunks.append(temp_chunk.strip())
                        temp_chunk = word + " "
                        temp_tokens = word_tokens
                    else:
                        temp_chunk += word + " "
                        temp_tokens += word_tokens
                
                if temp_chunk.strip():
                    current_chunk = temp_chunk
                    current_tokens = temp_tokens
                    
            else:
                if current_tokens + sentence_tokens > max_tokens and current_chunk:
                    chunks.append(current_chunk.strip())
                    
                    if overlap_tokens > 0 and chunks:
                        prev_words = current_chunk.split()
                        overlap_text = ""
                        overlap_count = 0
                        
                        for word in reversed(prev_words):
                            word_tokens = self._count_tokens(word + " ")
                            if overlap_count + word_tokens <= overlap_tokens:
                                overlap_text = word + " " + overlap_text
                                overlap_count += word_tokens
                            else:
                                break
                        
                        current_chunk = overlap_text + sentence + " "
                        current_tokens = self._count_tokens(current_chunk)
                    else:
                        current_chunk = sentence + " "
                        current_tokens = sentence_tokens
                else:
                    current_chunk += sentence + " "
                    current_tokens += sentence_tokens
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if chunk.strip()]
    
    def _chunk_text_by_paragraphs(self, text: str, max_tokens: int = 500, overlap_tokens: int = 50) -> List[str]:
        """Chunk text by paragraphs with fallback to sentences."""
        if not text.strip():
            return []
        
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        
        if not paragraphs:
            return self._chunk_text_by_sentences(text, max_tokens, overlap_tokens)
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for paragraph in paragraphs:
            paragraph_tokens = self._count_tokens(paragraph)
            
            if paragraph_tokens > max_tokens:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                    current_tokens = 0
                
                para_chunks = self._chunk_text_by_sentences(paragraph, max_tokens, overlap_tokens)
                chunks.extend(para_chunks)
                
            else:
                if current_tokens + paragraph_tokens > max_tokens and current_chunk:
                    chunks.append(current_chunk.strip())
                    
                    if overlap_tokens > 0:
                        prev_sentences = sent_tokenize(current_chunk)
                        if prev_sentences:
                            last_sentence = prev_sentences[-1]
                            if self._count_tokens(last_sentence) <= overlap_tokens:
                                current_chunk = last_sentence + "\n\n" + paragraph
                                current_tokens = self._count_tokens(current_chunk)
                            else:
                                current_chunk = paragraph
                                current_tokens = paragraph_tokens
                        else:
                            current_chunk = paragraph
                            current_tokens = paragraph_tokens
                    else:
                        current_chunk = paragraph
                        current_tokens = paragraph_tokens
                else:
                    if current_chunk:
                        current_chunk += "\n\n" + paragraph
                    else:
                        current_chunk = paragraph
                    current_tokens += paragraph_tokens
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if chunk.strip()]
    
    def chunk_document_text(self, pages_data: List[Dict[str, Any]], 
                           chunking_strategy: str = 'paragraph',
                           max_tokens: int = 500,
                           overlap_tokens: int = 50) -> List[Dict[str, Any]]:
        """Chunk document text into smaller pieces suitable for embedding."""
        all_chunks = []
        
        if chunking_strategy == 'page':
            for page_data in pages_data:
                clean_text = self._clean_text(page_data['text'])
                if clean_text.strip():
                    chunk = {
                        'text': clean_text,
                        'tokens': self._count_tokens(clean_text),
                        'page_number': page_data['page_number'],
                        'chunk_type': 'page',
                        'metadata': {
                            'original_char_count': page_data['char_count'],
                            'original_word_count': page_data['word_count'],
                            'has_tables': page_data.get('has_tables', False),
                            'table_count': page_data.get('table_count', 0),
                        }
                    }
                    all_chunks.append(chunk)
        
        else:
            full_text = ""
            page_boundaries = {}
            current_pos = 0
            
            for page_data in pages_data:
                clean_text = self._clean_text(page_data['text'])
                page_boundaries[page_data['page_number']] = (current_pos, current_pos + len(clean_text))
                full_text += clean_text + "\n\n"
                current_pos = len(full_text)
            
            if chunking_strategy == 'paragraph':
                text_chunks = self._chunk_text_by_paragraphs(full_text, max_tokens, overlap_tokens)
            else:
                text_chunks = self._chunk_text_by_sentences(full_text, max_tokens, overlap_tokens)
            
            for i, chunk_text in enumerate(text_chunks):
                chunk_start = full_text.find(chunk_text[:50])
                primary_page = 1
                
                for page_num, (start, end) in page_boundaries.items():
                    if start <= chunk_start < end:
                        primary_page = page_num
                        break
                
                chunk = {
                    'text': chunk_text,
                    'tokens': self._count_tokens(chunk_text),
                    'page_number': primary_page,
                    'chunk_type': chunking_strategy,
                    'metadata': {
                        'chunk_index': i,
                        'chunking_strategy': chunking_strategy,
                        'max_tokens': max_tokens,
                        'overlap_tokens': overlap_tokens,
                    }
                }
                all_chunks.append(chunk)
        
        return all_chunks
    
    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for text using Snowflake Arctic Embed model.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            List of floats representing the embedding, or None if failed
        """
        try:
            # For document chunks, don't use any special prompt
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return None
    
    def _batch_generate_embeddings(self, texts: List[str], batch_size: int = 32) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts in batches using Snowflake Arctic Embed.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process at once
            
        Returns:
            List of embeddings (or None for failed embeddings)
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.info(f"Generating embeddings for batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
            try:
                # Generate embeddings for the batch
                batch_embeddings = self.embedding_model.encode(
                    batch, 
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
                
                # Convert numpy arrays to lists
                for emb in batch_embeddings:
                    embeddings.append(emb.tolist())
                
            except Exception as e:
                logger.error(f"Error generating batch embeddings: {str(e)}")
                embeddings.extend([None] * len(batch))
        
        return embeddings
    
    def _insert_document_record(self, file_path: str, file_hash: str, metadata: Dict[str, Any], 
                               extraction_method: str, vector_table_name: str) -> int:
        """Insert document record and return document ID."""
        file_path_obj = Path(file_path)
        
        insert_sql = """
        INSERT INTO pdf_documents (file_name, file_path, file_hash, file_size, 
                                 total_pages, extraction_method, metadata, processing_status, 
                                 embedding_model, vector_table_name)
        VALUES (:file_name, :file_path, :file_hash, :file_size, 
                :total_pages, :extraction_method, :metadata, 'processing', 
                :embedding_model, :vector_table_name)
        RETURNING id
        """
        
        with self.engine.connect() as conn:
            result = conn.execute(text(insert_sql), {
                'file_name': file_path_obj.name,
                'file_path': str(file_path_obj.absolute()),
                'file_hash': file_hash,
                'file_size': file_path_obj.stat().st_size,
                'total_pages': metadata.get('total_pages', 0),
                'extraction_method': extraction_method,
                'metadata': json.dumps(metadata),
                'embedding_model': self.embedding_model_name,
                'vector_table_name': vector_table_name
            })
            doc_id = result.fetchone()[0]
            conn.commit()
            return doc_id
    
    def _insert_chunks_to_vector_table(self, vector_table_name: str, document_id: int, chunks: List[Dict[str, Any]]):
        """Insert chunks with embeddings into the specific vector table."""
        if not chunks:
            return
        
        # Generate embeddings for all chunks
        texts = [chunk['text'] for chunk in chunks]
        logger.info(f"Generating embeddings for {len(texts)} chunks using Snowflake Arctic Embed...")
        embeddings = self._batch_generate_embeddings(texts)
        
        # Prepare chunk records
        successful_chunks = 0
        
        insert_sql = f"""
        INSERT INTO "{vector_table_name}" (document_id, chunk_index, chunk_text, chunk_tokens, 
                                          chunk_hash, page_number, chunk_type, metadata, embedding, embedding_model)
        VALUES (:document_id, :chunk_index, :chunk_text, :chunk_tokens, 
                :chunk_hash, :page_number, :chunk_type, :metadata, :embedding, :embedding_model)
        """
        
        with self.engine.connect() as conn:
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                try:
                    chunk_hash = hashlib.md5(chunk['text'].encode()).hexdigest()
                    
                    chunk_record = {
                        'document_id': document_id,
                        'chunk_index': i,
                        'chunk_text': chunk['text'],
                        'chunk_tokens': chunk['tokens'],
                        'chunk_hash': chunk_hash,
                        'page_number': chunk['page_number'],
                        'chunk_type': chunk['chunk_type'],
                        'metadata': json.dumps(chunk['metadata']),
                        'embedding': embedding,
                        'embedding_model': self.embedding_model_name if embedding else None
                    }
                    
                    conn.execute(text(insert_sql), chunk_record)
                    
                    if embedding is not None:
                        successful_chunks += 1
                        
                except Exception as e:
                    logger.error(f"Error inserting chunk {i}: {str(e)}")
            
            conn.commit()
        
        logger.info(f"Inserted {len(chunks)} chunks to {vector_table_name}, {successful_chunks} with embeddings")
    
    def _update_document_statistics(self, document_id: int, total_chunks: int, total_tokens: int):
        """Update document with final statistics."""
        update_sql = """
        UPDATE pdf_documents 
        SET total_chunks = :total_chunks, total_tokens = :total_tokens
        WHERE id = :document_id
        """
        
        with self.engine.connect() as conn:
            conn.execute(text(update_sql), {
                'document_id': document_id,
                'total_chunks': total_chunks,
                'total_tokens': total_tokens
            })
            conn.commit()
    
    def _update_document_status(self, document_id: int, status: str, error_message: str = None):
        """Update document processing status."""
        update_sql = """
        UPDATE pdf_documents 
        SET processing_status = :status, error_message = :error_message
        WHERE id = :document_id
        """
        
        with self.engine.connect() as conn:
            conn.execute(text(update_sql), {
                'document_id': document_id,
                'status': status,
                'error_message': error_message
            })
            conn.commit()
    
    def auto_process_file(self, file_path: str) -> Dict[str, Any]:
        """Automatically process PDF file only if not already processed."""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                return {
                    'success': False,
                    'file_path': str(file_path),
                    'error': 'File does not exist'
                }
            
            # Check if already processed
            file_hash = self._get_file_hash(str(file_path))
            existing = self._document_exists(file_hash)
            
            if existing:
                document_id, vector_table_name = existing
                logger.info(f"Document already processed: {file_path}")
                return {
                    'success': True,
                    'file_path': str(file_path),
                    'action': 'skipped',
                    'reason': 'already_processed',
                    'document_id': document_id,
                    'vector_table_name': vector_table_name
                }
            
            # Process the file
            logger.info(f"Processing PDF file: {file_path}")
            return self.process_pdf_file(str(file_path))
            
        except Exception as e:
            logger.error(f"Error in auto-processing {file_path}: {str(e)}")
            return {
                'success': False,
                'file_path': str(file_path),
                'error': str(e)
            }
    
    def process_pdf_file(self, file_path: str, 
                        extraction_method: str = 'auto',
                        chunking_strategy: str = 'paragraph',
                        max_tokens: int = 500,
                        overlap_tokens: int = 50,
                        generate_embeddings: bool = True) -> Dict[str, Any]:
        """Process PDF file and store chunks with embeddings in separate table."""
        document_id = None
        vector_table_name = None
        
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"PDF file not found: {file_path}")
            
            # Generate file hash and vector table name
            file_hash = self._get_file_hash(str(file_path))
            vector_table_name = self._sanitize_table_name(file_path.name)
            
            # Check if already processed
            existing = self._document_exists(file_hash)
            if existing:
                document_id, existing_vector_table = existing
                return {
                    'success': True,
                    'file_path': str(file_path),
                    'action': 'skipped',
                    'reason': 'already_processed',
                    'document_id': document_id,
                    'vector_table_name': existing_vector_table
                }
            
            # Extract text from PDF
            logger.info(f"Extracting text from: {file_path}")
            pages_data, metadata, used_method = self.extract_text_from_pdf(
                str(file_path), extraction_method
            )
            
            # Validate extraction
            total_chars = sum(len(page['text']) for page in pages_data)
            if total_chars == 0:
                raise Exception("No text could be extracted from PDF")
            
            # Insert document record
            document_id = self._insert_document_record(
                str(file_path), file_hash, metadata, used_method, vector_table_name
            )
            
            # Create vector table for this PDF
            logger.info(f"Creating vector table: {vector_table_name}")
            if not self._create_vector_table_for_pdf(vector_table_name, document_id):
                raise Exception(f"Failed to create vector table: {vector_table_name}")
            
            # Chunk the text
            logger.info(f"Chunking text using strategy: {chunking_strategy}")
            chunks = self.chunk_document_text(
                pages_data, chunking_strategy, max_tokens, overlap_tokens
            )
            
            if not chunks:
                raise Exception("No chunks generated from document")
            
            # Insert chunks into the specific vector table
            if generate_embeddings:
                logger.info(f"Processing {len(chunks)} chunks with embeddings into table: {vector_table_name}")
                self._insert_chunks_to_vector_table(vector_table_name, document_id, chunks)
            else:
                logger.info(f"Processing {len(chunks)} chunks without embeddings into table: {vector_table_name}")
                # Insert without embeddings
                for chunk in chunks:
                    chunk_hash = hashlib.md5(chunk['text'].encode()).hexdigest()
                    
                    insert_sql = f"""
                    INSERT INTO "{vector_table_name}" (document_id, chunk_index, chunk_text, chunk_tokens, 
                                                      chunk_hash, page_number, chunk_type, metadata)
                    VALUES (:document_id, :chunk_index, :chunk_text, :chunk_tokens, 
                            :chunk_hash, :page_number, :chunk_type, :metadata)
                    """
                    
                    with self.engine.connect() as conn:
                        conn.execute(text(insert_sql), {
                            'document_id': document_id,
                            'chunk_index': chunks.index(chunk),
                            'chunk_text': chunk['text'],
                            'chunk_tokens': chunk['tokens'],
                            'chunk_hash': chunk_hash,
                            'page_number': chunk['page_number'],
                            'chunk_type': chunk['chunk_type'],
                            'metadata': json.dumps(chunk['metadata'])
                        })
                        conn.commit()
            
            # Update document statistics
            total_tokens = sum(chunk['tokens'] for chunk in chunks)
            self._update_document_statistics(document_id, len(chunks), total_tokens)
            
            # Update status to completed
            self._update_document_status(document_id, 'completed')
            
            # Calculate statistics
            avg_tokens = total_tokens / len(chunks) if chunks else 0
            
            result = {
                'success': True,
                'file_path': str(file_path),
                'action': 'processed',
                'document_id': document_id,
                'vector_table_name': vector_table_name,
                'extraction_method': used_method,
                'chunking_strategy': chunking_strategy,
                'embeddings_generated': generate_embeddings,
                'embedding_model': self.embedding_model_name,
                'statistics': {
                    'total_pages': len(pages_data),
                    'total_characters': total_chars,
                    'total_chunks': len(chunks),
                    'total_tokens': total_tokens,
                    'avg_tokens_per_chunk': round(avg_tokens, 2),
                    'max_tokens_setting': max_tokens,
                    'overlap_tokens_setting': overlap_tokens,
                }
            }
            
            logger.info(f"Successfully processed {file_path} into table {vector_table_name}: "
                       f"{len(chunks)} chunks, {total_tokens} tokens")
            
            return result
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error processing PDF {file_path}: {error_msg}")
            
            # Update document status if we created a record
            if document_id:
                self._update_document_status(document_id, 'failed', error_msg)
            
            return {
                'success': False,
                'file_path': str(file_path),
                'error': error_msg,
                'document_id': document_id,
                'vector_table_name': vector_table_name
            }
    
    def search_similar_chunks(self, query_text: str, 
                             vector_table_name: Optional[str] = None,
                             document_id: Optional[int] = None,
                             limit: int = 10, 
                             similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using vector similarity in specific table or all tables.
        Uses Snowflake Arctic Embed model for query encoding.
        
        Args:
            query_text: Text to search for
            vector_table_name: Specific vector table to search (if None, searches all)
            document_id: Specific document ID to search
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of similar chunks with similarity scores
        """
        try:
            # Generate embedding for query using the "query" prompt for better search
            logger.info(f"Generating query embedding using Snowflake Arctic Embed...")
            query_embedding = self.embedding_model.encode(
                query_text, 
                prompt_name="query",
                convert_to_numpy=True
            ).tolist()
            
            if not query_embedding:
                logger.error("Failed to generate embedding for query")
                return []
            
            # Determine which tables to search
            tables_to_search = []
            
            if vector_table_name:
                # Search specific table
                tables_to_search.append(vector_table_name)
            elif document_id:
                # Get table name for specific document
                with self.engine.connect() as conn:
                    result = conn.execute(text("""
                        SELECT vector_table_name FROM pdf_documents 
                        WHERE id = :doc_id AND processing_status = 'completed'
                    """), {"doc_id": document_id})
                    
                    row = result.fetchone()
                    if row:
                        tables_to_search.append(row.vector_table_name)
            else:
                # Search all vector tables
                with self.engine.connect() as conn:
                    result = conn.execute(text("""
                        SELECT DISTINCT vector_table_name FROM pdf_documents 
                        WHERE processing_status = 'completed' AND vector_table_created = TRUE
                    """))
                    
                    tables_to_search = [row.vector_table_name for row in result]
            
            if not tables_to_search:
                logger.warning("No vector tables found to search")
                return []
            
            all_results = []
            
            # Search each table
            for table_name in tables_to_search:
                try:
                    search_sql = f"""
                    SELECT v.id, v.chunk_text, v.chunk_tokens, v.page_number, v.chunk_type,
                           v.metadata, d.file_name, d.file_path, d.vector_table_name,
                           1 - (v.embedding <=> :query_embedding::vector) AS similarity_score
                    FROM "{table_name}" v
                    JOIN pdf_documents d ON v.document_id = d.id
                    WHERE v.embedding IS NOT NULL
                    AND 1 - (v.embedding <=> :query_embedding::vector) >= :similarity_threshold
                    ORDER BY v.embedding <=> :query_embedding::vector
                    LIMIT :limit
                    """
                    
                    with self.engine.connect() as conn:
                        result = conn.execute(text(search_sql), {
                            'query_embedding': query_embedding,
                            'similarity_threshold': similarity_threshold,
                            'limit': limit
                        })
                        
                        for row in result:
                            chunk_metadata = json.loads(row.metadata) if row.metadata else {}
                            
                            chunk = {
                                'id': row.id,
                                'text': row.chunk_text,
                                'tokens': row.chunk_tokens,
                                'page_number': row.page_number,
                                'chunk_type': row.chunk_type,
                                'file_name': row.file_name,
                                'file_path': row.file_path,
                                'vector_table_name': row.vector_table_name,
                                'similarity_score': float(row.similarity_score),
                                'metadata': chunk_metadata
                            }
                            all_results.append(chunk)
                
                except Exception as e:
                    logger.error(f"Error searching table {table_name}: {str(e)}")
                    continue
            
            # Sort all results by similarity score and limit
            all_results.sort(key=lambda x: x['similarity_score'], reverse=True)
            final_results = all_results[:limit]
            
            logger.info(f"Found {len(final_results)} similar chunks across {len(tables_to_search)} tables")
            return final_results
                
        except Exception as e:
            logger.error(f"Error searching similar chunks: {str(e)}")
            return []
    
    def get_vector_table_info(self, vector_table_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific vector table."""
        try:
            with self.engine.connect() as conn:
                # Check if table exists
                result = conn.execute(text("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = :table_name
                    )
                """), {"table_name": vector_table_name})
                
                if not result.scalar():
                    return None
                
                # Get statistics
                stats_result = conn.execute(text(f"""
                    SELECT COUNT(*) as total_chunks,
                           COUNT(embedding) as chunks_with_embeddings,
                           AVG(chunk_tokens) as avg_tokens,
                           MIN(page_number) as first_page,
                           MAX(page_number) as last_page
                    FROM "{vector_table_name}"
                """))
                
                stats = stats_result.fetchone()
                
                # Get document info
                doc_result = conn.execute(text("""
                    SELECT file_name, file_path, total_pages
                    FROM pdf_documents 
                    WHERE vector_table_name = :table_name
                """), {"table_name": vector_table_name})
                
                doc_info = doc_result.fetchone()
                
                return {
                    'table_name': vector_table_name,
                    'total_chunks': stats.total_chunks,
                    'chunks_with_embeddings': stats.chunks_with_embeddings,
                    'embedding_coverage': (stats.chunks_with_embeddings / stats.total_chunks * 100) if stats.total_chunks > 0 else 0,
                    'avg_tokens': round(stats.avg_tokens, 2) if stats.avg_tokens else 0,
                    'page_range': f"{stats.first_page}-{stats.last_page}" if stats.first_page else "",
                    'file_name': doc_info.file_name if doc_info else "",
                    'file_path': doc_info.file_path if doc_info else "",
                    'total_pages': doc_info.total_pages if doc_info else 0
                }
                
        except Exception as e:
            logger.error(f"Error getting vector table info for {vector_table_name}: {str(e)}")
            return None
    
    def list_vector_tables(self) -> List[Dict[str, Any]]:
        """List all vector tables and their information."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT vector_table_name, file_name, total_chunks, total_tokens, 
                           processing_status, processed_at, embedding_model
                    FROM pdf_documents 
                    WHERE vector_table_created = TRUE
                    ORDER BY processed_at DESC
                """))
                
                tables = []
                for row in result:
                    table_info = self.get_vector_table_info(row.vector_table_name)
                    if table_info:
                        table_info.update({
                            'processing_status': row.processing_status,
                            'processed_at': row.processed_at,
                            'total_tokens': row.total_tokens,
                            'embedding_model': row.embedding_model
                        })
                        tables.append(table_info)
                
                return tables
                
        except Exception as e:
            logger.error(f"Error listing vector tables: {str(e)}")
            return []
    
    def delete_pdf_and_vectors(self, document_id: int) -> bool:
        """Delete document and its dedicated vector table."""
        try:
            # Get vector table name
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT vector_table_name FROM pdf_documents WHERE id = :doc_id
                """), {"doc_id": document_id})
                
                row = result.fetchone()
                if not row:
                    logger.warning(f"Document {document_id} not found")
                    return False
                
                vector_table_name = row.vector_table_name
                
                # Drop the vector table
                conn.execute(text(f'DROP TABLE IF EXISTS "{vector_table_name}" CASCADE'))
                
                # Delete document record
                conn.execute(text("DELETE FROM pdf_documents WHERE id = :doc_id"), 
                           {"doc_id": document_id})
                
                conn.commit()
                
                logger.info(f"Deleted document {document_id} and vector table {vector_table_name}")
                return True
                
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            return False
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all processed documents with their vector table information."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT id, file_name, file_path, total_pages, extraction_method, 
                           processed_at, processing_status, metadata, total_chunks, 
                           total_tokens, embedding_model, vector_table_name, vector_table_created
                    FROM pdf_documents
                    ORDER BY processed_at DESC
                """))
                
                documents = []
                for row in result:
                    metadata = json.loads(row.metadata) if row.metadata else {}
                    
                    doc = {
                        'id': row.id,
                        'file_name': row.file_name,
                        'file_path': row.file_path,
                        'total_pages': row.total_pages,
                        'extraction_method': row.extraction_method,
                        'processed_at': row.processed_at,
                        'processing_status': row.processing_status,
                        'total_chunks': row.total_chunks or 0,
                        'total_tokens': row.total_tokens or 0,
                        'embedding_model': row.embedding_model,
                        'vector_table_name': row.vector_table_name,
                        'vector_table_created': row.vector_table_created,
                        'metadata': metadata
                    }
                    documents.append(doc)
                
                return documents
                
        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}")
            return []
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get overall processing statistics including vector table stats."""
        try:
            with self.engine.connect() as conn:
                # Document statistics
                doc_stats = conn.execute(text("""
                    SELECT processing_status, COUNT(*) as count
                    FROM pdf_documents
                    GROUP BY processing_status
                """)).fetchall()
                
                # Vector table statistics
                vector_stats = conn.execute(text("""
                    SELECT COUNT(*) as total_vector_tables,
                           COUNT(CASE WHEN vector_table_created = TRUE THEN 1 END) as created_tables
                    FROM pdf_documents
                    WHERE processing_status = 'completed'
                """)).fetchone()
                
                # Get sample of actual chunk counts from vector tables
                table_names = conn.execute(text("""
                    SELECT vector_table_name FROM pdf_documents 
                    WHERE vector_table_created = TRUE AND processing_status = 'completed'
                    LIMIT 10
                """)).fetchall()
                
                total_chunks_actual = 0
                total_embeddings_actual = 0
                
                for table_row in table_names:
                    try:
                        chunk_count = conn.execute(text(f"""
                            SELECT COUNT(*) as chunks, COUNT(embedding) as embeddings
                            FROM "{table_row.vector_table_name}"
                        """)).fetchone()
                        
                        total_chunks_actual += chunk_count.chunks
                        total_embeddings_actual += chunk_count.embeddings
                    except:
                        continue
                
                return {
                    'document_status_counts': {row.processing_status: row.count for row in doc_stats},
                    'vector_table_statistics': {
                        'total_vector_tables': vector_stats.total_vector_tables or 0,
                        'created_vector_tables': vector_stats.created_tables or 0,
                        'sample_total_chunks': total_chunks_actual,
                        'sample_total_embeddings': total_embeddings_actual,
                        'sample_embedding_coverage': round((total_embeddings_actual / max(total_chunks_actual, 1)) * 100, 2)
                    },
                    'embedding_model': self.embedding_model_name,
                    'embedding_dimension': self.embedding_dim
                }
                
        except Exception as e:
            logger.error(f"Error getting processing stats: {str(e)}")
            return {}
    
    def start_watching(self, directory: str = None):
        """Start watching directory for new PDF files."""
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
        event_handler = PDFFileHandler(self)
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
        """Process all existing PDF files in directory."""
        directory = Path(directory)
        pdf_extensions = {'.pdf'}
        
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in pdf_extensions:
                logger.info(f"Processing existing file: {file_path}")
                self.auto_process_file(str(file_path))
    
    def close(self):
        """Close database connections and stop watching."""
        self.stop_watching()
        if hasattr(self, 'engine'):
            self.engine.dispose()


# Integration functions
def setup_pdf_processor_separate_tables(pdf_directory: str = "data/documents/pdfs/",
                                        embedding_model: str = "Snowflake/snowflake-arctic-embed-l") -> PDFProcessorSeparateTables:
    """
    Set up PDF processor with Snowflake Arctic Embed and automatic file watching.
    
    Args:
        pdf_directory: Directory containing PDF files
        embedding_model: Snowflake embedding model to use
        
    Returns:
        Configured PDFProcessorSeparateTables instance
    """
    # Default PostgreSQL configuration
    db_config = {
        'host': 'localhost',
        'port': '5432',
        'database': 'ceo_rag_db',
        'username': 'postgres',
        'password': 'password'
    }
    
    # Initialize processor
    processor = PDFProcessorSeparateTables(
        db_config=db_config,
        embedding_model_name=embedding_model,
        watch_directory=pdf_directory
    )
    
    # Start watching for new files
    processor.start_watching()
    
    return processor


# Example usage and testing
if __name__ == "__main__":
    import time
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize processor
    print("Initializing PDF processor with Snowflake Arctic Embed...")
    processor = setup_pdf_processor_separate_tables(
        pdf_directory="data/documents/pdfs/",
        embedding_model="Snowflake/snowflake-arctic-embed-l"
    )
    
    try:
        print(f"\nPDF processor with Snowflake Arctic Embed started!")
        print(f"Embedding model: {processor.embedding_model_name}")
        print(f"Embedding dimension: {processor.embedding_dim}")
        print(f"Watching for new files in: {processor.watch_directory}")
        print("\nAvailable commands:")
        print("- Press 's' to show processing stats")
        print("- Press 'l' to list all documents")
        print("- Press 'v' to list vector tables")
        print("- Press 't <query>' to test search")
        print("- Press 'Ctrl+C' to stop")
        
        # Keep the script running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping PDF processor...")
        processor.close()
        print("PDF processor stopped.")
