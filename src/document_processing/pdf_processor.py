#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Rebuild PDF embeddings for ALL PDFs in a directory.

What it does:
- Scans a directory for *.pdf
- For each PDF:
  * Compute md5 hash
  * Upsert pdf_documents by file_hash (idempotent)
  * Drop and recreate its per-PDF vector table
  * Extract text page-by-page with pdfplumber
  * Chunk text and embed with SentenceTransformers (Snowflake Arctic)
  * Insert chunks + embeddings into the vector table
  * Mark processing_status = 'completed' and store vector_rows count

Notes:
- Uses only :param style in SQLAlchemy text() (no mixed paramstyles).
- Uses CAST(:metadata AS JSONB) to bind JSON safely.
- Deletes and rebuilds the vector table every run.
"""

import os
import re
import json
import time
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pdfplumber
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer

# -----------------------------
# Config
# -----------------------------
PDF_DIR = os.getenv("PDF_DIR", "data/documents/pdfs")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "ceo_rag_db")
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "password")

EMBED_MODEL = os.getenv("EMBED_MODEL_NAME", "Snowflake/snowflake-arctic-embed-l")
VECTOR_PREFIX = os.getenv("VECTOR_TABLE_PREFIX", "vectors_doc_")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1400"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("pdf_rebuilder")


# -----------------------------
# Helpers
# -----------------------------
def md5_file(path: Path) -> str:
    """Compute md5 hash for a file."""
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def slugify(text: str) -> str:
    """Make a safe suffix for table names."""
    base = Path(text).stem.lower()
    base = re.sub(r"[^a-z0-9]+", "_", base)
    base = re.sub(r"_+", "_", base).strip("_")
    return base


def chunk_text(txt: str, size: int, overlap: int) -> List[str]:
    """Simple character-based chunks with overlap."""
    txt = (txt or "").replace("\x00", "")
    n = len(txt)
    chunks = []
    start = 0
    while start < n:
        end = min(n, start + size)
        piece = txt[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - overlap, end) if overlap >= size else end - overlap
        if start <= len(chunks):  # safety against bad overlap
            start = end
    return chunks


def extract_pages(pdf_path: Path) -> List[str]:
    """Extract text from PDF pages using pdfplumber."""
    pages: List[str] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for p in pdf.pages:
            pages.append(p.extract_text() or "")
    return pages


def make_engine():
    """Create SQLAlchemy engine for Postgres."""
    pw = f":{DB_PASS}" if DB_PASS else ""
    dsn = f"postgresql://{DB_USER}{pw}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(dsn, echo=False)


def ensure_pgvector(conn):
    """Ensure pgvector extension is present."""
    try:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        logger.info("pgvector extension OK.")
    except Exception as e:
        logger.warning(f"Could not ensure pgvector extension: {e}")


def upsert_pdf(conn,
               *,
               file_name: str,
               file_path: str,
               file_hash: str,
               file_size: int,
               total_pages: int,
               metadata_json: str,
               embedding_model: str,
               vector_table: str) -> int:
    """
    Upsert into pdf_documents keyed by file_hash.
    Uses CAST(:metadata AS JSONB) to bind JSON.
    Returns id.
    """
    sql = text("""
        INSERT INTO pdf_documents
            (file_name, file_path, file_hash, file_size, total_pages,
             extraction_method, metadata, processing_status, embedding_model,
             vector_table_name)
        VALUES
            (:file_name, :file_path, :file_hash, :file_size, :total_pages,
             'pdfplumber', CAST(:metadata AS JSONB), 'processing', :embedding_model,
             :vector_table)
        ON CONFLICT (file_hash) DO UPDATE SET
            file_name = EXCLUDED.file_name,
            file_path = EXCLUDED.file_path,
            file_size = EXCLUDED.file_size,
            total_pages = EXCLUDED.total_pages,
            extraction_method = EXCLUDED.extraction_method,
            metadata = EXCLUDED.metadata,
            processing_status = 'processing',
            embedding_model = EXCLUDED.embedding_model,
            vector_table_name = EXCLUDED.vector_table_name
        RETURNING id
    """)
    row = conn.execute(sql, {
        "file_name": file_name,
        "file_path": file_path,
        "file_hash": file_hash,
        "file_size": file_size,
        "total_pages": total_pages,
        "metadata": metadata_json,
        "embedding_model": embedding_model,
        "vector_table": vector_table
    }).fetchone()
    return int(row[0])


def drop_and_create_vector_table(conn, table_name: str, dim: int):
    """Drop and recreate per-PDF vector table."""
    conn.execute(text(f'DROP TABLE IF EXISTS "{table_name}" CASCADE'))
    conn.execute(text(f"""
        CREATE TABLE "{table_name}" (
            id BIGSERIAL PRIMARY KEY,
            chunk_text TEXT NOT NULL,
            chunk_tokens INTEGER,
            page_number INTEGER,
            chunk_type TEXT,
            metadata JSONB,
            embedding vector({dim}),
            embedding_model TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """))
    logger.info(f'Recreated table "{table_name}" (vector dim {dim}).')


def insert_chunk(conn,
                 table_name: str,
                 *,
                 text_chunk: str,
                 tokens: int,
                 page_no: int,
                 metadata_obj: Dict[str, Any],
                 embedding_str: str,
                 embed_model: str):
    """
    Insert a chunk into the per-PDF vector table.
    Uses CAST(:metadata AS JSONB) and vector literal "[...]"::vector.
    """
    sql = text(f"""
        INSERT INTO "{table_name}"
            (chunk_text, chunk_tokens, page_number, chunk_type, metadata,
             embedding, embedding_model)
        VALUES
            (:chunk_text, :chunk_tokens, :page_number, 'content',
             CAST(:metadata AS JSONB), :embedding::vector, :embedding_model)
    """)
    conn.execute(sql, {
        "chunk_text": text_chunk,
        "chunk_tokens": tokens,
        "page_number": page_no,
        "metadata": json.dumps(metadata_obj or {}),
        "embedding": embedding_str,
        "embedding_model": embed_model
    })


def mark_completed(conn, pdf_id: int, total_chunks: int):
    """Set processing_status and store vector_rows in metadata."""
    sql = text("""
        UPDATE pdf_documents
        SET processing_status = 'completed',
            metadata = COALESCE(metadata, '{}'::jsonb) || jsonb_build_object('vector_rows', :rows)
        WHERE id = :id
    """)
    conn.execute(sql, {"id": pdf_id, "rows": total_chunks})


# -----------------------------
# Main processing
# -----------------------------
def process_pdf(engine, embedder: SentenceTransformer, pdf_path: Path):
    logger.info(f"Processing: {pdf_path.name}")

    # Hash and file size
    file_hash = md5_file(pdf_path)
    file_size = pdf_path.stat().st_size

    # Extract text per page
    pages = extract_pages(pdf_path)
    total_pages = len(pages)

    # Prepare table name and embedding dim
    vector_table = f"{VECTOR_PREFIX}{slugify(pdf_path.name)}"
    embed_dim = embedder.get_sentence_embedding_dimension()

    # Minimal metadata for pdf_documents.metadata
    metadata = {
        "total_pages": total_pages,
        "source": "pdf",
        "file_name": pdf_path.name
    }

    # Upsert pdf_documents and recreate the vector table
    with engine.begin() as conn:
        ensure_pgvector(conn)
        pdf_id = upsert_pdf(
            conn,
            file_name=pdf_path.name,
            file_path=str(pdf_path.resolve()),
            file_hash=file_hash,
            file_size=file_size,
            total_pages=total_pages,
            metadata_json=json.dumps(metadata),
            embedding_model=EMBED_MODEL,
            vector_table=vector_table
        )
        drop_and_create_vector_table(conn, vector_table, embed_dim)

    # Build chunks
    chunks: List[Tuple[int, str]] = []
    for pno, ptxt in enumerate(pages, start=1):
        for ch in chunk_text(ptxt, CHUNK_SIZE, CHUNK_OVERLAP):
            chunks.append((pno, ch))

    logger.info(f"Total chunks: {len(chunks)}")

    # Embed and insert in batches
    inserted = 0
    with engine.begin() as conn:
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            texts = [c[1] for c in batch]
            if not texts:
                continue
            embs = embedder.encode(texts, convert_to_numpy=True)
            for (page_no, text_chunk), emb in zip(batch, embs):
                embedding_str = "[" + ",".join(str(float(x)) for x in emb.tolist()) + "]"
                tokens = len(text_chunk.split())  # simple token approx
                insert_chunk(
                    conn,
                    vector_table,
                    text_chunk=text_chunk,
                    tokens=tokens,
                    page_no=page_no,
                    metadata_obj={"source_type": "pdf_vector", "file": pdf_path.name},
                    embedding=embedding_str,
                    embed_model=EMBED_MODEL
                )
                inserted += 1

        mark_completed(conn, pdf_id, inserted)

    logger.info(f"Completed {pdf_path.name}: inserted {inserted} chunks into {vector_table}.")


def main():
    engine = make_engine()
    logger.info(f"Loading embedding model: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)
    logger.info(f"Embedding dim: {embedder.get_sentence_embedding_dimension()}")

    pdf_dir = Path(PDF_DIR)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(p for p in pdf_dir.glob("*.pdf") if p.is_file())
    if not pdfs:
        logger.info("No PDF files found.")
        return

    for pdf in pdfs:
        try:
            process_pdf(engine, embedder, pdf)
        except Exception as e:
            logger.exception(f"Error processing {pdf}: {e}")


if __name__ == "__main__":
    main()

