#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Rebuild PDF embeddings for ALL PDFs in a given directory.

Steps:
1) Scan directory for *.pdf
2) For each PDF:
   - Compute hash
   - Upsert pdf_documents row using file_hash (status=processing)
   - Drop and recreate its per-PDF vector table
   - Extract text by page
   - Chunk
   - Embed with Snowflake Arctic (SentenceTransformer)
   - Insert chunks + embeddings
   - Mark status=completed and write vector_rows count
"""

import os
import re
import json
import hashlib
import logging
from pathlib import Path
from typing import List

import pdfplumber
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer

# -----------------------------
# CONFIG
# -----------------------------
PDF_DIR = os.getenv("PDF_DIR", "data/documents/pdfs")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "ceo_rag_db")
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "password")

EMBED_MODEL = os.getenv("EMBED_MODEL", "Snowflake/snowflake-arctic-embed-l")
VECTOR_PREFIX = os.getenv("VECTOR_PREFIX", "vectors_doc_")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1400"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("pdf_rebuilder")


# -----------------------------
# HELPERS
# -----------------------------
def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def slugify(text: str) -> str:
    text = Path(text).stem.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")

def chunk_text(txt: str, size: int, overlap: int) -> List[str]:
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
        if start <= 0:
            start = end
    return chunks

def extract_pages(pdf_path: Path) -> List[str]:
    pages = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for p in pdf.pages:
            pages.append(p.extract_text() or "")
    return pages

def make_engine():
    password_part = f":{DB_PASS}" if DB_PASS else ""
    dsn = f"postgresql://{DB_USER}{password_part}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(dsn, echo=False)

def ensure_pgvector(conn):
    try:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        logger.info("pgvector extension OK.")
    except Exception as e:
        logger.warning(f"Could not ensure pgvector extension: {e}")

def upsert_pdf(conn, *, file_name, file_path, file_hash, file_size,
               total_pages, metadata_obj, embedding_model, vector_table) -> int:
    """
    Upsert into pdf_documents keyed by file_hash.
    Cast metadata to jsonb to support jsonb ops later even if column is text.
    """
    sql = text("""
        INSERT INTO pdf_documents
            (file_name, file_path, file_hash, file_size, total_pages,
             extraction_method, metadata, processing_status, embedding_model,
             vector_table_name)
        VALUES
            (:file_name, :file_path, :file_hash, :file_size, :total_pages,
             'pdfplumber', :metadata::jsonb, 'processing', :embedding_model,
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
            vector_table_name = EXCLUDED.vector_table
        RETURNING id
    """)
    row = conn.execute(sql, {
        "file_name": file_name,
        "file_path": file_path,
        "file_hash": file_hash,
        "file_size": file_size,
        "total_pages": total_pages,
        "metadata": json.dumps(metadata_obj or {}),
        "embedding_model": embedding_model,
        "vector_table": vector_table
    }).fetchone()
    return int(row[0])

def drop_and_create_vector_table(conn, table_name: str, dim: int):
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
    logger.info(f'Recreated table "{table_name}" with vector({dim}).')

def insert_chunk(conn, table_name, *, text_chunk, tokens, page_no, metadata_obj, embedding, embed_model):
    sql = text(f"""
        INSERT INTO "{table_name}"
            (chunk_text, chunk_tokens, page_number, chunk_type, metadata,
             embedding, embedding_model)
        VALUES
            (:chunk_text, :chunk_tokens, :page_number, 'content', :metadata::jsonb,
             :embedding::vector, :embedding_model)
    """)
    conn.execute(sql, {
        "chunk_text": text_chunk,
        "chunk_tokens": tokens,
        "page_number": page_no,
        "metadata": json.dumps(metadata_obj or {}),
        "embedding": embedding,
        "embedding_model": embed_model
    })

def mark_completed(conn, pdf_id: int, total_chunks: int):
    """
    Make sure metadata is jsonb; if column is text, cast safely then merge.
    """
    sql = text("""
        UPDATE pdf_documents
        SET processing_status='completed',
            metadata = COALESCE(NULLIF(metadata, '')::jsonb, '{}'::jsonb)
                      || jsonb_build_object('vector_rows', :rows)
        WHERE id = :id
    """)
    conn.execute(sql, {"id": pdf_id, "rows": total_chunks})


# -----------------------------
# CORE
# -----------------------------
def process_pdf(engine, embedder: SentenceTransformer, pdf_path: Path):
    logger.info(f"Processing: {pdf_path.name}")

    file_hash = md5_file(pdf_path)
    file_size = pdf_path.stat().st_size
    pages = extract_pages(pdf_path)
    total_pages = len(pages)

    vector_table = f"{VECTOR_PREFIX}{slugify(pdf_path.name)}"
    embed_dim = embedder.get_sentence_embedding_dimension()

    metadata = {
        "total_pages": total_pages,
        "source": "pdf",
        "file_name": pdf_path.name
    }

    # Upsert and recreate vector table
    with engine.begin() as conn:
        ensure_pgvector(conn)
        pdf_id = upsert_pdf(
            conn,
            file_name=pdf_path.name,
            file_path=str(pdf_path.resolve()),
            file_hash=file_hash,
            file_size=file_size,
            total_pages=total_pages,
            metadata_obj=metadata,
            embedding_model=EMBED_MODEL,
            vector_table=vector_table
        )
        drop_and_create_vector_table(conn, vector_table, embed_dim)

    # Build chunks
    chunks = []
    for pno, ptxt in enumerate(pages, start=1):
        for ch in chunk_text(ptxt, CHUNK_SIZE, CHUNK_OVERLAP):
            chunks.append((pno, ch))
    logger.info(f"Total chunks: {len(chunks)}")

    # Embed and insert
    inserted = 0
    with engine.begin() as conn:
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            texts = [c[1] for c in batch]
            embs = embedder.encode(texts, convert_to_numpy=True)

            for (page_no, text_chunk), emb in zip(batch, embs):
                embedding_str = "[" + ",".join(str(float(x)) for x in emb.tolist()) + "]"
                tokens = len(text_chunk.split())
                insert_chunk(
                    conn,
                    vector_table,
                    text_chunk=text_chunk,
                    tokens=tokens,
                    page_no=page_no,
                    metadata_obj={"file": pdf_path.name, "source_type": "pdf_vector"},
                    embedding=embedding_str,
                    embed_model=EMBED_MODEL
                )
                inserted += 1

        mark_completed(conn, pdf_id, inserted)

    logger.info(f"Completed {pdf_path.name}: {inserted} chunks inserted into {vector_table}.")

def main():
    engine = make_engine()
    embedder = SentenceTransformer(EMBED_MODEL)

    pdf_dir = Path(PDF_DIR)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted([p for p in pdf_dir.glob("*.pdf") if p.is_file()])
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

