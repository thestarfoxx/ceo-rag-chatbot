#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Rebuild PDF embeddings for ALL PDFs in a given directory.

Steps:
1) Scan directory for *.pdf
2) For each PDF:
   - Compute hash
   - Upsert pdf_documents row using this hash
   - Drop and recreate its vector table (fresh rebuild)
   - Extract text by page (pdfplumber)
   - Chunk (char-based with overlap)
   - Embed with Snowflake Arctic (SentenceTransformer)
   - Insert all chunks into the per-PDF vector table
   - Mark status=completed (and store vector_rows in metadata)
"""

import os
import re
import json
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any

import pdfplumber
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer

# -----------------------------------------
# CONFIG
# -----------------------------------------
PDF_DIR = os.getenv("PDF_DIR", "data/documents/pdfs")  # change if needed
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "ceo_rag_db")
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "password")

EMBED_MODEL = os.getenv("EMBED_MODEL_NAME", "Snowflake/snowflake-arctic-embed-l")
VECTOR_PREFIX = "vectors_doc_"

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1400"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("pdf_rebuilder")

# -----------------------------------------
# HELPERS
# -----------------------------------------

def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def slugify(text: str) -> str:
    s = Path(text).stem.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")

def chunk_text(txt: str, size: int, overlap: int) -> List[str]:
    txt = (txt or "").replace("\x00", "")
    n = len(txt)
    chunks: List[str] = []
    start = 0
    while start < n:
        end = min(n, start + size)
        piece = txt[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        # prevent infinite loop if overlap >= size
        next_start = end - overlap
        start = next_start if next_start > start else end
    return chunks

def extract_pages(pdf_path: Path) -> List[str]:
    pages: List[str] = []
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
        logger.warning(f"Could not ensure pgvector: {e}")

def upsert_pdf(conn, *, file_name: str, file_path: str, file_hash: str,
               file_size: int, total_pages: int, metadata_json: str,
               embedding_model: str, vector_table: str) -> int:
    # IMPORTANT: no ::jsonb on bound params; let Postgres/column type handle it
    sql = text("""
        INSERT INTO pdf_documents
            (file_name, file_path, file_hash, file_size, total_pages,
             extraction_method, metadata, processing_status, embedding_model,
             vector_table_name)
        VALUES
            (:file_name, :file_path, :file_hash, :file_size, :total_pages,
             'pdfplumber', :metadata, 'processing', :embedding_model,
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
        "metadata": metadata_json,          # stringified JSON
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
    logger.info(f'Recreated table "{table_name}" (vector dim {dim}).')

def insert_chunk(conn, table_name: str, *, text_chunk: str, tokens: int,
                 page_no: int, metadata_obj: Dict[str, Any],
                 embedding_str: str, embedding_model: str):
    """
    Insert one chunk. 'embedding_str' must be pgvector literal: "[0.1,0.2,...]".
    """
    sql = text(f"""
        INSERT INTO "{table_name}"
            (chunk_text, chunk_tokens, page_number, chunk_type, metadata,
             embedding, embedding_model)
        VALUES
            (:chunk_text, :chunk_tokens, :page_number, 'content', :metadata,
             :embedding::vector, :embedding_model)
    """)
    conn.execute(sql, {
        "chunk_text": text_chunk,
        "chunk_tokens": tokens,
        "page_number": page_no,
        "metadata": json.dumps(metadata_obj or {}),
        "embedding": embedding_str,
        "embedding_model": embedding_model
    })

def mark_completed(conn, pdf_id: int, total_chunks: int):
    conn.execute(text("""
        UPDATE pdf_documents
        SET processing_status='completed',
            metadata = COALESCE(metadata, '{}'::jsonb) || jsonb_build_object('vector_rows', :rows)
        WHERE id = :id
    """), {"id": pdf_id, "rows": total_chunks})

# -----------------------------------------
# MAIN PROCESSING
# -----------------------------------------

def process_pdf(engine, embedder: SentenceTransformer, pdf_path: Path):
    logger.info(f"Processing: {pdf_path.name}")

    file_hash = md5_file(pdf_path)
    file_size = pdf_path.stat().st_size

    pages = extract_pages(pdf_path)
    total_pages = len(pages)

    vector_table = f'{VECTOR_PREFIX}{slugify(pdf_path.name)}'
    embed_dim = embedder.get_sentence_embedding_dimension()

    metadata = {
        "total_pages": total_pages,
        "source": "pdf",
        "file_name": pdf_path.name
    }

    # Upsert doc and recreate vectors table
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
    chunks: List[tuple[int, str]] = []
    for pno, ptxt in enumerate(pages, start=1):
        for ch in chunk_text(ptxt, CHUNK_SIZE, CHUNK_OVERLAP):
            chunks.append((pno, ch))
    logger.info(f"Total chunks: {len(chunks)}")

    # Embed & insert
    inserted = 0
    with engine.begin() as conn:
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i+BATCH_SIZE]
            texts = [c[1] for c in batch]
            embs = embedder.encode(texts, convert_to_numpy=True)
            for (page_no, text_chunk), emb in zip(batch, embs):
                # build pgvector literal
                embedding_str = "[" + ",".join(str(float(x)) for x in emb.tolist()) + "]"
                tokens = len(text_chunk.split())  # simple approximation
                insert_chunk(
                    conn,
                    vector_table,
                    text_chunk=text_chunk,
                    tokens=tokens,
                    page_no=page_no,
                    metadata_obj={"file": pdf_path.name, "source_type": "pdf_vector"},
                    embedding_str=embedding_str,          # <-- name matches function
                    embedding_model=EMBED_MODEL
                )
                inserted += 1
        mark_completed(conn, pdf_id, inserted)

    logger.info(f'Completed {pdf_path.name}: {inserted} chunks inserted into "{vector_table}".')

def main():
    logger.info(f"Loading embedding model: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)
    logger.info(f"Embedding dim: {embedder.get_sentence_embedding_dimension()}")

    engine = make_engine()

    pdf_dir = Path(PDF_DIR)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    pdfs = [p for p in sorted(pdf_dir.glob("*.pdf")) if p.is_file()]
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

