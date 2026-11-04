"""PostgreSQL + pgvector store with a tsvector column for lexical search.

Requires `pip install private-rag-kit[postgres]` and the `vector` extension.
Hybrid search is done in SQL so the database does the ranking; the Python side
only fuses the two result lists.
"""

from __future__ import annotations

import numpy as np

from ragkit.models import Chunk
from ragkit.store.base import Store

_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS chunks (
  chunk_id  text PRIMARY KEY,
  doc_id    text NOT NULL,
  source    text NOT NULL,
  title     text NOT NULL,
  heading   text NOT NULL DEFAULT '',
  text      text NOT NULL,
  position  integer NOT NULL,
  embedding vector({dim}) NOT NULL,
  tsv       tsvector GENERATED ALWAYS AS
              (to_tsvector('simple', title || ' ' || heading || ' ' || text)) STORED
);
CREATE INDEX IF NOT EXISTS chunks_doc_idx ON chunks (doc_id);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_emb_idx ON chunks USING hnsw (embedding vector_cosine_ops);
"""


def _row_to_chunk(row: tuple) -> Chunk:  # type: ignore[type-arg]
    return Chunk(
        chunk_id=row[0],
        doc_id=row[1],
        source=row[2],
        title=row[3],
        heading=row[4],
        text=row[5],
        position=row[6],
    )


class PgVectorStore(Store):
    def __init__(self, dsn: str, dim: int = 1536) -> None:
        import psycopg
        from pgvector.psycopg import register_vector

        self.conn = psycopg.connect(dsn, autocommit=True)
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)
        self.dim = dim
        with self.conn.cursor() as cur:
            cur.execute(_SCHEMA.format(dim=dim))

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if not chunks:
            return
        with self.conn.cursor() as cur:
            for doc_id in {c.doc_id for c in chunks}:
                cur.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
            cur.executemany(
                "INSERT INTO chunks (chunk_id, doc_id, source, title, heading, text, position, embedding) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                [
                    (c.chunk_id, c.doc_id, c.source, c.title, c.heading, c.text, c.position, vectors[i])
                    for i, c in enumerate(chunks)
                ],
            )

    def delete_document(self, doc_id: str) -> int:
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
            return cur.rowcount or 0

    def vector_search(self, query: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT chunk_id, doc_id, source, title, heading, text, position, "
                "1 - (embedding <=> %s) AS score "
                "FROM chunks ORDER BY embedding <=> %s LIMIT %s",
                (query, query, k),
            )
            return [(_row_to_chunk(r), float(r[7])) for r in cur.fetchall()]

    def lexical_search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT chunk_id, doc_id, source, title, heading, text, position, "
                "ts_rank_cd(tsv, q) AS score FROM chunks, plainto_tsquery('simple', %s) q "
                "WHERE tsv @@ q ORDER BY score DESC LIMIT %s",
                (query, k),
            )
            return [(_row_to_chunk(r), float(r[7])) for r in cur.fetchall()]

    def count(self) -> int:
        with self.conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM chunks")
            row = cur.fetchone()
            return int(row[0]) if row else 0

    def documents(self) -> list[tuple[str, str, str, int]]:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT doc_id, min(source), min(title), count(*) FROM chunks GROUP BY doc_id ORDER BY 2"
            )
            return [(r[0], r[1], r[2], int(r[3])) for r in cur.fetchall()]
