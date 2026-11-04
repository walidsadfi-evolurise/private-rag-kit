# Changelog

## 0.1.0 — 2026-11-04

- Ingestion for PDF, DOCX, HTML, Markdown and plain text with heading-aware chunking.
- Embedders: hash (offline), Ollama, OpenAI-compatible.
- Stores: in-memory with on-disk persistence, PostgreSQL + pgvector.
- Hybrid retrieval (BM25 + dense) fused with Reciprocal Rank Fusion.
- Grounded answering with verified citations and a refusal path; fake LLM for tests.
- `rag` CLI (ingest / ask / eval / status / serve) and FastAPI with bearer-token auth.
- Eval loop with hit@k, answer accuracy, citation rate, refusal recall/precision.
- Docker image and compose stack (API + pgvector).
