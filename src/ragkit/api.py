"""FastAPI surface: POST /ask, POST /ingest, GET /documents, GET /healthz."""

from __future__ import annotations

import shutil
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from ragkit.config import Settings
from ragkit.ingest import SUPPORTED_SUFFIXES
from ragkit.pipeline import Pipeline

app = FastAPI(title="private-rag-kit", version="0.1.0")


@lru_cache(maxsize=1)
def get_pipeline() -> Pipeline:
    return Pipeline.from_settings(Settings())


def require_token(authorization: Annotated[str | None, Header()] = None) -> None:
    token = get_pipeline().settings.api_token
    if token and authorization != f"Bearer {token}":
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing bearer token")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class Citation(BaseModel):
    ref: str
    source: str
    heading: str
    text: str


class AskResponse(BaseModel):
    question: str
    answer: str
    refused: bool
    model: str
    citations: list[Citation]


@app.get("/healthz")
def healthz() -> dict[str, str | int]:
    p = get_pipeline()
    return {"status": "ok", "chunks": p.store.count(), "embedder": p.embedder.name, "llm": p.llm.name}


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(require_token)])
def ask(req: AskRequest) -> AskResponse:
    ans = get_pipeline().answerer.ask(req.question)
    return AskResponse(
        question=ans.question,
        answer=ans.text,
        refused=ans.refused,
        model=ans.model,
        citations=[
            Citation(ref=c.citation, source=c.source, heading=c.heading, text=c.text) for c in ans.citations
        ],
    )


@app.post("/ingest", dependencies=[Depends(require_token)])
def ingest(files: list[UploadFile]) -> dict[str, int | list[str]]:
    tmp = Path(tempfile.mkdtemp(prefix="rag-ingest-"))
    try:
        for f in files:
            name = Path(f.filename or "upload").name
            if Path(name).suffix.lower() not in SUPPORTED_SUFFIXES:
                raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"unsupported file: {name}")
            with (tmp / name).open("wb") as out:
                shutil.copyfileobj(f.file, out)
        report = get_pipeline().ingest(tmp)
        return {"documents": report.documents, "chunks": report.chunks, "skipped": report.skipped}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@app.get("/documents", dependencies=[Depends(require_token)])
def documents() -> list[dict[str, str | int]]:
    return [
        {"doc_id": d, "source": s, "title": t, "chunks": n} for d, s, t, n in get_pipeline().store.documents()
    ]
