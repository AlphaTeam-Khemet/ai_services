#!/usr/bin/env python3
"""
main.py — Egyptian RAG FastAPI Server
──────────────────────────────────────
REST API for the Egyptian knowledge RAG system.

Endpoints:
    POST /ask        → answer free-form questions
    POST /describe   → visitor description for a monument
    POST /identify   → CV team: monument name + question
    GET  /health     → service health check

Run:
    uvicorn main:app --host 0.0.0.0 --port 8001 --reload
"""

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("khemet.api")

# ── Ensure the project root is importable ───────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from core.rag_engine import RAGEngine  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan hook: loads RAGEngine (ChromaDB + embedder + Groq client).
    """
    logger.info("🚀 Starting KHEMET Egyptian RAG server ...")
    t0 = time.time()

    logger.info("Loading RAG engine ...")
    app.state.engine = RAGEngine()
    logger.info("✅ RAG engine ready in %.1fs\n", time.time() - t0)

    yield
    logger.info("🛑 Shutting down KHEMET Egyptian RAG server.")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="KHEMET Egyptian RAG API",
    description=(
        "Retrieval-Augmented Generation API for ancient Egyptian knowledge."
    ),
    version="3.0.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════════
# Request / Response Models
# ═══════════════════════════════════════════════════════════════════════════════

class AskRequest(BaseModel):
    question: str = Field(..., description="The question to ask about ancient Egypt.")
    topic: Optional[str] = Field(None, description="Optional topic to narrow the search (English recommended)")
    history: Optional[list[dict]] = Field(
        None,
        description=(
            "Prior conversation turns as a list of {\"role\": \"user\"|\"assistant\", \"content\": str} dicts. "
            "The last 6 turns are used (3 exchanges). Pass null or omit to start a fresh conversation."
        ),
    )


class SourceInfo(BaseModel):
    topic_name: str
    topic_id: str
    category: str
    section: str
    period: str
    location: str
    similarity: float


class AskResponse(BaseModel):
    answer: str = Field(..., description="Answer to the question.")
    sources: list[SourceInfo]
    latency_ms: float = Field(..., description="End-to-end latency in milliseconds.")


class DescribeRequest(BaseModel):
    monument_name: str = Field(..., description="Name of the monument to describe (English recommended for best retrieval)")


class DescribeResponse(BaseModel):
    description: str


class IdentifyRequest(BaseModel):
    monument_name: str = Field(..., description="Monument name detected by the CV model (English)")
    question: str = Field(..., description="Question about the identified monument.")


class IdentifyResponse(BaseModel):
    answer: str
    monument: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    version: str
    total_chunks: int
    response_time_ms: float


class TranslateHieroglyphsRequest(BaseModel):
    symbol_sequence: list[str] = Field(
        ...,
        description="Ordered list of Gardiner codes, e.g. ['G17', 'N35', 'A1']",
    )
    context_hint: Optional[str] = Field(
        None,
        description="Optional context, e.g. 'tomb inscription'",
    )


class TranslateHieroglyphsResponse(BaseModel):
    translation: str = Field(..., description="English translation of the symbol sequence")
    confidence_note: str = Field(..., description="LLM confidence or caveat note")


# ═══════════════════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/ask", response_model=AskResponse)
async def ask_question(req: AskRequest, request: Request):
    """Answer a free-form question about ancient Egypt."""
    engine: RAGEngine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialised yet.")

    t0 = time.time()
    chunks = engine.retrieve(req.question)
    answer = engine.answer(req.question, chunks=chunks, history=req.history)
    latency_ms = (time.time() - t0) * 1000

    logger.info(
        "POST /ask | latency=%.0fms | query='%s'",
        latency_ms,
        req.question[:60],
    )

    sources = [
        SourceInfo(
            topic_name=c["metadata"]["topic_name"],
            topic_id=c["metadata"]["topic_id"],
            category=c["metadata"]["category"],
            section=c["metadata"]["section"],
            period=c["metadata"]["period"],
            location=c["metadata"]["location"],
            similarity=round(c["similarity"], 4),
        )
        for c in chunks
    ]

    return AskResponse(answer=answer, sources=sources, latency_ms=round(latency_ms, 2))


@app.post("/describe", response_model=DescribeResponse)
async def describe_monument(req: DescribeRequest, request: Request):
    """Generate a visitor-friendly description for a monument."""
    engine: RAGEngine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialised yet.")

    description = engine.describe_monument(req.monument_name)

    logger.info("POST /describe | monument='%s'", req.monument_name)

    return DescribeResponse(description=description)


@app.post("/identify", response_model=IdentifyResponse)
async def identify_monument(req: IdentifyRequest, request: Request):
    """
    For the computer vision team:
    Takes a detected monument name (English) + a question and returns an answer.
    """
    engine: RAGEngine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialised yet.")

    t0 = time.time()
    # Build a monument-focused query so retrieval is anchored to the
    # detected monument, not just the raw question alone.
    focused_query = f"{req.monument_name}: {req.question}"
    chunks = engine.retrieve(focused_query)
    answer = engine.answer(req.question, chunks=chunks)
    latency_ms = (time.time() - t0) * 1000

    logger.info(
        "POST /identify | monument='%s' | latency=%.0fms",
        req.monument_name,
        latency_ms,
    )

    return IdentifyResponse(
        answer=answer,
        monument=req.monument_name,
        latency_ms=round(latency_ms, 2),
    )


@app.post("/api/v1/llm/translate-hieroglyphs", response_model=TranslateHieroglyphsResponse)
async def translate_hieroglyphs(
    req: TranslateHieroglyphsRequest, request: Request
):
    """Translate a sequence of Gardiner hieroglyph codes into English.

    Called internally by the hieroglyph_translator service — not exposed to clients.
    """
    engine: RAGEngine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialised yet.")

    codes_str = ", ".join(req.symbol_sequence)
    context_str = req.context_hint or "general hieroglyphic inscription"

    prompt = (
        f"You are an expert Egyptologist. Translate the following sequence of "
        f"Egyptian hieroglyph Gardiner codes into English. "
        f"Codes: {codes_str}. "
        f"Context: {context_str}. "
        f"Provide a concise translation and any relevant notes. "
        f"Respond in JSON with exactly two keys: \\'translation\\' and \\'confidence_note\\'."
    )

    messages = [
        {"role": "system", "content": "You are KHEMET, an expert Egyptologist at the Grand Egyptian Museum."},
        {"role": "user", "content": prompt},
    ]

    try:
        response = engine.groq_client.chat.completions.create(
            model=engine.llm_model,
            messages=messages,
            max_tokens=400,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content.strip()
        import json as _json
        data = _json.loads(raw)
        translation = data.get("translation", raw)
        confidence_note = data.get(
            "confidence_note",
            "Translation generated by LLM based on Gardiner sign list.",
        )
    except Exception as exc:
        logger.error("Hieroglyph translation failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Translation failed: {exc}")

    logger.info(
        "POST /api/v1/llm/translate-hieroglyphs | codes=%s",
        codes_str[:80],
    )

    return TranslateHieroglyphsResponse(
        translation=translation,
        confidence_note=confidence_note,
    )


@app.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    """Service health check — returns status, version, and chunk count."""
    t0 = time.time()
    engine = getattr(request.app.state, "engine", None)
    total = engine.collection.count() if engine else 0
    latency = (time.time() - t0) * 1000
    return HealthResponse(
        status="ok",
        version="3.0.0",
        total_chunks=total,
        response_time_ms=round(latency, 2),
    )


# ── Run directly ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
    )
