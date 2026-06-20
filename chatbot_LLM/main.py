#!/usr/bin/env python3
"""
main.py — Egyptian RAG FastAPI Server
──────────────────────────────────────
REST API for the Egyptian knowledge RAG system.

Endpoints:
    POST /ask        → answer free-form questions
    POST /describe   → visitor description for a monument
    POST /story      → immersive tour-guide narrative
    POST /identify   → CV team: monument name + question
    POST /api/v1/llm/translate-hieroglyphs → YOLO codes → translation
    GET  /health     → service health check

Run:
    uvicorn main:app --host 0.0.0.0 --port 8001 --reload
"""

import json
import os
import signal
import sys
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from middleware.request_id import RequestIDMiddleware
from utils.logger import get_logger
from utils.startup import validate_env

logger = get_logger(__name__)

# ── Ensure the project root is importable ───────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from core.rag_engine import RAGEngine  # noqa: E402
from core.hieroglyph_translator import translate_sequence  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan hook: validates env then loads RAGEngine
    (ChromaDB + embedder + Groq client).
    """
    validate_env()
    logger.info("Starting KHEMET Egyptian RAG server")
    t0 = time.time()

    logger.info("Loading RAG engine")
    app.state.engine = RAGEngine()
    logger.info("RAG engine ready", extra={"elapsed_s": round(time.time() - t0, 1)})

    yield
    logger.info("Shutting down KHEMET Egyptian RAG server")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="KHEMET Egyptian RAG API",
    description=(
        "Retrieval-Augmented Generation API for ancient Egyptian knowledge."
    ),
    version="3.0.0",
    lifespan=lifespan,
    # Disable interactive docs in production
    docs_url="/docs" if os.getenv("ENV") != "production" else None,
    redoc_url="/redoc" if os.getenv("ENV") != "production" else None,
)

# ── Middleware ────────────────────────────────────────────────────────────────

# Request ID — must be registered first
app.add_middleware(RequestIDMiddleware)

# CORS — read from env, never hardcode *
allowed_origins = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

# ── NOTE ─────────────────────────────────────────────────────────────────────
# Voice narration has been moved to AI_services/voice_tour_guide (port 8003).
# This service handles only RAG (Retrieval-Augmented Generation) queries.


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


class StoryRequest(BaseModel):
    artifact_name: str = Field(..., description="Name of the artifact")
    base_description: str = Field(..., description="Current dry description")
    language: str = Field(..., description="'en' or 'ar'")


class StoryResponse(BaseModel):
    story: str


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
    supported_languages: list[str]
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


class GlyphInfo(BaseModel):
    code: str
    english_name: Optional[str] = None
    phonetic: Optional[str] = None
    unicode: Optional[str] = None
    meaning: Optional[str] = None
    category: Optional[str] = None
    determinative: Optional[bool] = None
    found: bool


class TranslateHieroglyphsResponse(BaseModel):
    detected_glyphs: list[GlyphInfo] = []
    combined_phonetics: str = ""
    translation: str = Field(..., description="English translation of the symbol sequence")
    confidence_note: str = Field(..., description="LLM confidence or caveat note")
    cultural_context: str = ""
    transliteration: Optional[str] = None
    type: Optional[str] = None
    unknown_codes: list[str] = []


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
        "POST /ask",
        extra={
            "request_id": getattr(request.state, "request_id", "unknown"),
            "latency_ms": round(latency_ms),
            "query_preview": req.question[:60],
        },
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

    logger.info(
        "POST /describe",
        extra={
            "request_id": getattr(request.state, "request_id", "unknown"),
            "monument": req.monument_name,
        },
    )

    return DescribeResponse(description=description)


@app.post("/story", response_model=StoryResponse)
async def generate_story(req: StoryRequest, request: Request):
    """Rewrite a dry description into an engaging tour guide narrative."""
    engine: RAGEngine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialised yet.")

    prompt_ar = (
        f"أنت مرشد سياحي خبير في المتحف المصري الكبير. "
        f"قم بإعادة كتابة الوصف التالي عن '{req.artifact_name}' إلى قصة مشوقة ومثيرة بصوت مرشد سياحي (فقرة أو فقرتين كحد أقصى). "
        f"أنت تتحدث في ملف صوتي. استخدم المؤثرات الصوتية الخاصة بنظام ElevenLabs بوضعها بين قوسين معقوفين (مثل: [تنهد]، [يضحك بخفة]، [يتوقف قليلاً للتشويق]). "
        f"تحذير هام جدًا: يجب أن تركز فقط على هذه القطعة الأثرية. لا تخرج عن السياق ولا تخترع معلومات غير موجودة في الوصف الأساسي أو السياق المسترجع. لا تتحدث عن تاريخ مصر العام. "
        f"الوصف الأساسي: {req.base_description}"
    )
    prompt_en = (
        f"You are an expert tour guide at the Grand Egyptian Museum. "
        f"Rewrite this description of '{req.artifact_name}' into a captivating, immersive short story (1-2 paragraphs max) that feels like a live tour guide speaking to visitors. "
        f"CRITICAL RULES:\n"
        f"1. You MUST sprinkle in 'expression cues' in brackets to make the performance lifelike! Examples: [laughs softly], [sighs], [pauses for dramatic effect]. Use them naturally.\n"
        f"2. DO NOT go out of context. Focus STRICTLY on the artifact itself.\n"
        f"3. DO NOT invent new facts. Use only the base description and the retrieved context.\n"
        f"4. DO NOT talk about general Egyptian history or unrelated topics.\n"
        f"Base description: {req.base_description}"
    )

    question = prompt_en if req.language == "en" else prompt_ar
    chunks = engine.retrieve(req.artifact_name)
    story = engine.answer(question, chunks=chunks, max_tokens=300)

    logger.info("POST /story generated narrative for %s", req.artifact_name)
    return StoryResponse(story=story)


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
    focused_query = f"{req.monument_name}: {req.question}"
    chunks = engine.retrieve(focused_query)
    answer = engine.answer(req.question, chunks=chunks)
    latency_ms = (time.time() - t0) * 1000

    logger.info(
        "POST /identify",
        extra={
            "request_id": getattr(request.state, "request_id", "unknown"),
            "monument": req.monument_name,
            "latency_ms": round(latency_ms),
        },
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
    engine: RAGEngine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialised yet.")

    filtered_codes = req.symbol_sequence

    # 1. Map codes to gardiner_master.json
    resolved      = translate_sequence(filtered_codes)
    glyphs        = resolved["glyphs"]
    combined      = resolved["combined_phonetics"]
    unknown_codes = resolved["unknown_codes"]

    known_glyphs = [g for g in glyphs if g["found"]]
    if not known_glyphs:
        return TranslateHieroglyphsResponse(
            translation="Uncertain translation",
            confidence_note="None of the provided codes were found in the master list.",
            unknown_codes=unknown_codes,
        )

    glyph_summary_lines = [
        f"- {g['code']} ({g['english_name']}, phonetic: {g['phonetic'] or '?'})"
        for g in known_glyphs
    ]
    glyph_details = "\n".join(glyph_summary_lines)
    context_str   = req.context_hint or "general hieroglyphic inscription"

    # ── Improved prompt: instructs the LLM to check for compound royal titles
    #    (e.g. Nesu Bity = King of Upper and Lower Egypt) BEFORE translating
    #    individual glyphs in isolation. This matches the hero_v5 approach that
    #    produced correct translations.
    prompt = f"""You are an expert Egyptologist with deep knowledge of ancient Egyptian hieroglyphs.
Detected hieroglyphs in correct reading order:
{glyph_details}

Combined phonetics: {combined}
User Context: {context_str}

Instructions:
1. Ignore any glyphs that are likely false positives based on context.
2. Check if this sequence matches any known Egyptian royal titles, deity names, or common phrases (e.g. Nesu Bity = King of Upper and Lower Egypt, Ankh = Life, Sa Ra = Son of Ra).
3. If you recognize a known phrase or title, use that — do not translate individual glyphs in isolation.
4. The 'translation' field MUST be the exact, literal English meaning (e.g. 'King of Upper and Lower Egypt', 'Life', 'Son of Ra'). NEVER put Egyptian pronunciation/phonetics in the translation field.
5. The 'transliteration' field MUST be the Egyptian phonetic pronunciation (e.g. 'nsw-bity', 'ꜥnḫ', 'sꜣ rꜥ').
6. If unsure, say "uncertain translation" rather than guessing.
7. Keep translation short and accurate.

Respond ONLY in this exact JSON format:
{{
  "translation": "short accurate English meaning",
  "transliteration": "Egyptian phonetic spelling/pronunciation",
  "type": "royal title / deity name / common phrase / uncertain",
  "confidence": "high / medium / low",
  "context": "one sentence of cultural context explaining the symbols"
}}"""

    try:
        response = engine.groq_client.chat.completions.create(
            model=engine.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert Egyptologist. You respond only with the requested JSON — no markdown fences, no commentary.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=300,
            temperature=0.2,
        )
        raw = response.choices[0].message.content.strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        llm_data         = json.loads(raw)
        translation      = llm_data.get("translation", "").strip()
        cultural_context = llm_data.get("context", "").strip()
        transliteration  = llm_data.get("transliteration", "").strip()
        type_str         = llm_data.get("type", "").strip()
        confidence_str   = llm_data.get("confidence", "").strip()

    except Exception as exc:
        logger.error("LLM call failed in /translate-hieroglyphs: %s", exc)
        translation      = "Uncertain translation (error)"
        confidence_str   = "low"
        cultural_context = str(exc)
        transliteration  = ""
        type_str         = ""

    detected_glyphs = [
        GlyphInfo(
            code=g["code"],
            english_name=g["english_name"],
            phonetic=g["phonetic"],
            unicode=g["unicode"],
            meaning=g["meaning"],
            category=g["category"],
            determinative=g["determinative"],
            found=g["found"],
        )
        for g in glyphs
    ]

    return TranslateHieroglyphsResponse(
        detected_glyphs=detected_glyphs,
        combined_phonetics=combined,
        translation=translation,
        confidence_note=confidence_str,
        cultural_context=cultural_context,
        transliteration=transliteration,
        type=type_str,
        unknown_codes=unknown_codes,
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
        supported_languages=["ar", "de", "en", "es", "fr", "ru", "zh"],
        response_time_ms=round(latency, 2),
    )


# ── Graceful Shutdown ─────────────────────────────────────────────────────────
def _handle_shutdown(signum, frame):
    logger.info("Shutdown signal received", extra={"signal": signum})
    sys.exit(0)


signal.signal(signal.SIGTERM, _handle_shutdown)
signal.signal(signal.SIGINT, _handle_shutdown)


# ── Run directly ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        access_log=False,  # Logging is handled via structured middleware
    )
