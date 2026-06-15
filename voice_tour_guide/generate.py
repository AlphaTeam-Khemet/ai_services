"""
generate.py
===========
Voice generation endpoint for the KHEMET Voice Tour Guide service.

This service has a single responsibility: receive text and generate
an MP3 audio file using ElevenLabs TTS.

It does NOT handle caching, database operations, or business logic.
All of that is handled by the Node.js backend.

Endpoint:
    POST /generate

Request Body:
    artifact_id  : Unique identifier for the artifact (used for filename)
    language     : "en" or "ar"
    text         : Text to convert to speech (max 5000 chars)

Response:
    audio_url    : Relative URL to the generated MP3 file
                   null if ElevenLabs generation failed
"""

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field, field_validator

from services.elevenlabs_service import generate_audio_elevenlabs
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Generate"])

# Maximum text length to prevent abuse and control ElevenLabs costs
MAX_TEXT_LENGTH = 5000


class GenerateRequest(BaseModel):
    artifact_id: str = Field(
        ...,
        description="Artifact UUID from monuments table",
        max_length=100,
    )
    language: str = Field(..., description="'en' or 'ar'")
    text: str = Field(
        ...,
        description="Text to convert to speech",
        max_length=MAX_TEXT_LENGTH,
    )

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        if v not in ("en", "ar"):
            raise ValueError("language must be 'en' or 'ar'")
        return v

    @field_validator("artifact_id")
    @classmethod
    def validate_artifact_id(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("artifact_id cannot be empty")
        return v.strip()

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text cannot be empty")
        return v.strip()


class GenerateResponse(BaseModel):
    audio_url: "str | None"


@router.post("/generate", response_model=GenerateResponse)
async def generate_narration(req: GenerateRequest, request: Request):
    """
    Generate an MP3 audio file from the provided text using ElevenLabs TTS.
    Returns the audio URL on success, null if generation failed.
    No DB operations — this service only generates audio.
    """
    request_id = getattr(request.state, "request_id", "unknown")

    logger.info(
        "Audio generation requested",
        extra={
            "request_id": request_id,
            "artifact_id": req.artifact_id,
            "language": req.language,
            "text_length": len(req.text),
        },
    )

    audio_url = await generate_audio_elevenlabs(
        artifact_id=req.artifact_id,
        text=req.text,
        language=req.language,
    )

    if audio_url:
        logger.info(
            "Audio generation succeeded",
            extra={
                "request_id": request_id,
                "artifact_id": req.artifact_id,
                "audio_url": audio_url,
            },
        )
    else:
        logger.warning(
            "Audio generation failed — ElevenLabs returned None",
            extra={
                "request_id": request_id,
                "artifact_id": req.artifact_id,
            },
        )

    return GenerateResponse(audio_url=audio_url)
