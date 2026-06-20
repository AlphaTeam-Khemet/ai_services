"""
services/elevenlabs_service.py
==============================
TTS engine using the ElevenLabs API — the sole audio generation backend.

Converts narration text to MP3 audio using the eleven_multilingual_v2 model,
which supports both English and Arabic with high naturalness.

Audio files are saved to static/audio/ and served via the /static route.
This service never raises exceptions — all errors are logged as warnings
and None is returned to the caller (no fallback TTS exists).

Environment Variables:
    ELEVENLABS_API_KEY  : ElevenLabs API key (required)
    ELEVENLABS_VOICE_EN : Voice ID for English (default: George JBFqnCBsd6RMkjVDRZzb)
    ELEVENLABS_VOICE_AR : Voice ID for Arabic  (default: Sarah  EXAVITQu4vr4xnSDxMaL)

Voice Selection:
    Free-tier voices are used by default to ensure compatibility.
    Professional voices (James, Oliver Silk) require a paid plan
    and will return HTTP 402 on free accounts.
"""

import asyncio
import logging
import os

logger = logging.getLogger("khemet.voice.elevenlabs")

# Default free-tier voices — work on all ElevenLabs accounts
_FALLBACK_EN = "JBFqnCBsd6RMkjVDRZzb"  # George — warm storyteller voice
_FALLBACK_AR = "EXAVITQu4vr4xnSDxMaL"  # Sarah  — mature, confident voice


async def _generate_with_elevenlabs(
    artifact_id: str,
    text: str,
    language: str,
) -> "str | None":
    """
    Call the ElevenLabs TTS API and save the resulting audio to disk.

    Returns the relative audio_url ("/static/audio/...") on success,
    or None on any failure. Never raises.
    """
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        logger.warning(
            "ELEVENLABS_API_KEY not set — skipping ElevenLabs for %s [%s]",
            artifact_id,
            language,
        )
        return None

    try:
        from elevenlabs.client import ElevenLabs

        # Select the appropriate voice for the requested language
        voice_id = (
            os.getenv("ELEVENLABS_VOICE_EN", _FALLBACK_EN)
            if language == "en"
            else os.getenv("ELEVENLABS_VOICE_AR", _FALLBACK_AR)
        )

        logger.info(
            "ElevenLabs TTS → voice=%s  lang=%s  artifact=%s",
            voice_id,
            language,
            artifact_id,
        )

        el_client = ElevenLabs(api_key=api_key)

        def _call_elevenlabs() -> bytes:
            """
            Call the ElevenLabs API and drain the response generator fully
            inside the worker thread — safe to block here.
            The generator MUST be consumed in the same thread that called
            convert() otherwise it may stall or be exhausted prematurely.
            """
            audio_generator = el_client.text_to_speech.convert(
                text=text,
                voice_id=voice_id,
                model_id="eleven_multilingual_v2",
            )
            # Drain the generator into a single bytes object
            chunks = []
            for chunk in audio_generator:
                if isinstance(chunk, bytes):
                    chunks.append(chunk)
            return b"".join(chunks)

        audio_bytes = await asyncio.to_thread(_call_elevenlabs)

        if not audio_bytes:
            logger.warning("ElevenLabs returned empty audio for %s [%s]", artifact_id, language)
            return None

        audio_dir = "static/audio"
        os.makedirs(audio_dir, exist_ok=True)

        # Sanitize artifact_id: replace spaces and non-alphanumeric chars
        # so the filename is always safe on any filesystem.
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(artifact_id))
        filename = f"narration_{safe_id}_{language}.mp3"
        filepath = os.path.join(audio_dir, filename)

        with open(filepath, "wb") as f:
            f.write(audio_bytes)

        audio_url = f"/static/audio/{filename}"
        logger.info("ElevenLabs audio saved → %s (%d bytes)", filepath, len(audio_bytes))
        return audio_url

    except Exception as exc:
        logger.warning("ElevenLabs TTS failed for %s [%s]: %s", artifact_id, language, exc)
        return None


async def generate_audio_elevenlabs(
    artifact_id: str,
    text: str,
    language: str,
) -> "str | None":
    """
    Public entry point for ElevenLabs TTS.

    Returns audio_url on success, None on failure.
    Called by the router as the sole TTS attempt.
    """
    return await _generate_with_elevenlabs(artifact_id, text, language)
