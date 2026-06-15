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

        # The ElevenLabs SDK is synchronous — wrap in asyncio.to_thread so it
        # does not block the FastAPI event loop during network I/O
        audio_data = await asyncio.to_thread(
            el_client.text_to_speech.convert,
            text=text,
            voice_id=voice_id,
            model_id="eleven_multilingual_v2",
        )

        audio_dir = "static/audio"
        os.makedirs(audio_dir, exist_ok=True)

        # Filename encodes artifact_id and language to prevent collisions
        filename = f"narration_{artifact_id}_{language}.mp3"
        filepath = os.path.join(audio_dir, filename)

        # The SDK may return raw bytes or a generator of chunks depending on
        # the response mode — handle both cases uniformly before writing
        audio_bytes = (
            audio_data if isinstance(audio_data, bytes) else b"".join(list(audio_data))
        )
        with open(filepath, "wb") as f:
            f.write(audio_bytes)

        audio_url = f"/static/audio/{filename}"
        logger.info("ElevenLabs audio saved → %s", filepath)
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
