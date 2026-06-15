"""
services/llm_bridge_service.py — async bridge to the chatbot_LLM /translate-hieroglyphs endpoint.
"""

from __future__ import annotations

import logging
import os

import httpx

from schemas import TranslationResponse

logger = logging.getLogger("khemet.hieroglyph.llm_bridge")

# Timeout in seconds for the LLM call (LLaMA can be slow on CPU).
_LLM_TIMEOUT = float(os.getenv("LLM_BRIDGE_TIMEOUT", "120"))


async def request_translation(
    symbol_sequence: list[str],
    context_hint: str | None = None,
) -> TranslationResponse:
    """POST the symbol sequence to chatbot_LLM and return a TranslationResponse.

    Args:
        symbol_sequence: Ordered list of Gardiner codes, e.g. ['G17', 'N35', 'A1'].
        context_hint: Optional context, e.g. 'tomb inscription'.

    Returns:
        :class:`~schemas.TranslationResponse` with translation and confidence_note.

    Raises:
        httpx.HTTPStatusError: On non-2xx response from the LLM service.
        httpx.RequestError: On network / connection failure.
    """
    llm_url = os.getenv("LLM_SERVICE_URL", "http://chatbot-llm:8001")
    endpoint = f"{llm_url.rstrip('/')}/api/v1/llm/translate-hieroglyphs"

    payload: dict = {"symbol_sequence": symbol_sequence}
    if context_hint:
        payload["context_hint"] = context_hint

    logger.info(
        "Requesting LLM translation for %d symbols → %s",
        len(symbol_sequence),
        endpoint,
    )

    async with httpx.AsyncClient(timeout=_LLM_TIMEOUT) as client:
        response = await client.post(endpoint, json=payload)
        response.raise_for_status()
        data = response.json()

    return TranslationResponse(
        detected_glyphs=data.get("detected_glyphs", []),
        combined_phonetics=data.get("combined_phonetics", ""),
        translation=data.get("translation", ""),
        confidence_note=data.get("confidence_note", ""),
        cultural_context=data.get("cultural_context", ""),
        transliteration=data.get("transliteration", ""),
        type=data.get("type", ""),
        unknown_codes=data.get("unknown_codes", [])
    )
