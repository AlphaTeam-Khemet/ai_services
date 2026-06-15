#!/usr/bin/env python3
"""
hieroglyph_translator.py — Gardiner-code → metadata resolver
──────────────────────────────────────────────────────────────
Loads gardiner_master.json once at import time and exposes a
single public function:

    translate_sequence(gardiner_codes: list[str]) -> dict

Returns per-glyph metadata plus combined phonetics and the list
of any codes that were not found in the master file.
"""

import json
import logging
import os
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

# ── Master-file path ──────────────────────────────────────────
# chatbot_LLM/core/  →  ../../../knowledge/gardiner_master.json
_CORE_DIR   = os.path.dirname(os.path.abspath(__file__))
_MASTER_PATH = os.path.abspath(
    os.path.join(_CORE_DIR, "..", "knowledge", "gardiner_master.json")
)


# ── Load once, cache forever ──────────────────────────────────
@lru_cache(maxsize=1)
def _load_master() -> dict:
    """Load and cache the Gardiner master JSON file."""
    if not os.path.exists(_MASTER_PATH):
        raise FileNotFoundError(
            f"gardiner_master.json not found at: {_MASTER_PATH}\n"
            "Run the knowledge-base build scripts before starting the server."
        )
    with open(_MASTER_PATH, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    logger.info("✅ Loaded gardiner_master.json — %d entries from %s", len(data), _MASTER_PATH)
    return data


# ── Public API ────────────────────────────────────────────────

def translate_sequence(gardiner_codes: list[str]) -> dict:
    """
    Resolve a list of Gardiner codes against the master file.

    Parameters
    ----------
    gardiner_codes : list[str]
        Ordered list of Gardiner sign codes, e.g. ["G1", "S29", "X1"].

    Returns
    -------
    dict with keys:
        glyphs            — list of per-glyph dicts (see below)
        combined_phonetics— hyphen-joined phonetics of resolved glyphs
        meanings          — list of english_name values (one per resolved glyph)
        unknown_codes     — list of codes not found in the master file

    Each glyph dict:
        code          : str            — original Gardiner code
        english_name  : str | None
        phonetic      : str | None
        unicode       : str | None     — single Egyptian hieroglyph character
        meaning       : str | None     — short meaning string from KB
        category      : str | None
        determinative : bool | None
        found         : bool           — False when the code is unknown
    """
    master = _load_master()

    glyphs: list[dict] = []
    unknown_codes: list[str] = []
    phonetics: list[str] = []
    meanings: list[str] = []

    for code in gardiner_codes:
        entry = master.get(code)

        if entry is None:
            unknown_codes.append(code)
            glyphs.append({
                "code":          code,
                "english_name":  None,
                "phonetic":      None,
                "unicode":       None,
                "meaning":       None,
                "category":      None,
                "determinative": None,
                "found":         False,
            })
            continue

        phonetic: Optional[str] = entry.get("phonetic")
        english:  Optional[str] = entry.get("english_name")

        glyphs.append({
            "code":          code,
            "english_name":  english,
            "phonetic":      phonetic,
            "unicode":       entry.get("unicode"),
            "meaning":       entry.get("meaning"),
            "category":      entry.get("category"),
            "determinative": entry.get("determinative"),
            "found":         True,
        })

        if phonetic:
            phonetics.append(phonetic)
        if english:
            meanings.append(english)

    return {
        "glyphs":             glyphs,
        "combined_phonetics": "-".join(phonetics) if phonetics else "",
        "meanings":           meanings,
        "unknown_codes":      unknown_codes,
    }


def get_master_size() -> int:
    """Return the total number of entries in the master file (for health checks)."""
    return len(_load_master())


def sort_glyphs_by_position(codes, confidences, bboxes, reading_direction="ltr"):
    """
    Sort glyphs spatially based on bounding boxes.
    Sort by y1 first (top to bottom), then x1 (left to right or right to left).
    """
    combined = list(zip(codes, confidences, bboxes))
    if reading_direction == "rtl":
        # top to bottom, right to left
        combined.sort(key=lambda item: (item[2][1], -item[2][0]))
    else:
        # top to bottom, left to right
        combined.sort(key=lambda item: (item[2][1], item[2][0]))
    return zip(*combined)
