"""
schemas/__init__.py — Pydantic models for the hieroglyph translator service.

Field names are frozen — Flutter/React clients depend on them.
"""

from pydantic import BaseModel, Field


class DetectedSymbol(BaseModel):
    """A single hieroglyph symbol detected by YOLO."""

    gardiner_code: str = Field(..., description="Gardiner classification code, e.g. 'G17'")
    label: str = Field(..., description="Human-readable label, e.g. 'Owl'")
    confidence: float = Field(..., description="Detection confidence in [0, 1]")
    bbox: list[float] = Field(
        ...,
        description="Normalised bounding box [x1, y1, x2, y2] in [0, 1] range",
    )


class DetectionResult(BaseModel):
    """Output of the YOLO detection stage."""

    symbols: list[DetectedSymbol] = Field(default_factory=list)
    symbol_sequence: list[str] = Field(
        default_factory=list,
        description="Ordered list of Gardiner codes in reading order",
    )


class TranslationRequest(BaseModel):
    """Input to the LLM translation stage."""

    symbol_sequence: list[str] = Field(
        ..., description="Ordered Gardiner codes to translate"
    )
    context_hint: str | None = Field(
        None, description="Optional context hint, e.g. 'tomb inscription'"
    )


class GlyphInfo(BaseModel):
    code: str
    english_name: str | None = None
    phonetic: str | None = None
    unicode: str | None = None
    meaning: str | None = None
    category: str | None = None
    determinative: bool | None = None
    found: bool

class TranslationResponse(BaseModel):
    """Output of the LLM translation stage."""
    detected_glyphs: list[GlyphInfo] = Field(default_factory=list)
    combined_phonetics: str = ""
    translation: str = Field(..., description="English translation of the symbol sequence")
    confidence_note: str = Field(..., description="Confidence or caveat note from the LLM")
    cultural_context: str = ""
    transliteration: str | None = None
    type: str | None = None
    unknown_codes: list[str] = Field(default_factory=list)


class HieroglyphPipelineResponse(BaseModel):
    """Full pipeline response returned to the Node.js backend."""

    image_id: str = Field(..., description="8-character UUID prefix for this request")
    detection: DetectionResult
    translation: TranslationResponse | None = Field(
        None,
        description="Translation result, or null if no symbols were detected",
    )
