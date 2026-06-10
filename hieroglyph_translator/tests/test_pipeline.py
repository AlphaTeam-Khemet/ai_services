"""
tests/test_pipeline.py — smoke tests for hieroglyph_translator.

Run from the AI_services/hieroglyph_translator/ directory:
    pytest tests/ -v
"""

from __future__ import annotations

import sys
import os
import types

# ── Ensure the service root is importable without installing the package ──────
SERVICE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SERVICE_ROOT not in sys.path:
    sys.path.insert(0, SERVICE_ROOT)

# ── Stub ultralytics so tests don't require GPU/model weights ────────────────
ultralytics_stub = types.ModuleType("ultralytics")
ultralytics_stub.YOLO = None  # type: ignore[attr-defined]
sys.modules.setdefault("ultralytics", ultralytics_stub)

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from schemas import DetectedSymbol, DetectionResult, TranslationResponse, HieroglyphPipelineResponse
from utils.gardiner_lookup import get_symbol_label
from utils.ordering import sort_symbols_reading_order


# ═══════════════════════════════════════════════════════════════════════════════
# utils/ordering.py
# ═══════════════════════════════════════════════════════════════════════════════

def _make_symbol(code: str, x1: float, y1: float, x2: float, y2: float) -> DetectedSymbol:
    return DetectedSymbol(
        gardiner_code=code,
        label=code,
        confidence=0.9,
        bbox=[x1, y1, x2, y2],
    )


class TestSortSymbolsReadingOrder:
    def test_empty_list(self):
        assert sort_symbols_reading_order([]) == []

    def test_single_symbol(self):
        s = _make_symbol("G17", 0.1, 0.1, 0.2, 0.2)
        result = sort_symbols_reading_order([s])
        assert result == [s]

    def test_two_rows_sorted_top_to_bottom(self):
        # Row 2 (lower on page) should come after Row 1
        row2 = _make_symbol("N35", 0.1, 0.6, 0.2, 0.7)
        row1 = _make_symbol("G17", 0.1, 0.1, 0.2, 0.2)
        result = sort_symbols_reading_order([row2, row1])
        assert result[0].gardiner_code == "G17"
        assert result[1].gardiner_code == "N35"

    def test_same_row_sorted_left_to_right(self):
        right = _make_symbol("A1", 0.7, 0.1, 0.8, 0.2)
        left = _make_symbol("G17", 0.1, 0.1, 0.2, 0.2)
        result = sort_symbols_reading_order([right, left])
        assert result[0].gardiner_code == "G17"
        assert result[1].gardiner_code == "A1"

    def test_multi_row_multi_col(self):
        # Row 0: G17 (left), N35 (right)
        # Row 1: A1 (centre)
        g17 = _make_symbol("G17", 0.05, 0.05, 0.25, 0.25)
        n35 = _make_symbol("N35", 0.55, 0.05, 0.75, 0.25)
        a1  = _make_symbol("A1",  0.30, 0.60, 0.50, 0.80)
        result = sort_symbols_reading_order([a1, n35, g17])
        codes = [s.gardiner_code for s in result]
        assert codes == ["G17", "N35", "A1"]


# ═══════════════════════════════════════════════════════════════════════════════
# utils/gardiner_lookup.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetSymbolLabel:
    def test_known_code_returns_string(self):
        label = get_symbol_label("G17")
        assert isinstance(label, str)
        assert len(label) > 0

    def test_known_code_contains_code(self):
        label = get_symbol_label("N35")
        assert "N35" in label

    def test_unknown_code_returns_code_itself(self):
        label = get_symbol_label("ZZZ99")
        assert label == "ZZZ99"

    def test_common_codes_present(self):
        for code in ["A1", "G17", "N35", "O1", "X1"]:
            label = get_symbol_label(code)
            assert label != "", f"Empty label for {code}"


# ═══════════════════════════════════════════════════════════════════════════════
# services/pipeline_service.py (mocked integration)
# ═══════════════════════════════════════════════════════════════════════════════

FAKE_DETECTION = DetectionResult(
    symbols=[
        DetectedSymbol(gardiner_code="G17", label="Bird sign G17", confidence=0.91, bbox=[0.1, 0.1, 0.3, 0.3]),
        DetectedSymbol(gardiner_code="N35", label="Sky/Earth/Water sign N35", confidence=0.85, bbox=[0.4, 0.1, 0.6, 0.3]),
        DetectedSymbol(gardiner_code="A1",  label="Man sign A1", confidence=0.78, bbox=[0.7, 0.1, 0.9, 0.3]),
    ],
    symbol_sequence=["G17", "N35", "A1"],
)

FAKE_TRANSLATION = TranslationResponse(
    translation="In the name of the great one...",
    confidence_note="High confidence based on common funerary formula",
)


@pytest.mark.asyncio
async def test_run_pipeline_full():
    """Mocked integration: detect_symbols + request_translation both succeed."""
    from services.pipeline_service import run_pipeline

    with (
        patch("services.pipeline_service.detect_symbols", new=AsyncMock(return_value=FAKE_DETECTION)),
        patch("services.pipeline_service.request_translation", new=AsyncMock(return_value=FAKE_TRANSLATION)),
    ):
        result = await run_pipeline(b"fake_image_bytes", context_hint="tomb inscription")

    assert isinstance(result, HieroglyphPipelineResponse)
    assert len(result.image_id) == 8
    assert result.detection.symbol_sequence == ["G17", "N35", "A1"]
    assert result.translation is not None
    assert "great one" in result.translation.translation


@pytest.mark.asyncio
async def test_run_pipeline_no_symbols_skips_llm():
    """When detection returns no symbols, LLM must NOT be called."""
    from services.pipeline_service import run_pipeline

    empty_detection = DetectionResult(symbols=[], symbol_sequence=[])
    mock_translate = AsyncMock()

    with (
        patch("services.pipeline_service.detect_symbols", new=AsyncMock(return_value=empty_detection)),
        patch("services.pipeline_service.request_translation", new=mock_translate),
    ):
        result = await run_pipeline(b"fake_image_bytes")

    assert result.translation is None
    mock_translate.assert_not_called()


@pytest.mark.asyncio
async def test_run_pipeline_returns_unique_image_ids():
    """Each pipeline call should produce a different image_id."""
    from services.pipeline_service import run_pipeline

    with (
        patch("services.pipeline_service.detect_symbols", new=AsyncMock(return_value=FAKE_DETECTION)),
        patch("services.pipeline_service.request_translation", new=AsyncMock(return_value=FAKE_TRANSLATION)),
    ):
        r1 = await run_pipeline(b"img1")
        r2 = await run_pipeline(b"img2")

    assert r1.image_id != r2.image_id
