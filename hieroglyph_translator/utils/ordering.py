"""
utils/ordering.py — hieroglyph reading-order sorter.

Egyptian hieroglyphs are read left-to-right, top-to-bottom (in the most common
direction).  This module sorts detected symbols by their bounding-box position
using a grid-row approach: symbols whose vertical centres fall within the same
"row band" are sorted horizontally; rows are then sorted top-to-bottom.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from schemas import DetectedSymbol

# Fraction of the median symbol height used to group symbols into the same row.
_ROW_TOLERANCE = 0.5


def sort_symbols_reading_order(symbols: list["DetectedSymbol"]) -> list["DetectedSymbol"]:
    """Sort *symbols* in left-to-right, top-to-bottom reading order.

    Algorithm:
    1. Compute each symbol's vertical centre from its bbox.
    2. Estimate a row-height tolerance from the median symbol height.
    3. Group symbols into rows (symbols within *tolerance* of each other share a row).
    4. Sort rows top-to-bottom, symbols within each row left-to-right.

    Args:
        symbols: List of :class:`~schemas.DetectedSymbol` objects.

    Returns:
        A new list sorted in reading order.
    """
    if not symbols:
        return []

    if len(symbols) == 1:
        return list(symbols)

    # Compute vertical centres and heights from normalised [x1,y1,x2,y2] bboxes.
    def centre_y(s: "DetectedSymbol") -> float:
        return (s.bbox[1] + s.bbox[3]) / 2.0

    def height(s: "DetectedSymbol") -> float:
        return abs(s.bbox[3] - s.bbox[1])

    def centre_x(s: "DetectedSymbol") -> float:
        return (s.bbox[0] + s.bbox[2]) / 2.0

    heights = [height(s) for s in symbols]
    heights_sorted = sorted(heights)
    median_h = heights_sorted[len(heights_sorted) // 2]
    tolerance = median_h * _ROW_TOLERANCE

    # Sort symbols by their vertical centre first.
    sorted_by_y = sorted(symbols, key=centre_y)

    # Group into rows.
    rows: list[list["DetectedSymbol"]] = []
    current_row: list["DetectedSymbol"] = [sorted_by_y[0]]
    current_row_y = centre_y(sorted_by_y[0])

    for sym in sorted_by_y[1:]:
        if abs(centre_y(sym) - current_row_y) <= tolerance:
            current_row.append(sym)
        else:
            rows.append(current_row)
            current_row = [sym]
            current_row_y = centre_y(sym)
    rows.append(current_row)

    # Sort each row left-to-right, then flatten.
    result: list["DetectedSymbol"] = []
    for row in rows:
        result.extend(sorted(row, key=centre_x))

    return result
