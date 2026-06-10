#!/usr/bin/env python3
"""
pipeline/audit_raw_files.py
───────────────────────────
Scans every .txt file in data/raw/ and prints a structured quality report:

  Section 1 — Files with valid headers
  Section 2 — Files with missing / incomplete headers
  Section 3 — Files with body quality issues  (thin content, bullets, no headings)
  Section 4 — Summary table

Run from the project root:
    python pipeline/audit_raw_files.py
"""

import os
import re

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
RAW_DIR    = os.path.join(ROOT_DIR, "data", "raw")

# ── Constants ──────────────────────────────────────────────────────────────────
REQUIRED_FIELDS    = {"topic_id", "topic_name", "category", "period", "location"}
MIN_BODY_CHARS     = 500          # bodies shorter than this are flagged as too thin
HEADING_RE         = re.compile(  # mirrors the heuristic in build_chunks.py
    r"^[A-Z][^\.\n]{0,79}$"
)
BULLET_RE          = re.compile(r"^\s*[-•]")


# ── Helpers ────────────────────────────────────────────────────────────────────

def parse_file(content: str) -> tuple[dict, str]:
    """
    Split *content* into (metadata_dict, body).
    Returns ({}, "") if the separator is not found.
    """
    if "---" not in content:
        return {}, ""

    header_text, _, body = content.partition("---")
    meta: dict[str, str] = {}
    for line in header_text.strip().splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()

    return meta, body.strip()


def has_section_headings(body: str) -> bool:
    """
    Return True if at least one line in the body looks like a section heading.
    Uses the same heuristic as build_chunks.py:
      - length ≤ 80 chars
      - starts with an uppercase letter
      - does NOT end with a period
      - surrounded by blank-ish context (simplified here: just checks structure)
    """
    lines = body.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or len(stripped) > 80:
            continue
        if not stripped[0].isupper():
            continue
        if stripped.endswith("."):
            continue
        # Require at least one blank line in the neighbourhood
        prev_blank = (i == 0) or (lines[i - 1].strip() == "")
        next_blank = (i == len(lines) - 1) or (lines[i + 1].strip() == "")
        if prev_blank or next_blank:
            return True
    return False


def has_bullets(body: str) -> bool:
    """Return True if any line starts with a bullet marker (- or •)."""
    return any(BULLET_RE.match(line) for line in body.splitlines())


# ── Main audit ─────────────────────────────────────────────────────────────────

def audit() -> None:
    txt_files = sorted(
        f for f in os.listdir(RAW_DIR)
        if f.endswith(".txt") and os.path.isfile(os.path.join(RAW_DIR, f))
    )

    if not txt_files:
        print(f"No .txt files found in {RAW_DIR}")
        return

    valid:           list[tuple[str, str, str]]  = []   # (filename, category, topic_name)
    broken:          list[tuple[str, list[str]]] = []   # (filename, [issues])
    quality_issues:  list[tuple[str, list[str]]] = []   # (filename, [issues])
    too_short_count: int = 0

    for fname in txt_files:
        path = os.path.join(RAW_DIR, fname)
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            content = fh.read()

        meta, body = parse_file(content)

        # ── Header checks ──────────────────────────────────────────────────
        header_problems: list[str] = []

        if "---" not in content:
            header_problems.append("missing --- separator")

        missing_fields = REQUIRED_FIELDS - set(meta.keys())
        if missing_fields:
            header_problems.append(
                "missing field(s): " + ", ".join(sorted(missing_fields))
            )

        if not body:
            header_problems.append("empty body after separator")

        if header_problems:
            broken.append((fname, header_problems))
            continue   # no point running quality checks on broken files

        # ── Body quality checks ────────────────────────────────────────────
        body_problems: list[str] = []

        body_len = len(body)
        if body_len < MIN_BODY_CHARS:
            body_problems.append(
                f"body too short ({body_len} chars, minimum {MIN_BODY_CHARS})"
            )
            too_short_count += 1

        if has_bullets(body):
            body_problems.append(
                "contains bullet points (lines starting with - or •) — "
                "bullets chunk poorly, prefer prose"
            )

        if not has_section_headings(body):
            body_problems.append(
                "no section headings detected — single-section files produce "
                "only 1-2 chunks, low retrieval coverage"
            )

        if body_problems:
            quality_issues.append((fname, body_problems))
            # A file can be both valid (header OK) AND have quality issues;
            # still count it as valid so the summary is accurate.

        valid.append((fname, meta.get("category", "?"), meta.get("topic_name", "?")))

    # ── Report ─────────────────────────────────────────────────────────────────
    W = 80  # report width

    # ── Section 1 ─────────────────────────────────────────────────────────────
    print()
    print("═" * W)
    print("  SECTION 1 — FILES WITH VALID HEADERS")
    print("═" * W)
    if valid:
        for fname, category, topic_name in valid:
            print(f"  ✓  {fname:<45s}  [{category}]  {topic_name}")
    else:
        print("  (none)")

    # ── Section 2 ─────────────────────────────────────────────────────────────
    print()
    print("═" * W)
    print("  SECTION 2 — FILES WITH MISSING OR INCOMPLETE HEADERS")
    print("═" * W)
    if broken:
        for fname, problems in broken:
            for prob in problems:
                print(f"  ✗  {fname:<45s}  {prob}")
    else:
        print("  (none — all files have complete headers)")

    # ── Section 3 ─────────────────────────────────────────────────────────────
    print()
    print("═" * W)
    print("  SECTION 3 — FILES WITH BODY QUALITY ISSUES")
    print("═" * W)
    if quality_issues:
        for fname, problems in quality_issues:
            for prob in problems:
                print(f"  ⚠  {fname:<45s}  {prob}")
    else:
        print("  (none — all bodies passed quality checks)")

    # ── Section 4 — Summary ────────────────────────────────────────────────────
    total          = len(txt_files)
    n_valid        = len(valid)
    n_broken       = len(broken)
    n_quality      = len(quality_issues)

    print()
    print("═" * W)
    print("  SECTION 4 — SUMMARY")
    print("═" * W)
    print(f"  {'Total files scanned':<35s}: {total}")
    print(f"  {'Valid and ready':<35s}: {n_valid - n_quality}  "
          f"(header OK + no quality issues)")
    print(f"  {'Valid but has quality issues':<35s}: {n_quality}  "
          f"← need content improvement")
    print(f"  {'Missing/broken headers':<35s}: {n_broken}  "
          f"← need header fix")
    print(f"  {'Body quality issues total':<35s}: {n_quality}  "
          f"← need content fix")
    print(f"  {'Too short (< ' + str(MIN_BODY_CHARS) + ' chars)':<35s}: {too_short_count}  "
          f"← need more content")
    print("═" * W)
    print()


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    audit()
