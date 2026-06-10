#!/usr/bin/env python3
"""
step1_prepare_data.py
─────────────────────
Reads raw Wikipedia .txt files from data/raw/, parses the metadata header,
detects section headings, chunks text into ~400-char pieces with 80-char
overlap (at sentence boundaries), and saves everything to data/chunks/all_chunks.json.
"""

import json
import os
import re
from collections import defaultdict

# ── paths ─────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
RAW_DIR    = os.path.join(ROOT_DIR, "data", "raw")
CHUNKS_DIR = os.path.join(ROOT_DIR, "data", "chunks")

os.makedirs(CHUNKS_DIR, exist_ok=True)

CHUNK_SIZE = 600       # target characters per chunk
CHUNK_OVERLAP = 120    # overlap characters between consecutive chunks

# Sections to skip (Wikipedia boilerplate)
SKIP_SECTIONS = {
    "see also", "references", "sources", "further reading",
    "external links", "notes", "bibliography", "footnotes",
    "citations",
}


# ─────────────────────────────────────────────────────────────────────────
# 1. Parse header
# ─────────────────────────────────────────────────────────────────────────
def parse_header(text: str) -> tuple[dict, str]:
    """
    Split the file into metadata dict and body text.
    The header ends at the first '---' separator line.
    """
    parts = text.split("---", 1)
    header_text = parts[0]
    body = parts[1].strip() if len(parts) > 1 else ""

    meta = {}
    for line in header_text.strip().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()

    return meta, body


# ─────────────────────────────────────────────────────────────────────────
# 2. Detect sections
# ─────────────────────────────────────────────────────────────────────────
def is_section_heading(line: str, prev_blank: bool, next_blank: bool) -> bool:
    """
    Heuristic: a line is a section heading if it is short, starts with
    an uppercase letter, is surrounded by blank-ish context, and doesn't
    look like a regular sentence (no trailing period).
    """
    stripped = line.strip()
    if not stripped:
        return False
    if len(stripped) > 80:
        return False
    if not stripped[0].isupper():
        return False
    if stripped.endswith("."):
        return False
    # Must have a blank line before or after (or be the very first line)
    if not (prev_blank or next_blank):
        return False
    # Reject lines that look like list items or sentences
    if stripped.startswith(("It ", "He ", "She ", "The ", "A ", "In ", "On ", "At ")):
        if len(stripped) > 50:
            return False
    return True


def split_into_sections(body: str) -> list[tuple[str, str]]:
    """
    Returns [(section_name, section_text), ...].
    The first block before any heading gets section name "Introduction".
    """
    lines = body.splitlines()
    sections: list[tuple[str, str]] = []
    current_section = "Introduction"
    current_lines: list[str] = []

    for i, line in enumerate(lines):
        prev_blank = (i == 0) or (lines[i - 1].strip() == "")
        next_blank = (i == len(lines) - 1) or (lines[i + 1].strip() == "")

        if is_section_heading(line, prev_blank, next_blank):
            # Save previous section
            text = "\n".join(current_lines).strip()
            if text and current_section.lower() not in SKIP_SECTIONS:
                sections.append((current_section, text))
            current_section = line.strip()
            current_lines = []
        else:
            current_lines.append(line)

    # Last section
    text = "\n".join(current_lines).strip()
    if text and current_section.lower() not in SKIP_SECTIONS:
        sections.append((current_section, text))

    return sections


# ─────────────────────────────────────────────────────────────────────────
# 3. Chunk text at sentence boundaries
# ─────────────────────────────────────────────────────────────────────────
_SENTENCE_RE = re.compile(
    r'(?<=[.!?])\s+(?=[A-Z"\'\(])'      # split after .!? followed by space + capital
)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences using regex."""
    sentences = _SENTENCE_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Break text into chunks of ~chunk_size characters with ~overlap characters
    of overlap, splitting at sentence boundaries.
    """
    # Normalise whitespace but keep paragraph breaks
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    sentences = split_sentences(text)
    if not sentences:
        return [text.strip()] if text.strip() else []

    chunks: list[str] = []
    current_chunk: list[str] = []
    current_len = 0

    for sentence in sentences:
        sent_len = len(sentence)

        # If a single sentence exceeds chunk_size, add it as its own chunk
        if sent_len > chunk_size and not current_chunk:
            chunks.append(sentence)
            continue

        if current_len + sent_len + 1 > chunk_size and current_chunk:
            # Save current chunk
            chunks.append(" ".join(current_chunk))

            # Build overlap: walk backwards until we have ~overlap chars
            overlap_sentences: list[str] = []
            overlap_len = 0
            for s in reversed(current_chunk):
                if overlap_len + len(s) > overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_len += len(s) + 1

            current_chunk = overlap_sentences
            current_len = sum(len(s) for s in current_chunk) + max(0, len(current_chunk) - 1)

        current_chunk.append(sentence)
        current_len += sent_len + (1 if current_len > 0 else 0)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# ─────────────────────────────────────────────────────────────────────────
# 4. Main pipeline
# ─────────────────────────────────────────────────────────────────────────
def _sanitize_for_id(text: str) -> str:
    """Convert text into a safe, lowercase, underscore-separated ID fragment."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)   # replace non-alphanum with _
    text = re.sub(r"_+", "_", text)            # collapse multiple _
    return text.strip("_")


def main():
    all_chunks: list[dict] = []
    category_counts: dict[str, int] = defaultdict(int)
    file_count = 0
    section_count = 0
    global_idx = 0         # global counter to guarantee unique IDs

    raw_files = sorted(
        f for f in os.listdir(RAW_DIR)
        if f.endswith(".txt") and "TEMPLATE" not in f
    )

    for fname in raw_files:
        filepath = os.path.join(RAW_DIR, fname)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        meta, body = parse_header(content)
        if not body:
            print(f"  ⚠  Skipping {fname} (empty body)")
            continue

        topic_id = meta.get("topic_id", os.path.splitext(fname)[0])
        topic_name = meta.get("topic_name", topic_id.replace("_", " ").title())
        category = meta.get("category", "unknown")
        period = meta.get("period", "unknown")
        location = meta.get("location", "unknown")

        sections = split_into_sections(body)
        file_count += 1
        section_count += len(sections)

        for section_name, section_text in sections:
            chunks = chunk_text(section_text)
            section_slug = _sanitize_for_id(section_name)

            for idx, chunk_text_str in enumerate(chunks):
                chunk_id = f"{topic_id}__{section_slug}__{idx:03d}__{global_idx:05d}"
                global_idx += 1
                # Searchable field: combines topic + metadata + section + text
                searchable = (
                    f"Topic: {topic_name}. "
                    f"Category: {category}. "
                    f"Period: {period}. "
                    f"Location: {location}. "
                    f"Section: {section_name}. "
                    f"{chunk_text_str}"
                )

                chunk_record = {
                    "chunk_id": chunk_id,
                    "topic_name": topic_name,
                    "topic_id": topic_id,
                    "category": category,
                    "period": period,
                    "location": location,
                    "section": section_name,
                    "text": chunk_text_str,
                    "searchable": searchable,
                }
                all_chunks.append(chunk_record)
                category_counts[category] += 1

    # ── Save ──────────────────────────────────────────────────────────
    output_path = os.path.join(CHUNKS_DIR, "all_chunks.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    # ── Summary ───────────────────────────────────────────────────────
    print("=" * 60)
    print("  STEP 1 — DATA PREPARATION SUMMARY")
    print("=" * 60)
    print(f"  Files processed      : {file_count}")
    print(f"  Sections detected    : {section_count}")
    print(f"  Total chunks created : {len(all_chunks)}")
    print(f"  Output saved to      : {output_path}")
    print("-" * 60)
    print("  Chunks per category:")
    for cat in sorted(category_counts):
        bar = "█" * (category_counts[cat] // 5)
        print(f"    {cat:<20s} : {category_counts[cat]:>5d}  {bar}")
    print("-" * 60)

    # Show a sample chunk
    if all_chunks:
        print("\n  📄 Sample chunk:")
        sample = all_chunks[0]
        for k, v in sample.items():
            display = v if len(str(v)) < 100 else str(v)[:97] + "..."
            print(f"    {k:<14s}: {display}")

    print("=" * 60)


if __name__ == "__main__":
    main()
