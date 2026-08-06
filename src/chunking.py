"""
Simple recursive character text splitter.
No heavy dependency needed — good enough for legal/civic documents
which are mostly clean paragraph text.
"""

from typing import List

from src.config import CHUNK_SIZE, CHUNK_OVERLAP

SEPARATORS = ["\n\n", "\n", ". ", "। ", " "]  # "।" = Devanagari full stop


def _split_on_separator(text: str, sep: str) -> List[str]:
    return [p for p in text.split(sep) if p.strip()]


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    if len(text) <= chunk_size:
        return [text.strip()]

    # Try separators in order until we get reasonably sized pieces
    pieces = [text]
    for sep in SEPARATORS:
        new_pieces = []
        for piece in pieces:
            if len(piece) <= chunk_size:
                new_pieces.append(piece)
            else:
                new_pieces.extend(_split_on_separator(piece, sep))
        pieces = new_pieces
        if all(len(p) <= chunk_size for p in pieces):
            break

    # Merge small pieces back together up to chunk_size, with overlap
    chunks = []
    current = ""
    for piece in pieces:
        candidate = (current + " " + piece).strip() if current else piece
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            current = piece

    if current:
        chunks.append(current.strip())

    # Add overlap between consecutive chunks
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-overlap:]
            overlapped.append((tail + " " + chunks[i]).strip())
        chunks = overlapped

    return [c for c in chunks if c.strip()]


def chunk_document(source: str, text: str) -> List[dict]:
    """
    Returns [{"id": "labor_act.pdf::0", "text": "...", "source": "labor_act.pdf", "chunk_index": 0}, ...]
    """
    raw_chunks = chunk_text(text)
    result = []
    for i, c in enumerate(raw_chunks):
        result.append({
            "id": f"{source}::{i}",
            "text": c,
            "source": source,
            "chunk_index": i,
        })
    return result
