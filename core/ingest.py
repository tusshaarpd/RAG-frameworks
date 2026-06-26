"""PDF extraction and chunking - the 'no framework' ingestion path.

Arenas that let us inject our own ingestion (raw, LangGraph, CrewAI) use this.
LangChain and LlamaIndex insist on their own loaders - we let them, because
that difference IS part of the demo.
"""
from pypdf import PdfReader

MAX_PAGES = 20        # demo cap: keeps latency and token cost sane
CHUNK_SIZE = 800      # characters, not tokens - simple on purpose
CHUNK_OVERLAP = 100


def extract_text(pdf_path: str) -> tuple[str, int, bool]:
    """Return (full_text, pages_used, was_truncated)."""
    reader = PdfReader(pdf_path)
    pages = reader.pages[:MAX_PAGES]
    truncated = len(reader.pages) > MAX_PAGES
    text = "\n".join(page.extract_text() or "" for page in pages)
    return text, len(pages), truncated


def chunk_text(text: str, size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Fixed-size sliding window chunking. Deliberately naive: this is the
    baseline every framework's fancy splitter gets compared against."""
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start + size].strip()
        if chunk:
            chunks.append(chunk)
        start += size - overlap
    return chunks
