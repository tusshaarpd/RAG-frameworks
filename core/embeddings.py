"""Shared embedding model + FAISS helpers.

One local model for every arena: sentence-transformers all-MiniLM-L6-v2,
CPU only. Free, fast, no API key. The model is cached at module level so
five arenas embedding the same PDF don't reload weights five times.
"""
import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"
_model = None


def get_model():
    """Lazy singleton - sentence-transformers import is slow, do it once."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME, device="cpu")
    return _model


def embed(texts: list[str]) -> np.ndarray:
    """Embed and L2-normalize so FAISS inner product == cosine similarity."""
    vectors = get_model().encode(texts, show_progress_bar=False)
    vectors = np.asarray(vectors, dtype="float32")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-10, None)


def build_index(chunks: list[str]):
    """Flat (exact, brute-force) FAISS index over normalized vectors."""
    import faiss
    vectors = embed(chunks)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def search(index, chunks: list[str], query: str, k: int = 4):
    """Return [(chunk_text, cosine_score)] for the top-k chunks."""
    q = embed([query])
    scores, ids = index.search(q, min(k, len(chunks)))
    return [(chunks[i], float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]
