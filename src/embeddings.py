"""
Embedding wrapper for embeddinggemma-300m, loaded directly via
sentence-transformers and forced onto CPU.

It runs locally without an embedding server and is forced onto CPU,
which sidesteps CUDA/PTX crashes on some Windows and older GPU setups.
"""

from typing import List

from sentence_transformers import SentenceTransformer

# device="cpu" is explicit and non-negotiable here -- this is what fixes
# the crash. Loaded once at import time and reused for every call.
_model = SentenceTransformer("google/embeddinggemma-300m", device="cpu")


def embed_documents(texts: List[str]) -> List[List[float]]:
    """Embed a batch of document chunks (for indexing)."""
    embeddings = _model.encode_document(texts, convert_to_numpy=True)
    return embeddings.tolist()


def embed_query(text: str) -> List[float]:
    """Embed a single user query (for retrieval)."""
    embedding = _model.encode_query(text, convert_to_numpy=True)
    return embedding.tolist()
