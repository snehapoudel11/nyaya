"""
Embedding wrapper for embeddinggemma-300m, loaded directly via
sentence-transformers and forced onto CPU.

This bypasses Ollama entirely (no server, no GPU auto-detection),
which sidesteps the CUDA/PTX crash some Windows + old-GPU setups hit.
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