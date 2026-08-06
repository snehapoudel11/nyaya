"""
Thin wrapper around a persistent ChromaDB collection.
"""

from typing import List, Dict

import chromadb

from src.config import CHROMA_DIR, COLLECTION_NAME


class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: List[Dict], embeddings: List[List[float]]):
        self.collection.add(
            ids=[c["id"] for c in chunks],
            embeddings=embeddings,
            documents=[c["text"] for c in chunks],
            metadatas=[{"source": c["source"], "chunk_index": c["chunk_index"]} for c in chunks],
        )

    def count(self) -> int:
        return self.collection.count()

    def reset(self):
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def query(self, query_embedding: List[float], top_k: int) -> List[Dict]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )
        hits = []
        for text, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append({
                "text": text,
                "source": meta["source"],
                "chunk_index": meta["chunk_index"],
                "score": 1 - dist,  # cosine distance -> similarity
            })
        return hits
