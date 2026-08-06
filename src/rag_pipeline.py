from typing import Dict

from src.config import TOP_K
from src.embeddings import embed_query
from src.vectorstore import VectorStore
from src.llm import generate_answer


class RAGPipeline:
    def __init__(self):
        self.store = VectorStore()

    def query(self, question: str, top_k: int = TOP_K) -> Dict:
        if self.store.count() == 0:
            return {
                "answer": "The knowledge base is empty. Run `build_index.py` after "
                          "adding documents to `data/raw/`.",
                "sources": [],
            }

        q_embedding = embed_query(question)
        chunks = self.store.query(q_embedding, top_k=top_k)

        if not chunks:
            return {
                "answer": "I couldn't find anything relevant in the documents.",
                "sources": [],
            }

        answer = generate_answer(question, chunks)

        sources = [
            {"n": i + 1, "source": c["source"], "score": round(c["score"], 3)}
            for i, c in enumerate(chunks)
        ]
        return {"answer": answer, "sources": sources}
