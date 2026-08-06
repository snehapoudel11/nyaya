"""
Run this whenever you add/change documents in data/raw/.

Usage:
    python build_index.py            # incremental add
    python build_index.py --reset    # wipe and rebuild from scratch
"""

import argparse

from src.config import DATA_DIR
from src.loaders import load_documents
from src.chunking import chunk_document
from src.embeddings import embed_documents
from src.vectorstore import VectorStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Wipe the collection before indexing")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    store = VectorStore()
    if args.reset:
        print("Resetting collection...")
        store.reset()

    print(f"Loading documents from {DATA_DIR} ...")
    docs = load_documents(DATA_DIR)
    if not docs:
        print("No documents found. Add .pdf / .docx / .txt / .md files to data/raw/ and rerun.")
        return

    all_chunks = []
    for doc in docs:
        chunks = chunk_document(doc["source"], doc["text"])
        all_chunks.extend(chunks)
        print(f"  {doc['source']}: {len(chunks)} chunks")

    print(f"Total chunks: {len(all_chunks)}")
    print("Embedding with embeddinggemma:300m (via Ollama) ...")

    batch_size = args.batch_size
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        texts = [c["text"] for c in batch]
        embeddings = embed_documents(texts)
        store.add_chunks(batch, embeddings)
        print(f"  Indexed {min(i + batch_size, len(all_chunks))}/{len(all_chunks)}")

    print(f"Done. Collection now has {store.count()} chunks.")


if __name__ == "__main__":
    main()
