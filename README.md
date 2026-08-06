# Nyaya — Bilingual Civic & Legal RAG Assistant

A retrieval-augmented Q&A assistant over legal/civic documents (e.g. Nepal's
Constitution, Labor Act, Company Act), answering in **Nepali or English**
with cited sources.

- **Embeddings:** `embeddinggemma:300m` served locally via **Ollama** (runs fine on 16GB RAM / 2GB VRAM — CPU is enough for a 300M model)
- **Vector store:** ChromaDB (local, persistent)
- **Generation:** **Groq API** (fast, free-tier hosted LLM — no local GPU needed for generation)
- **UI:** Gradio

## 1. Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- A free [Groq API key](https://console.groq.com)

## 2. Pull the embedding model

```bash
ollama pull embeddinggemma:300m
```

## 3. Set up the project

```bash
git clone <this repo>
cd nyaya-rag
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and paste your GROQ_API_KEY
```

## 4. Add your documents

Drop `.pdf`, `.docx`, `.txt`, or `.md` files into:

```
data/raw/
```

## 5. Build the vector index

```bash
python build_index.py
# to rebuild from scratch later:
python build_index.py --reset
```

## 6. Launch the app

```bash
python app.py
```

Open the local URL Gradio prints (usually `http://127.0.0.1:7860`).

## Project structure

```
nyaya-rag/
├── data/raw/            # put source documents here
├── chroma_db/           # auto-created persistent vector store
├── src/
│   ├── config.py        # paths, model names, chunk/retrieval settings
│   ├── loaders.py        # pdf/docx/txt loaders
│   ├── chunking.py       # recursive text splitter
│   ├── embeddings.py     # embeddinggemma:300m wrapper (Ollama)
│   ├── vectorstore.py    # ChromaDB wrapper
│   ├── llm.py             # Groq generation + grounded system prompt
│   └── rag_pipeline.py    # orchestrates retrieval + generation
├── build_index.py        # CLI: ingest + chunk + embed + store
├── app.py                 # Gradio chat UI
├── requirements.txt
└── .env.example
```

## Notes on design choices

- **Query vs. document prefixes**: EmbeddingGemma is trained with task-specific
  prompt prefixes. `embeddings.py` applies `"task: search result | query: "` to
  queries and `"title: none | text: "` to document chunks — this measurably
  improves retrieval quality over embedding raw text.
- **Grounded generation**: `llm.py`'s system prompt forces the model to answer
  only from retrieved context and cite `[n]` sources, reducing hallucination —
  critical for a legal/civic-info use case.
- **Bilingual by design**: the assistant mirrors the language of the question
  (Nepali or English) rather than forcing one language.
