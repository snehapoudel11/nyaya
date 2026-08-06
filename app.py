import gradio as gr

from src.rag_pipeline import RAGPipeline
from src.vectorstore import VectorStore

pipeline = RAGPipeline()


def format_sources(sources):
    if not sources:
        return "_No sources retrieved._"
    lines = [f"**[{s['n']}]** `{s['source']}`  — similarity: {s['score']}" for s in sources]
    return "\n\n".join(lines)


def respond(message, history):
    result = pipeline.query(message)
    history = history + [(message, result["answer"])]
    return history, format_sources(result["sources"]), ""


def get_kb_status():
    store = VectorStore()
    n = store.count()
    return f"📚 Knowledge base: **{n}** chunks indexed."


with gr.Blocks(title="Nyaya — Bilingual Civic & Legal RAG Assistant") as demo:
    gr.Markdown(
        """
        # ⚖️ Nyaya — Bilingual Civic & Legal Q&A Assistant
        Ask questions in **Nepali or English** about the indexed legal / civic documents.
        Answers are grounded in retrieved source passages — nothing is made up.

        Embeddings: `embeddinggemma:300m` (local, via Ollama) · Generation: Groq (`llama-3.3-70b-versatile`)
        """
    )

    kb_status = gr.Markdown(get_kb_status())

    chatbot = gr.Chatbot(height=450, label="Conversation")

    with gr.Row():
        msg = gr.Textbox(
            placeholder="e.g. नेपालमा कम्पनी दर्ता कसरी गर्ने? / How do I register a company in Nepal?",
            scale=4,
            label="Your question",
        )
        send_btn = gr.Button("Ask", variant="primary", scale=1)

    with gr.Accordion("📎 Sources for last answer", open=False):
        sources_box = gr.Markdown("_Ask a question to see sources here._")

    clear_btn = gr.Button("Clear conversation")

    send_btn.click(respond, inputs=[msg, chatbot], outputs=[chatbot, sources_box, msg])
    msg.submit(respond, inputs=[msg, chatbot], outputs=[chatbot, sources_box, msg])
    clear_btn.click(lambda: ([], "_Ask a question to see sources here._", ""), outputs=[chatbot, sources_box, msg])

if __name__ == "__main__":
    demo.launch()
