import gradio as gr

from src.rag_pipeline import RAGPipeline
from src.vectorstore import VectorStore


pipeline = RAGPipeline()


def format_sources(sources):
    if not sources:
        return "No sources were retrieved for this answer."
    lines = [
        f"**[{source['n']}]** `{source['source']}`  <br>Similarity: {source['score']}"
        for source in sources
    ]
    return "\n\n".join(lines)


def respond(message, history):
    if not message or not message.strip():
        return history, "Ask a question to view supporting sources.", ""

    result = pipeline.query(message)
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": result["answer"]},
    ]
    return history, format_sources(result["sources"]), ""


def get_kb_status():
    store = VectorStore()
    return f"Knowledge base: **{store.count()}** chunks indexed"


with gr.Blocks(
    title="Nyaya | Civic and Legal Research",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate", neutral_hue="slate"),
    css="""
    .gradio-container { max-width: 1120px !important; }
    #hero { margin: 1.5rem 0 0.5rem; }
    #hero h1 { margin-bottom: 0.35rem; }
    #kb-status { border: 1px solid var(--border-color-primary); border-radius: 10px; padding: 0.65rem 0.9rem; }
    #chat { border: 1px solid var(--border-color-primary); border-radius: 12px; overflow: hidden; }
    """,
) as demo:
    gr.Markdown(
        """
        # Nyaya
        ### Bilingual civic and legal research assistant
        Ask questions in Nepali or English about the indexed documents. Answers are based on retrieved passages from the knowledge base.
        """,
        elem_id="hero",
    )

    gr.Markdown(get_kb_status(), elem_id="kb-status")

    chatbot = gr.Chatbot(
        height=500,
        label="Conversation",
        placeholder="Your conversation will appear here.",
        elem_id="chat",
    )

    with gr.Row():
        msg = gr.Textbox(
            placeholder="Ask a legal or civic question in Nepali or English",
            scale=4,
            label="Your question",
        )
        send_btn = gr.Button("Ask", variant="primary", scale=1)

    with gr.Accordion("Sources for the latest answer", open=False):
        sources_box = gr.Markdown("Ask a question to view supporting sources.")

    clear_btn = gr.Button("Clear conversation")

    send_btn.click(respond, inputs=[msg, chatbot], outputs=[chatbot, sources_box, msg])
    msg.submit(respond, inputs=[msg, chatbot], outputs=[chatbot, sources_box, msg])
    clear_btn.click(
        lambda: ([], "Ask a question to view supporting sources.", ""),
        outputs=[chatbot, sources_box, msg],
    )

if __name__ == "__main__":
    demo.launch()
