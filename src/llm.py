"""
Calls Groq's hosted LLM to generate the final answer, grounded in the
chunks retrieved from the local vector store.
"""

from typing import List, Dict

from groq import Groq

from src.config import GROQ_MODEL

_client = Groq()  # reads GROQ_API_KEY from the environment automatically

SYSTEM_PROMPT = """You are Nyaya, a bilingual (Nepali and English) civic and legal
information assistant. You answer ONLY using the numbered context passages
provided below. Rules:

1. Reply in the same language the user asked in (Nepali or English). If mixed, mirror it.
2. Every factual claim must be backed by the context. Cite sources inline like [1], [2].
3. If the answer is not in the context, say clearly that you don't have enough
   information in the provided documents — do NOT guess or use outside knowledge.
4. This is general informational content, not formal legal advice. If the question
   concerns a serious personal legal matter, gently suggest consulting a lawyer
   or the relevant government office.
5. Keep answers concise and well-structured.
"""


def _build_context_block(chunks: List[Dict]) -> str:
    lines = []
    for i, c in enumerate(chunks, start=1):
        lines.append(f"[{i}] (source: {c['source']})\n{c['text']}")
    return "\n\n".join(lines)


def generate_answer(question: str, chunks: List[Dict]) -> str:
    context_block = _build_context_block(chunks)

    user_prompt = f"""Context passages:
{context_block}

Question: {question}

Answer using only the context above, with [n] citations."""

    completion = _client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_completion_tokens=1024,
        top_p=1,
        reasoning_effort="medium",
        stream=True,
        stop=None,
    )

    # Collect the streamed chunks into one string (Gradio call site expects
    # a single string). Swap this loop for a generator if you want token-by-
    # token streaming in the UI instead.
    full_response = ""
    for chunk in completion:
        full_response += chunk.choices[0].delta.content or ""
    return full_response