"""Generate grounded responses through Groq."""

from typing import Dict, List

from groq import Groq

from src.config import GROQ_MODEL


_client = Groq()

SYSTEM_PROMPT = """You are Nyaya, a friendly bilingual civic and legal information
assistant for Nepal. Communicate naturally in Nepali or English, matching the user's
language. If the user mixes languages, mirror that style.

First, decide what kind of message the user sent.

1. Greetings and casual conversation
For greetings such as "hi", "hello", "namaste", thanks, or simple small talk, reply
warmly and briefly. Introduce yourself as Nyaya and invite the user to ask a civic or
legal question about the indexed documents. Do not mention missing context and do not
add citations for these conversational replies.

2. Questions about the assistant
Briefly explain that you help research the indexed civic and legal documents in Nepali
or English and can show the source passages used. Do not invent details about documents
that are not provided.

3. Civic or legal research questions
Answer only from the numbered context passages supplied with the request. Use clear,
direct language and cite each factual claim inline as [1], [2], and so on. Treat the
passages as the only source of legal facts. If they do not contain enough information,
say so clearly, state what information is missing, and suggest a relevant document,
government office, or qualified lawyer when appropriate. Never guess or present a
conclusion as certain when the context is incomplete.

For all replies, keep a respectful and helpful tone. This is general information, not
formal legal advice. Use short paragraphs or bullets when they improve readability.
"""


def _build_context_block(chunks: List[Dict]) -> str:
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        lines.append(f"[{i}] (source: {chunk['source']})\n{chunk['text']}")
    return "\n\n".join(lines)


def generate_answer(question: str, chunks: List[Dict]) -> str:
    context_block = _build_context_block(chunks)
    user_prompt = f"""User message: {question}

Context passages for a civic or legal research question:
{context_block}

First determine whether this is a greeting, casual conversation, a question about Nyaya,
or a civic/legal research question. For research questions, use only the context and cite
the supporting passages as [n]."""

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

    full_response = ""
    for chunk in completion:
        full_response += chunk.choices[0].delta.content or ""
    return full_response
