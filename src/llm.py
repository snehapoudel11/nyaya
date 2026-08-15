"""Generate grounded responses through Groq."""

from typing import Dict, List

from groq import Groq

from src.config import GROQ_MODEL


_client = Groq()

SYSTEM_PROMPT = """You are Nyaya, a friendly bilingual civic and legal information
assistant for Nepal.

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


def _response_language(question: str) -> str:
    """Choose the reply language from the script used in the question.

    Nepali is normally written in the Devanagari Unicode block. Questions without
    Devanagari characters are treated as English, which keeps the behaviour
    predictable for English-language queries.
    """
    if any("\u0900" <= character <= "\u097f" for character in question):
        return "Nepali"
    return "English"


def _build_context_block(chunks: List[Dict]) -> str:
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        lines.append(f"[{i}] (source: {chunk['source']})\n{chunk['text']}")
    return "\n\n".join(lines)


def generate_answer(question: str, chunks: List[Dict]) -> str:
    context_block = _build_context_block(chunks)
    response_language = _response_language(question)
    user_prompt = f"""User message: {question}

Required answer language: {response_language}. Write the entire answer in
{response_language}. This requirement takes priority over the language of the context
passages. Do not switch languages unless quoting a source title or a necessary legal
term.

Context passages for a civic or legal research question:
{context_block}

First determine whether this is a greeting, casual conversation, a question about Nyaya,
or a civic/legal research question. For research questions, use only the context and cite
the supporting passages as [n]."""

    base_kwargs = dict(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        top_p=1,
        stream=True,
        stop=None,
    )

    # Try several common token-parameter names and optional `reasoning_effort`
    # to handle SDK differences across versions.
    token_param_variants = [
        {"max_output_tokens": 1024, "reasoning_effort": "medium"},
        {"max_completion_tokens": 1024, "reasoning_effort": "medium"},
        {"max_tokens": 1024, "reasoning_effort": "medium"},
        {"max_new_tokens": 1024, "reasoning_effort": "medium"},
        {"max_output_tokens": 1024},
        {"max_completion_tokens": 1024},
        {"max_tokens": 1024},
        {"max_new_tokens": 1024},
        {},
    ]

    last_exc = None
    completion = None
    for variant in token_param_variants:
        try:
            kwargs = dict(base_kwargs)
            kwargs.update(variant)
            completion = _client.chat.completions.create(**kwargs)
            last_exc = None
            break
        except TypeError as e:
            last_exc = e
            continue
    if completion is None and last_exc is not None:
        raise last_exc

    full_response = ""
    for chunk in completion:
        full_response += chunk.choices[0].delta.content or ""
    return full_response
