"""Pluggable answer generation: optional LLM, honest extractive fallback.

When OPENAI_API_KEY is set, answers are generated with an OpenAI-compatible
chat-completions API (stdlib urllib only — no extra dependency). Otherwise the
pipeline uses a clearly-labeled extractive fallback that quotes the retrieved
passages. The fallback is never presented as LLM output.
"""

import json
import urllib.request

from . import config
from .retrieve import ScoredChunk


def llm_configured() -> bool:
    return bool(config.OPENAI_API_KEY)


def _build_prompt(question: str, hits: list[ScoredChunk]) -> tuple[str, str]:
    context = "\n\n".join(
        f"[{h.chunk.doc_id} :: chunk {h.chunk.chunk_index}]\n{h.chunk.text}"
        for h in hits
    )
    system = (
        "You answer questions using ONLY the provided document excerpts. "
        "If the excerpts do not contain the answer, reply exactly: "
        f'"{config.DONT_KNOW_RESPONSE}" Keep answers concise (2-4 sentences).'
    )
    user = f"Documents:\n{context}\n\nQuestion: {question}"
    return system, user


def generate_with_llm(question: str, hits: list[ScoredChunk]) -> str:
    """Generate an answer via an OpenAI-compatible API. Raises on failure."""
    system, user = _build_prompt(question, hits)
    payload = json.dumps(
        {
            "model": config.OPENAI_MODEL,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{config.OPENAI_BASE_URL.rstrip('/')}/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.OPENAI_API_KEY}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"].strip()


def generate_extractive(question: str, hits: list[ScoredChunk]) -> str:
    """Honest fallback: quote the most relevant passage; never claim to be an LLM."""
    top = hits[0]
    passage = top.chunk.text
    if len(passage) > 600:
        passage = passage[:600].rsplit(" ", 1)[0] + "…"
    return (
        "[Extractive mode — no language-model API key is configured, so this "
        "answer quotes the most relevant retrieved passage directly.]\n\n"
        f'Relevant passage from "{top.chunk.title}":\n"{passage}"'
    )
