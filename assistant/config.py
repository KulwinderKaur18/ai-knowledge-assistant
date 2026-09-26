"""Configuration for the AI knowledge assistant."""

import os
from pathlib import Path

# Directory containing the document corpus (overridable via env var).
_corpus_env = os.environ.get("ASSISTANT_CORPUS_DIR", "")
CORPUS_DIR = (
    Path(_corpus_env)
    if _corpus_env
    else Path(__file__).resolve().parent.parent / "corpus"
)

# Retrieval settings.
TOP_K = int(os.environ.get("ASSISTANT_TOP_K", "3"))

# Minimum cosine similarity for the top hit before we answer.
# Below this, we honestly say we don't know. Tuned against eval/questions.json;
# raise it to be more conservative, lower it to be more permissive.
SIMILARITY_THRESHOLD = float(os.environ.get("ASSISTANT_THRESHOLD", "0.18"))

# Chunking settings (word counts).
CHUNK_TARGET_WORDS = 220
CHUNK_MAX_WORDS = 320
CHUNK_MIN_WORDS = 120

# Optional LLM settings (OpenAI-compatible API). When OPENAI_API_KEY is not
# set, the assistant runs in extractive fallback mode instead.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# Exact "don't know" response — the pipeline must never invent answers.
DONT_KNOW_RESPONSE = "I don't know based on the provided documents."
