"""AI knowledge assistant: retrieval-augmented Q&A over a local document set."""

from .answer import Answer, answer_question
from .ingest import Chunk, load_corpus
from .retrieve import Retriever

__all__ = ["Answer", "answer_question", "Chunk", "load_corpus", "Retriever"]
