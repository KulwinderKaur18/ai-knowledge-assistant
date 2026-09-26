"""Answer orchestration: retrieve -> threshold check -> generate -> cite."""

from dataclasses import dataclass, field

from . import config
from .ingest import Chunk, load_corpus
from .llm import generate_extractive, generate_with_llm, llm_configured
from .retrieve import Retriever, ScoredChunk


@dataclass
class Source:
    doc_id: str
    title: str
    chunk_index: int
    score: float


@dataclass
class Answer:
    text: str
    sources: list[Source] = field(default_factory=list)
    answered: bool = False
    mode: str = "unknown"  # one of: "llm", "extractive", "unknown"

    def format(self) -> str:
        lines = [self.text]
        if self.sources:
            lines.append("\nSources:")
            for s in self.sources:
                lines.append(
                    f"- {s.doc_id} (chunk {s.chunk_index}) — relevance {s.score:.2f}"
                )
        return "\n".join(lines)


def _build_retriever() -> Retriever:
    return Retriever(load_corpus())


def answer_question(
    question: str,
    retriever: Retriever | None = None,
    top_k: int = config.TOP_K,
    threshold: float = config.SIMILARITY_THRESHOLD,
) -> Answer:
    """Answer a question over the corpus, with citations and honest abstention."""
    retriever = retriever or _build_retriever()
    hits = retriever.search(question, top_k=top_k)

    if not hits or hits[0].score < threshold:
        return Answer(
            text=config.DONT_KNOW_RESPONSE, sources=[], answered=False, mode="unknown"
        )

    sources = [
        Source(
            doc_id=h.chunk.doc_id,
            title=h.chunk.title,
            chunk_index=h.chunk.chunk_index,
            score=h.score,
        )
        for h in hits
    ]

    if llm_configured():
        try:
            text = generate_with_llm(question, hits)
            mode = "llm"
        except Exception as exc:  # network/API failure -> honest fallback
            text = (
                "[Extractive mode — the language-model API call failed "
                f"({exc}), so this answer quotes the retrieved passage directly.]\n\n"
                + generate_extractive(question, hits).split("\n\n", 1)[1]
            )
            mode = "extractive"
    else:
        text = generate_extractive(question, hits)
        mode = "extractive"

    return Answer(text=text, sources=sources, answered=True, mode=mode)
