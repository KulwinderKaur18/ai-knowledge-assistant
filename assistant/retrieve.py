"""TF-IDF retrieval over document chunks (no external services needed)."""

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .ingest import Chunk


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float  # cosine similarity in [0, 1]


class Retriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), sublinear_tf=True
        )
        self.matrix = self.vectorizer.fit_transform([c.text for c in chunks])

    def search(self, query: str, top_k: int = 3) -> list[ScoredChunk]:
        """Return the top-k chunks ranked by cosine similarity to the query."""
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix).flatten()
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results = [
            ScoredChunk(chunk=self.chunks[i], score=float(scores[i]))
            for i in ranked[:top_k]
            if scores[i] > 0
        ]
        return results
