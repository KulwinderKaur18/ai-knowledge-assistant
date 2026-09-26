"""Document ingestion and chunking."""

from dataclasses import dataclass
from pathlib import Path

from . import config

# Files that are documentation about the corpus, not corpus content.
SKIP_FILES = {"DATA_NOTE.md"}


@dataclass
class Chunk:
    """A chunk of text with its provenance."""

    doc_id: str  # filename, e.g. "employee-handbook.md"
    title: str  # document title from its first heading
    chunk_index: int  # 0-based index within the document
    text: str


def _clean_lines(text: str) -> list[str]:
    """Drop synthetic-data notice blockquotes so they don't pollute retrieval."""
    return [line for line in text.splitlines() if not line.lstrip().startswith(">")]


def _extract_title(lines: list[str], fallback: str) -> str:
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return fallback


def _split_sections(lines: list[str]) -> list[str]:
    """Split a document into sections on '## ' headings; each section is a chunk.

    Section-level chunks keep retrieval focused: a question about one policy
    matches the single section that answers it instead of a whole-document
    blend. The heading text is kept so it contributes to matching.
    """
    sections: list[str] = []
    current: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# ") or not stripped:
            continue  # title handled separately; blanks are just separators
        if stripped.startswith("## "):
            if current:
                sections.append(" ".join(current))
            current = [stripped[3:].strip()]
        elif current:
            current.append(stripped)
    if current:
        sections.append(" ".join(current))
    return [s for s in sections if len(s.split()) >= 8]


def load_corpus(corpus_dir: Path | str = config.CORPUS_DIR) -> list[Chunk]:
    """Load .md/.txt documents from a directory and return their chunks."""
    corpus_dir = Path(corpus_dir)
    chunks: list[Chunk] = []
    for path in sorted(corpus_dir.glob("*.md")) + sorted(corpus_dir.glob("*.txt")):
        if path.name in SKIP_FILES:
            continue
        lines = _clean_lines(path.read_text(encoding="utf-8"))
        title = _extract_title(lines, path.stem)
        for i, text in enumerate(_split_sections(lines)):
            chunks.append(
                Chunk(doc_id=path.name, title=title, chunk_index=i, text=text)
            )
    if not chunks:
        raise FileNotFoundError(f"No .md/.txt documents found in {corpus_dir}")
    return chunks
