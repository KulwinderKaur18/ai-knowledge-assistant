"""CLI: python -m assistant "your question here\""""

import argparse
import json

from . import config
from .answer import answer_question


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask a question over the local document corpus."
    )
    parser.add_argument("question", help="The question to answer.")
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    parser.add_argument("--threshold", type=float, default=config.SIMILARITY_THRESHOLD)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args()

    answer = answer_question(args.question, top_k=args.top_k, threshold=args.threshold)

    if args.json:
        print(
            json.dumps(
                {
                    "answered": answer.answered,
                    "mode": answer.mode,
                    "text": answer.text,
                    "sources": [
                        {
                            "doc_id": s.doc_id,
                            "title": s.title,
                            "chunk_index": s.chunk_index,
                            "score": round(s.score, 3),
                        }
                        for s in answer.sources
                    ],
                },
                indent=2,
            )
        )
    else:
        print(answer.format())


if __name__ == "__main__":
    main()
