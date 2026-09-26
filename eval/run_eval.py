"""Run the evaluation set and report pass/fail per question plus an overall score.

Pass criteria (heuristic, documented honestly):
- Answerable question: pipeline answered, every expected keyword appears in the
  answer text (case-insensitive), and at least one source is cited.
- Unanswerable question: pipeline abstained with the exact "don't know" response.

This measures retrieval + abstention behavior on a tiny synthetic set — it is
not a claim about production quality.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from assistant import config
from assistant.answer import answer_question
from assistant.retrieve import Retriever
from assistant.ingest import load_corpus


def evaluate(retriever: Retriever, questions: list[dict]) -> list[dict]:
    results = []
    for q in questions:
        answer = answer_question(q["question"], retriever=retriever)
        text_lower = answer.text.lower()
        if q["answerable"]:
            keywords_ok = all(kw.lower() in text_lower for kw in q["expected_keywords"])
            doc_ok = any(s.doc_id == q["expected_doc"] for s in answer.sources)
            passed = answer.answered and keywords_ok and doc_ok
            reason = (
                f"answered={answer.answered} keywords_ok={keywords_ok} "
                f"expected_doc_cited={doc_ok} mode={answer.mode}"
            )
        else:
            passed = (not answer.answered) and (
                config.DONT_KNOW_RESPONSE.lower() in text_lower
            )
            reason = f"abstained={not answer.answered}"
        results.append(
            {"question": q["question"], "passed": passed, "reason": reason}
        )
    return results


def main() -> int:
    eval_dir = Path(__file__).resolve().parent
    questions = json.loads((eval_dir / "questions.json").read_text(encoding="utf-8"))
    retriever = Retriever(load_corpus())

    results = evaluate(retriever, questions)
    passed = sum(1 for r in results if r["passed"])
    total = len(results)

    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"[{status}] {r['question']}\n       {r['reason']}")
    print(f"\nScore: {passed}/{total} ({100 * passed / total:.0f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
