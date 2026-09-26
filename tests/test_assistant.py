"""Tests for retrieval, abstention, and citations."""

import subprocess
import sys
from pathlib import Path

import pytest

from assistant import config
from assistant.answer import answer_question
from assistant.ingest import load_corpus
from assistant.retrieve import Retriever

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def retriever() -> Retriever:
    return Retriever(load_corpus())


def test_corpus_loads_chunks():
    chunks = load_corpus()
    assert len(chunks) >= 4
    assert {c.doc_id for c in chunks} == {
        "employee-handbook.md",
        "benefits-faq.md",
        "it-onboarding.md",
        "office-safety.md",
    }


def test_retrieval_returns_right_document(retriever):
    hits = retriever.search("What are the standard working hours?", top_k=3)
    assert hits, "expected at least one retrieval hit"
    assert hits[0].chunk.doc_id == "employee-handbook.md"


def test_retrieval_vacation_question(retriever):
    hits = retriever.search("How many vacation days do new employees get?", top_k=3)
    assert hits and hits[0].chunk.doc_id == "benefits-faq.md"


def test_unanswerable_triggers_dont_know(retriever):
    answer = answer_question(
        "Who won the 2022 FIFA World Cup?", retriever=retriever
    )
    assert not answer.answered
    assert answer.text == config.DONT_KNOW_RESPONSE
    assert answer.sources == []


def test_unanswerable_stock_price(retriever):
    answer = answer_question(
        "What is the current stock price?", retriever=retriever
    )
    assert not answer.answered
    assert "don't know" in answer.text.lower()


def test_answerable_includes_citations(retriever):
    answer = answer_question(
        "When are fire drills held?", retriever=retriever
    )
    assert answer.answered
    assert answer.sources, "answerable answers must cite sources"
    assert answer.sources[0].doc_id == "office-safety.md"
    assert "Sources:" in answer.format()


def test_extractive_mode_is_honestly_labeled(retriever, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    # config reads env at import; force fallback path by patching the check
    import assistant.answer as answer_mod

    monkeypatch.setattr(answer_mod, "llm_configured", lambda: False)
    answer = answer_question("What is the password policy?", retriever=retriever)
    assert answer.mode == "extractive"
    assert "extractive" in answer.text.lower()
    assert "no language-model api key" in answer.text.lower()


def test_eval_script_executes():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "eval" / "run_eval.py")],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Score:" in proc.stdout
