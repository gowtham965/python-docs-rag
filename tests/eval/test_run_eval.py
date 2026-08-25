import json

from pyrag.eval.run_eval import run_eval
from pyrag.models import Chunk, RetrievedChunk
from pyrag.retrieval.retriever import RetrievalResult
from pyrag.generation.pipeline import AnswerResult


class FakeRetriever:
    def retrieve(self, query):
        chunk = Chunk(id="a", text="t", source_file="library/stdtypes.rst", section_title="s")
        return RetrievalResult(chunks=[RetrievedChunk(chunk=chunk, score=0.9)], is_confident=True)


class FakePipeline:
    def answer(self, question):
        return AnswerResult(answer="an answer", sources=[], is_out_of_scope=False)


class FakeJudgeLLMClient:
    def generate(self, prompt):
        return '{"faithfulness": 5, "relevance": 5}'


def test_run_eval_writes_report_and_returns_rows(tmp_path):
    questions_path = tmp_path / "questions.json"
    questions_path.write_text(
        json.dumps(
            [
                {"question": "q1", "expected_source_file": "library/stdtypes.rst", "is_in_scope": True},
                {"question": "q2", "expected_source_file": None, "is_in_scope": False},
            ]
        )
    )
    report_path = tmp_path / "report.json"

    rows = run_eval(
        str(questions_path), FakeRetriever(), FakePipeline(), FakeJudgeLLMClient(), str(report_path)
    )

    assert len(rows) == 1
    assert rows[0].hit_at_5 is True
    assert rows[0].faithfulness == 5
    assert report_path.exists()
    saved = json.loads(report_path.read_text())
    assert len(saved) == 1


class FlakyRetriever:
    """Raises on the second call so we can simulate a mid-run failure."""

    def __init__(self):
        self.calls = 0

    def retrieve(self, query):
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("simulated retrieval/API failure")
        chunk = Chunk(id="a", text="t", source_file="library/stdtypes.rst", section_title="s")
        return RetrievalResult(chunks=[RetrievedChunk(chunk=chunk, score=0.9)], is_confident=True)


def test_run_eval_skips_failing_question_and_saves_partial_report(tmp_path):
    questions_path = tmp_path / "questions.json"
    questions_path.write_text(
        json.dumps(
            [
                {"question": "q1", "expected_source_file": "library/stdtypes.rst", "is_in_scope": True},
                {"question": "q2", "expected_source_file": "library/stdtypes.rst", "is_in_scope": True},
                {"question": "q3", "expected_source_file": "library/stdtypes.rst", "is_in_scope": True},
            ]
        )
    )
    report_path = tmp_path / "report.json"

    rows = run_eval(
        str(questions_path), FlakyRetriever(), FakePipeline(), FakeJudgeLLMClient(), str(report_path)
    )

    # run_eval must not raise, and must skip only the failing question.
    assert [r.question for r in rows] == ["q1", "q3"]

    # The report file reflects the partial (successful-only) results.
    saved = json.loads(report_path.read_text())
    assert [r["question"] for r in saved] == ["q1", "q3"]
