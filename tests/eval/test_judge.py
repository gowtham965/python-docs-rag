from pyrag.eval.judge import judge_answer, build_judge_prompt
from pyrag.models import Chunk, RetrievedChunk


class FakeLLMClient:
    def __init__(self, response):
        self._response = response

    def generate(self, prompt):
        return self._response


def test_build_judge_prompt_includes_question_answer_and_context():
    chunks = [
        RetrievedChunk(
            chunk=Chunk(id="a", text="dict maps keys to values", source_file="f", section_title="s"),
            score=0.9,
        )
    ]
    prompt = build_judge_prompt("What is a dict?", "A dict maps keys to values.", chunks)
    assert "What is a dict?" in prompt
    assert "A dict maps keys to values." in prompt
    assert "dict maps keys to values" in prompt


def test_judge_answer_parses_json_score():
    llm_client = FakeLLMClient('{"faithfulness": 5, "relevance": 4}')
    score = judge_answer("q", "a", [], llm_client)
    assert score.faithfulness == 5
    assert score.relevance == 4
