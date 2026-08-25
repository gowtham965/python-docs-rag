from pyrag.generation.prompt import build_prompt
from pyrag.models import Chunk, RetrievedChunk


def test_build_prompt_includes_question_and_citations():
    chunks = [
        RetrievedChunk(
            chunk=Chunk(id="a", text="dict() creates a dictionary.", source_file="f", section_title="Dictionaries"),
            score=0.9,
        )
    ]
    prompt = build_prompt("How do I create a dictionary?", chunks)
    assert "How do I create a dictionary?" in prompt
    assert "[Section: Dictionaries]" in prompt
    assert "dict() creates a dictionary." in prompt


def test_build_prompt_instructs_model_to_admit_uncertainty():
    prompt = build_prompt("irrelevant question", [])
    assert "I don't know based on the provided documentation." in prompt
