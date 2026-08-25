from pyrag.generation.pipeline import RagPipeline, OUT_OF_SCOPE_MESSAGE
from pyrag.retrieval.retriever import RetrievalResult
from pyrag.models import Chunk, RetrievedChunk


class FakeRetriever:
    def __init__(self, result):
        self._result = result

    def retrieve(self, query):
        return self._result


class FakeLLMClient:
    def __init__(self, response):
        self._response = response

    def generate(self, prompt):
        return self._response


def test_pipeline_returns_out_of_scope_message_when_not_confident():
    retriever = FakeRetriever(RetrievalResult(chunks=[], is_confident=False))
    pipeline = RagPipeline(retriever=retriever, llm_client=FakeLLMClient("unused"))

    result = pipeline.answer("unrelated question")

    assert result.is_out_of_scope is True
    assert result.answer == OUT_OF_SCOPE_MESSAGE
    assert result.sources == []


def test_pipeline_returns_generated_answer_when_confident():
    chunk = RetrievedChunk(chunk=Chunk(id="a", text="t", source_file="f", section_title="s"), score=0.9)
    retriever = FakeRetriever(RetrievalResult(chunks=[chunk], is_confident=True))
    pipeline = RagPipeline(retriever=retriever, llm_client=FakeLLMClient("the real answer"))

    result = pipeline.answer("a real question")

    assert result.is_out_of_scope is False
    assert result.answer == "the real answer"
    assert result.sources == [chunk]
