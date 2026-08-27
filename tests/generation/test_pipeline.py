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
        self.call_count = 0

    def generate(self, prompt):
        self.call_count += 1
        return self._response


class FakeStreamingLLMClient:
    def __init__(self, deltas, error=None):
        self._deltas = deltas
        self._error = error
        self.call_count = 0

    def generate_stream(self, prompt):
        self.call_count += 1
        for delta in self._deltas:
            yield delta
        if self._error:
            raise self._error


def test_pipeline_returns_out_of_scope_message_when_not_confident():
    retriever = FakeRetriever(RetrievalResult(chunks=[], is_confident=False))
    llm_client = FakeLLMClient("unused")
    pipeline = RagPipeline(retriever=retriever, llm_client=llm_client)

    result = pipeline.answer("unrelated question")

    assert result.is_out_of_scope is True
    assert result.answer == OUT_OF_SCOPE_MESSAGE
    assert result.sources == []
    assert llm_client.call_count == 0


def test_pipeline_returns_generated_answer_when_confident():
    chunk = RetrievedChunk(chunk=Chunk(id="a", text="t", source_file="f", section_title="s"), score=0.9)
    retriever = FakeRetriever(RetrievalResult(chunks=[chunk], is_confident=True))
    llm_client = FakeLLMClient("the real answer")
    pipeline = RagPipeline(retriever=retriever, llm_client=llm_client)

    result = pipeline.answer("a real question")

    assert result.is_out_of_scope is False
    assert result.answer == "the real answer"
    assert result.sources == [chunk]
    assert llm_client.call_count == 1


def test_answer_stream_yields_out_of_scope_message_without_calling_llm():
    retriever = FakeRetriever(RetrievalResult(chunks=[], is_confident=False))
    llm_client = FakeStreamingLLMClient(deltas=[])
    pipeline = RagPipeline(retriever=retriever, llm_client=llm_client)

    events = list(pipeline.answer_stream("unrelated question"))

    assert events == [
        {"type": "token", "text": OUT_OF_SCOPE_MESSAGE},
        {"type": "done", "sources": [], "is_out_of_scope": True},
    ]
    assert llm_client.call_count == 0


def test_answer_stream_yields_tokens_then_done_when_confident():
    chunk = RetrievedChunk(chunk=Chunk(id="a", text="t", source_file="f", section_title="s"), score=0.9)
    retriever = FakeRetriever(RetrievalResult(chunks=[chunk], is_confident=True))
    llm_client = FakeStreamingLLMClient(deltas=["Hel", "lo"])
    pipeline = RagPipeline(retriever=retriever, llm_client=llm_client)

    events = list(pipeline.answer_stream("a real question"))

    assert events == [
        {"type": "token", "text": "Hel"},
        {"type": "token", "text": "lo"},
        {"type": "done", "sources": [chunk], "is_out_of_scope": False},
    ]
    assert llm_client.call_count == 1


def test_answer_stream_yields_error_event_on_mid_stream_failure():
    chunk = RetrievedChunk(chunk=Chunk(id="a", text="t", source_file="f", section_title="s"), score=0.9)
    retriever = FakeRetriever(RetrievalResult(chunks=[chunk], is_confident=True))
    llm_client = FakeStreamingLLMClient(deltas=["partial"], error=RuntimeError("boom"))
    pipeline = RagPipeline(retriever=retriever, llm_client=llm_client)

    events = list(pipeline.answer_stream("a real question"))

    assert events == [
        {"type": "token", "text": "partial"},
        {"type": "error", "message": "boom"},
    ]


class FailingRetriever:
    def retrieve(self, query):
        raise RuntimeError("retrieval failed")


def test_answer_stream_yields_error_event_on_pre_stream_failure():
    retriever = FailingRetriever()
    llm_client = FakeStreamingLLMClient(deltas=[])
    pipeline = RagPipeline(retriever=retriever, llm_client=llm_client)

    events = list(pipeline.answer_stream("a question"))

    assert events == [{"type": "error", "message": "retrieval failed"}]
    assert llm_client.call_count == 0
