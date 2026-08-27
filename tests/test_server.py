from fastapi.testclient import TestClient

from pyrag.server import app, get_pipeline


class FakePipeline:
    def __init__(self, events):
        self._events = events

    def answer_stream(self, question):
        yield from self._events


def test_chat_streams_tokens_and_done_event():
    app.dependency_overrides[get_pipeline] = lambda: FakePipeline(
        [
            {"type": "token", "text": "Hel"},
            {"type": "token", "text": "lo"},
            {"type": "done", "sources": [], "is_out_of_scope": False},
        ]
    )
    client = TestClient(app)

    response = client.post("/chat", json={"question": "a question"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert 'data: {"type": "token", "text": "Hel"}' in response.text
    assert 'data: {"type": "token", "text": "lo"}' in response.text
    assert '"type": "done"' in response.text
    assert '"is_out_of_scope": false' in response.text

    app.dependency_overrides.clear()


def test_chat_streams_out_of_scope_event():
    app.dependency_overrides[get_pipeline] = lambda: FakePipeline(
        [
            {"type": "token", "text": "I couldn't find relevant information."},
            {"type": "done", "sources": [], "is_out_of_scope": True},
        ]
    )
    client = TestClient(app)

    response = client.post("/chat", json={"question": "unrelated"})

    assert response.status_code == 200
    assert '"is_out_of_scope": true' in response.text

    app.dependency_overrides.clear()


def test_chat_streams_error_event():
    app.dependency_overrides[get_pipeline] = lambda: FakePipeline(
        [
            {"type": "token", "text": "partial"},
            {"type": "error", "message": "boom"},
        ]
    )
    client = TestClient(app)

    response = client.post("/chat", json={"question": "a question"})

    assert response.status_code == 200
    assert 'data: {"type": "error", "message": "boom"}' in response.text

    app.dependency_overrides.clear()


def test_chat_rejects_missing_question_field():
    app.dependency_overrides[get_pipeline] = lambda: FakePipeline([])
    client = TestClient(app)

    response = client.post("/chat", json={})

    assert response.status_code == 422

    app.dependency_overrides.clear()


def test_cors_allows_the_configured_frontend_origin():
    app.dependency_overrides[get_pipeline] = lambda: FakePipeline(
        [{"type": "done", "sources": [], "is_out_of_scope": True}]
    )
    client = TestClient(app)

    response = client.post(
        "/chat",
        json={"question": "hi"},
        headers={"Origin": "http://localhost:5173"},
    )

    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"

    app.dependency_overrides.clear()


def test_sources_are_serialized_to_json_safe_dicts():
    from pyrag.models import Chunk, RetrievedChunk

    source = RetrievedChunk(
        chunk=Chunk(id="a", text="t", source_file="f.rst", section_title="Intro"),
        score=0.87,
    )
    app.dependency_overrides[get_pipeline] = lambda: FakePipeline(
        [{"type": "done", "sources": [source], "is_out_of_scope": False}]
    )
    client = TestClient(app)

    response = client.post("/chat", json={"question": "hi"})

    assert '"section_title": "Intro"' in response.text
    assert '"source_file": "f.rst"' in response.text
    assert '"score": 0.87' in response.text

    app.dependency_overrides.clear()
