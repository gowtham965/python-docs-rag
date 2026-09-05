from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from pyrag.generation.gemini_client import GeminiClient
from pyrag.generation.llm_client import GroqClient, build_llm_client
from pyrag.generation.openai_client import OpenAIClient


def test_generate_returns_content_on_success():
    with patch("pyrag.generation.llm_client.Groq") as MockGroq:
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "the answer"
        MockGroq.return_value.chat.completions.create.return_value = mock_response

        client = GroqClient(api_key="key", model="test-model")
        result = client.generate("a prompt")

        assert result == "the answer"


def test_generate_retries_then_raises_after_max_attempts():
    with patch("pyrag.generation.llm_client.Groq") as MockGroq, patch(
        "pyrag.generation.llm_client.time.sleep"
    ):
        MockGroq.return_value.chat.completions.create.side_effect = RuntimeError("boom")

        client = GroqClient(api_key="key", model="test-model", max_retries=2, backoff_seconds=0.01)

        with pytest.raises(RuntimeError, match="failed after 2 attempts"):
            client.generate("a prompt")

        assert MockGroq.return_value.chat.completions.create.call_count == 2


def test_build_llm_client_returns_groq_client_by_default():
    config = SimpleNamespace(
        llm_provider="groq",
        groq_api_key="key",
        groq_model="test-model",
        gemini_api_key=None,
        gemini_model="test-gemini-model",
    )

    client = build_llm_client(config)

    assert isinstance(client, GroqClient)


def test_build_llm_client_returns_gemini_client_when_configured():
    config = SimpleNamespace(
        llm_provider="gemini",
        groq_api_key="key",
        groq_model="test-model",
        gemini_api_key="gemini-key",
        gemini_model="test-gemini-model",
    )

    client = build_llm_client(config)

    assert isinstance(client, GeminiClient)


def test_build_llm_client_raises_when_gemini_selected_without_api_key():
    config = SimpleNamespace(
        llm_provider="gemini",
        groq_api_key="key",
        groq_model="test-model",
        gemini_api_key=None,
        gemini_model="test-gemini-model",
    )

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        build_llm_client(config)


def test_build_llm_client_returns_openai_client_when_configured():
    config = SimpleNamespace(
        llm_provider="openai",
        groq_api_key="key",
        groq_model="test-model",
        gemini_api_key=None,
        gemini_model="test-gemini-model",
        openai_api_key="openai-key",
        openai_model="test-openai-model",
    )

    client = build_llm_client(config)

    assert isinstance(client, OpenAIClient)


def test_build_llm_client_raises_when_openai_selected_without_api_key():
    config = SimpleNamespace(
        llm_provider="openai",
        groq_api_key="key",
        groq_model="test-model",
        gemini_api_key=None,
        gemini_model="test-gemini-model",
        openai_api_key=None,
        openai_model="test-openai-model",
    )

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        build_llm_client(config)


def test_generate_stream_yields_deltas_as_they_arrive():
    with patch("pyrag.generation.llm_client.Groq") as MockGroq:
        chunk1 = MagicMock()
        chunk1.choices[0].delta.content = "Hel"
        chunk2 = MagicMock()
        chunk2.choices[0].delta.content = "lo"
        MockGroq.return_value.chat.completions.create.return_value = [chunk1, chunk2]

        client = GroqClient(api_key="key", model="test-model")
        result = list(client.generate_stream("a prompt"))

        assert result == ["Hel", "lo"]
        _, kwargs = MockGroq.return_value.chat.completions.create.call_args
        assert kwargs["stream"] is True


def test_generate_stream_skips_chunks_with_no_content_delta():
    with patch("pyrag.generation.llm_client.Groq") as MockGroq:
        chunk1 = MagicMock()
        chunk1.choices[0].delta.content = "Hello"
        chunk2 = MagicMock()
        chunk2.choices[0].delta.content = None
        MockGroq.return_value.chat.completions.create.return_value = [chunk1, chunk2]

        client = GroqClient(api_key="key", model="test-model")
        result = list(client.generate_stream("a prompt"))

        assert result == ["Hello"]


def test_generate_stream_retries_the_initial_call_then_raises():
    with patch("pyrag.generation.llm_client.Groq") as MockGroq, patch(
        "pyrag.generation.llm_client.time.sleep"
    ):
        MockGroq.return_value.chat.completions.create.side_effect = RuntimeError("boom")

        client = GroqClient(api_key="key", model="test-model", max_retries=2, backoff_seconds=0.01)

        with pytest.raises(RuntimeError, match="failed after 2 attempts"):
            list(client.generate_stream("a prompt"))

        assert MockGroq.return_value.chat.completions.create.call_count == 2


def test_generate_stream_does_not_retry_mid_stream_failures():
    with patch("pyrag.generation.llm_client.Groq") as MockGroq, patch(
        "pyrag.generation.llm_client.time.sleep"
    ):
        def broken_stream():
            chunk = MagicMock()
            chunk.choices[0].delta.content = "partial"
            yield chunk
            raise RuntimeError("connection dropped")

        MockGroq.return_value.chat.completions.create.return_value = broken_stream()

        client = GroqClient(api_key="key", model="test-model", max_retries=3, backoff_seconds=0.01)

        gen = client.generate_stream("a prompt")
        assert next(gen) == "partial"
        with pytest.raises(RuntimeError, match="connection dropped"):
            next(gen)

        assert MockGroq.return_value.chat.completions.create.call_count == 1
