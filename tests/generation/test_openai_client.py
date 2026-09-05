from unittest.mock import MagicMock, patch

import pytest

from pyrag.generation.openai_client import OpenAIClient


def test_generate_returns_content_on_success():
    with patch("pyrag.generation.openai_client.OpenAI") as MockOpenAI:
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "the answer"
        MockOpenAI.return_value.chat.completions.create.return_value = mock_response

        client = OpenAIClient(api_key="key", model="test-model")
        result = client.generate("a prompt")

        assert result == "the answer"


def test_generate_retries_then_raises_after_max_attempts():
    with patch("pyrag.generation.openai_client.OpenAI") as MockOpenAI, patch(
        "pyrag.generation.openai_client.time.sleep"
    ):
        MockOpenAI.return_value.chat.completions.create.side_effect = RuntimeError("boom")

        client = OpenAIClient(api_key="key", model="test-model", max_retries=2, backoff_seconds=0.01)

        with pytest.raises(RuntimeError, match="failed after 2 attempts"):
            client.generate("a prompt")

        assert MockOpenAI.return_value.chat.completions.create.call_count == 2


def test_generate_stream_yields_deltas_as_they_arrive():
    with patch("pyrag.generation.openai_client.OpenAI") as MockOpenAI:
        chunk1 = MagicMock()
        chunk1.choices[0].delta.content = "Hel"
        chunk2 = MagicMock()
        chunk2.choices[0].delta.content = "lo"
        MockOpenAI.return_value.chat.completions.create.return_value = [chunk1, chunk2]

        client = OpenAIClient(api_key="key", model="test-model")
        result = list(client.generate_stream("a prompt"))

        assert result == ["Hel", "lo"]
        _, kwargs = MockOpenAI.return_value.chat.completions.create.call_args
        assert kwargs["stream"] is True


def test_generate_stream_skips_chunks_with_no_content_delta():
    with patch("pyrag.generation.openai_client.OpenAI") as MockOpenAI:
        chunk1 = MagicMock()
        chunk1.choices[0].delta.content = "Hello"
        chunk2 = MagicMock()
        chunk2.choices[0].delta.content = None
        MockOpenAI.return_value.chat.completions.create.return_value = [chunk1, chunk2]

        client = OpenAIClient(api_key="key", model="test-model")
        result = list(client.generate_stream("a prompt"))

        assert result == ["Hello"]


def test_generate_stream_retries_the_initial_call_then_raises():
    with patch("pyrag.generation.openai_client.OpenAI") as MockOpenAI, patch(
        "pyrag.generation.openai_client.time.sleep"
    ):
        MockOpenAI.return_value.chat.completions.create.side_effect = RuntimeError("boom")

        client = OpenAIClient(api_key="key", model="test-model", max_retries=2, backoff_seconds=0.01)

        with pytest.raises(RuntimeError, match="failed after 2 attempts"):
            list(client.generate_stream("a prompt"))

        assert MockOpenAI.return_value.chat.completions.create.call_count == 2


def test_generate_stream_does_not_retry_mid_stream_failures():
    with patch("pyrag.generation.openai_client.OpenAI") as MockOpenAI, patch(
        "pyrag.generation.openai_client.time.sleep"
    ):
        def broken_stream():
            chunk = MagicMock()
            chunk.choices[0].delta.content = "partial"
            yield chunk
            raise RuntimeError("connection dropped")

        MockOpenAI.return_value.chat.completions.create.return_value = broken_stream()

        client = OpenAIClient(api_key="key", model="test-model", max_retries=3, backoff_seconds=0.01)

        gen = client.generate_stream("a prompt")
        assert next(gen) == "partial"
        with pytest.raises(RuntimeError, match="connection dropped"):
            next(gen)

        assert MockOpenAI.return_value.chat.completions.create.call_count == 1
