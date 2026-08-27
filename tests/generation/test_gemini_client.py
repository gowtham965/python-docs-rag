from unittest.mock import MagicMock, patch

import pytest

from pyrag.generation.gemini_client import GeminiClient


def test_generate_returns_content_on_success():
    with patch("pyrag.generation.gemini_client.genai") as mock_genai:
        mock_response = MagicMock()
        mock_response.text = "the answer"
        mock_genai.GenerativeModel.return_value.generate_content.return_value = mock_response

        client = GeminiClient(api_key="key", model="test-model")
        result = client.generate("a prompt")

        assert result == "the answer"


def test_generate_passes_request_timeout_to_the_sdk():
    with patch("pyrag.generation.gemini_client.genai") as mock_genai:
        mock_response = MagicMock()
        mock_response.text = "the answer"
        mock_generate_content = mock_genai.GenerativeModel.return_value.generate_content
        mock_generate_content.return_value = mock_response

        client = GeminiClient(api_key="key", model="test-model", request_timeout_seconds=5.0)
        client.generate("a prompt")

        _, kwargs = mock_generate_content.call_args
        assert kwargs["request_options"] == {"timeout": 5.0}


def test_generate_retries_then_raises_after_max_attempts():
    with patch("pyrag.generation.gemini_client.genai") as mock_genai, patch(
        "pyrag.generation.gemini_client.time.sleep"
    ):
        mock_genai.GenerativeModel.return_value.generate_content.side_effect = RuntimeError("boom")

        client = GeminiClient(api_key="key", model="test-model", max_retries=2, backoff_seconds=0.01)

        with pytest.raises(RuntimeError, match="failed after 2 attempts"):
            client.generate("a prompt")

        assert mock_genai.GenerativeModel.return_value.generate_content.call_count == 2


def test_generate_stream_yields_text_deltas():
    with patch("pyrag.generation.gemini_client.genai") as mock_genai:
        chunk1 = MagicMock(text="Hel")
        chunk2 = MagicMock(text="lo")
        mock_generate_content = mock_genai.GenerativeModel.return_value.generate_content
        mock_generate_content.return_value = [chunk1, chunk2]

        client = GeminiClient(api_key="key", model="test-model")
        result = list(client.generate_stream("a prompt"))

        assert result == ["Hel", "lo"]
        _, kwargs = mock_generate_content.call_args
        assert kwargs["stream"] is True
        assert kwargs["request_options"] == {"timeout": 30.0}


def test_generate_stream_skips_chunks_with_no_text():
    with patch("pyrag.generation.gemini_client.genai") as mock_genai:
        chunk1 = MagicMock(text="Hello")
        chunk2 = MagicMock(text=None)
        mock_genai.GenerativeModel.return_value.generate_content.return_value = [chunk1, chunk2]

        client = GeminiClient(api_key="key", model="test-model")
        result = list(client.generate_stream("a prompt"))

        assert result == ["Hello"]


def test_generate_stream_retries_the_initial_call_then_raises():
    with patch("pyrag.generation.gemini_client.genai") as mock_genai, patch(
        "pyrag.generation.gemini_client.time.sleep"
    ):
        mock_genai.GenerativeModel.return_value.generate_content.side_effect = RuntimeError("boom")

        client = GeminiClient(api_key="key", model="test-model", max_retries=2, backoff_seconds=0.01)

        with pytest.raises(RuntimeError, match="failed after 2 attempts"):
            list(client.generate_stream("a prompt"))

        assert mock_genai.GenerativeModel.return_value.generate_content.call_count == 2


def test_generate_stream_does_not_retry_mid_stream_failures():
    with patch("pyrag.generation.gemini_client.genai") as mock_genai, patch(
        "pyrag.generation.gemini_client.time.sleep"
    ):
        def broken_stream():
            yield MagicMock(text="partial")
            raise RuntimeError("connection dropped")

        mock_genai.GenerativeModel.return_value.generate_content.return_value = broken_stream()

        client = GeminiClient(api_key="key", model="test-model", max_retries=3, backoff_seconds=0.01)

        gen = client.generate_stream("a prompt")
        assert next(gen) == "partial"
        with pytest.raises(RuntimeError, match="connection dropped"):
            next(gen)

        assert mock_genai.GenerativeModel.return_value.generate_content.call_count == 1
