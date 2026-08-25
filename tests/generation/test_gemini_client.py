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
