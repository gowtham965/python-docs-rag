from unittest.mock import MagicMock, patch

import pytest

from pyrag.generation.llm_client import GroqClient


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
