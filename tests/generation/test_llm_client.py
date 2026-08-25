from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from pyrag.generation.gemini_client import GeminiClient
from pyrag.generation.llm_client import GroqClient, build_llm_client


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
