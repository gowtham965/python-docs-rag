import pytest
from pyrag.config import load_config


def test_load_config_reads_api_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key-123")
    config = load_config()
    assert config.groq_api_key == "test-key-123"
    assert config.embedding_model == "BAAI/bge-small-en-v1.5"


def test_load_config_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        load_config()


def test_load_config_defaults_llm_provider_to_groq(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key-123")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    config = load_config()
    assert config.llm_provider == "groq"
    assert config.gemini_api_key is None
    assert config.gemini_model == "gemini-3.6-flash"


def test_load_config_reads_gemini_settings_when_set(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key-123")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key-456")
    config = load_config()
    assert config.llm_provider == "gemini"
    assert config.gemini_api_key == "gemini-key-456"


def test_load_config_reads_openai_settings_when_set(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key-123")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key-789")
    config = load_config()
    assert config.llm_provider == "openai"
    assert config.openai_api_key == "openai-key-789"
    assert config.openai_model == "gpt-4o-mini"
