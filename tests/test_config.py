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
