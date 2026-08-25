import pytest

from pyrag.wiring import build_pipeline


def test_build_pipeline_raises_clear_error_when_chunks_file_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    with pytest.raises(FileNotFoundError):
        build_pipeline(chunks_path=str(tmp_path / "missing.json"))
