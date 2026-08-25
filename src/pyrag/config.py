import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    groq_api_key: str
    groq_model: str = "openai/gpt-oss-20b"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    chroma_path: str = "data/chroma"
    relevance_threshold: float = 0.3
    llm_provider: str = "groq"
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-3.6-flash"


def load_config() -> Config:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    llm_provider = os.environ.get("LLM_PROVIDER", "groq")
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    return Config(
        groq_api_key=api_key,
        llm_provider=llm_provider,
        gemini_api_key=gemini_api_key,
    )
