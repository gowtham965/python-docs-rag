import time

from groq import Groq

from pyrag.generation.gemini_client import GeminiClient


class GroqClient:
    def __init__(self, api_key: str, model: str, max_retries: int = 3, backoff_seconds: float = 1.0):
        self._client = Groq(api_key=api_key)
        self._model = model
        self._max_retries = max_retries
        self._backoff_seconds = backoff_seconds

    def generate(self, prompt: str) -> str:
        last_error = None
        for attempt in range(self._max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.choices[0].message.content
            except Exception as error:
                last_error = error
                if attempt < self._max_retries - 1:
                    time.sleep(self._backoff_seconds * (2 ** attempt))
        raise RuntimeError(f"Groq API failed after {self._max_retries} attempts") from last_error

    def generate_stream(self, prompt: str):
        stream = None
        last_error = None
        for attempt in range(self._max_retries):
            try:
                stream = self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    stream=True,
                )
                break
            except Exception as error:
                last_error = error
                if attempt < self._max_retries - 1:
                    time.sleep(self._backoff_seconds * (2 ** attempt))
        else:
            raise RuntimeError(f"Groq API failed after {self._max_retries} attempts") from last_error

        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


def build_llm_client(config):
    if config.llm_provider == "gemini":
        if not config.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key, "
                "or set LLM_PROVIDER back to 'groq'."
            )
        return GeminiClient(api_key=config.gemini_api_key, model=config.gemini_model)
    return GroqClient(api_key=config.groq_api_key, model=config.groq_model)
