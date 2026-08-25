import time

from groq import Groq


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
