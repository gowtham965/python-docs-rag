import time

# NOTE: google-generativeai is deprecated upstream in favor of google.genai.
# Still fully functional as of this writing; migrate when convenient.
import google.generativeai as genai


class GeminiClient:
    def __init__(self, api_key: str, model: str, max_retries: int = 3, backoff_seconds: float = 1.0):
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model)
        self._max_retries = max_retries
        self._backoff_seconds = backoff_seconds

    def generate(self, prompt: str) -> str:
        last_error = None
        for attempt in range(self._max_retries):
            try:
                response = self._model.generate_content(prompt)
                return response.text
            except Exception as error:
                last_error = error
                if attempt < self._max_retries - 1:
                    time.sleep(self._backoff_seconds * (2 ** attempt))
        raise RuntimeError(f"Gemini API failed after {self._max_retries} attempts") from last_error
