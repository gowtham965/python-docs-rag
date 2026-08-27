from dataclasses import dataclass
from typing import List

from pyrag.models import RetrievedChunk
from pyrag.generation.prompt import build_prompt

OUT_OF_SCOPE_MESSAGE = "I couldn't find relevant information in the Python docs for this."


@dataclass
class AnswerResult:
    answer: str
    sources: List[RetrievedChunk]
    is_out_of_scope: bool


class RagPipeline:
    def __init__(self, retriever, llm_client):
        self._retriever = retriever
        self._llm_client = llm_client

    def answer(self, question: str) -> AnswerResult:
        retrieval = self._retriever.retrieve(question)

        if not retrieval.is_confident:
            return AnswerResult(answer=OUT_OF_SCOPE_MESSAGE, sources=[], is_out_of_scope=True)

        prompt = build_prompt(question, retrieval.chunks)
        answer_text = self._llm_client.generate(prompt)
        return AnswerResult(answer=answer_text, sources=retrieval.chunks, is_out_of_scope=False)

    def answer_stream(self, question: str):
        try:
            retrieval = self._retriever.retrieve(question)

            if not retrieval.is_confident:
                yield {"type": "token", "text": OUT_OF_SCOPE_MESSAGE}
                yield {"type": "done", "sources": [], "is_out_of_scope": True}
                return

            prompt = build_prompt(question, retrieval.chunks)
            for delta in self._llm_client.generate_stream(prompt):
                yield {"type": "token", "text": delta}
        except Exception as error:
            yield {"type": "error", "message": str(error)}
            return

        yield {"type": "done", "sources": retrieval.chunks, "is_out_of_scope": False}
