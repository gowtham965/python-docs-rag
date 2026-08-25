import json
import re
from typing import List

from pyrag.models import RetrievedChunk

JUDGE_PROMPT_TEMPLATE = """You are grading an AI-generated answer for a documentation Q&A system.

Question: {question}

Retrieved context:
{context}

Generated answer: {answer}

Score the answer from 1 (worst) to 5 (best) on two dimensions:
- faithfulness: is every claim in the answer supported by the retrieved context (no hallucination)?
- relevance: does the answer actually address the question?

Respond with ONLY a JSON object in this exact format, no other text:
{{"faithfulness": <int 1-5>, "relevance": <int 1-5>}}
"""


class JudgeScore:
    def __init__(self, faithfulness: int, relevance: int):
        self.faithfulness = faithfulness
        self.relevance = relevance


def _parse_json_response(raw_response: str) -> dict:
    """Parse JSON response from LLM, handling markdown code fences and prose wrapping.

    Handles common LLM output patterns:
    - Raw JSON: {"key": value}
    - Markdown-fenced JSON: ```json\n{"key": value}\n```
    - Prose-wrapped JSON: Some text {"key": value} more text
    """
    text = raw_response.strip()

    # Remove markdown code fences if present
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()

    # Try direct parsing first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback: extract first {...} block from prose
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def build_judge_prompt(question: str, answer: str, chunks: List[RetrievedChunk]) -> str:
    context = "\n\n".join(rc.chunk.text for rc in chunks)
    return JUDGE_PROMPT_TEMPLATE.format(question=question, context=context, answer=answer)


def judge_answer(question: str, answer: str, chunks: List[RetrievedChunk], llm_client) -> JudgeScore:
    prompt = build_judge_prompt(question, answer, chunks)
    raw_response = llm_client.generate(prompt)
    data = _parse_json_response(raw_response)
    return JudgeScore(faithfulness=int(data["faithfulness"]), relevance=int(data["relevance"]))
