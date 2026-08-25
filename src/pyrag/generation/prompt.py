from typing import List

from pyrag.models import RetrievedChunk

SYSTEM_INSTRUCTIONS = (
    "You are a helpful assistant answering questions about the Python "
    "standard library using only the provided documentation excerpts. "
    "Cite the section title for every claim you make, using the format "
    "[Section: <title>]. If the excerpts do not contain the answer, say "
    "\"I don't know based on the provided documentation.\" Do not use "
    "outside knowledge."
)


def build_prompt(question: str, chunks: List[RetrievedChunk]) -> str:
    context_blocks = "\n\n".join(
        f"[Section: {rc.chunk.section_title}]\n{rc.chunk.text}" for rc in chunks
    )
    return (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"Documentation excerpts:\n{context_blocks}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )
