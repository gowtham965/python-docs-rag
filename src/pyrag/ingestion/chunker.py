import uuid
from typing import List, Tuple

from pyrag.models import Chunk

HEADING_CHARS = set("=-~^\"'`#*+.:_")
MAX_WORDS = 400
OVERLAP_WORDS = 50


def _is_heading_underline(line: str) -> bool:
    stripped = line.strip()
    if len(stripped) < 3:
        return False
    return len(set(stripped)) == 1 and stripped[0] in HEADING_CHARS


def split_into_sections(text: str) -> List[Tuple[str, str]]:
    lines = text.splitlines()
    sections: List[Tuple[str, str]] = []
    current_heading = "Introduction"
    current_body: List[str] = []

    i = 0
    while i < len(lines):
        line = lines[i]
        next_line = lines[i + 1] if i + 1 < len(lines) else ""
        if (
            line.strip()
            and _is_heading_underline(next_line)
            and len(next_line.strip()) >= len(line.strip()) * 0.8
        ):
            if current_body:
                sections.append((current_heading, "\n".join(current_body).strip()))
            current_heading = line.strip()
            current_body = []
            i += 2
            continue
        current_body.append(line)
        i += 1

    if current_body:
        sections.append((current_heading, "\n".join(current_body).strip()))

    return [(heading, body) for heading, body in sections if body]


def _split_long_body(body: str, max_words: int, overlap_words: int) -> List[str]:
    words = body.split()
    if len(words) <= max_words:
        return [body]

    windows = []
    start = 0
    while start < len(words):
        end = start + max_words
        windows.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap_words
    return windows


def chunk_rst_file(text: str, source_file: str) -> List[Chunk]:
    chunks: List[Chunk] = []
    for heading, body in split_into_sections(text):
        for window_text in _split_long_body(body, MAX_WORDS, OVERLAP_WORDS):
            chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    text=window_text,
                    source_file=source_file,
                    section_title=heading,
                )
            )
    return chunks
