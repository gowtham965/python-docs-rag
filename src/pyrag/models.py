from dataclasses import dataclass


@dataclass
class Chunk:
    id: str
    text: str
    source_file: str
    section_title: str


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
