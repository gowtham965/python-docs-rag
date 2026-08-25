from typing import List


def hit_rate_at_k(retrieved_source_files: List[str], expected_source_file: str, k: int) -> bool:
    return expected_source_file in retrieved_source_files[:k]


def mean_reciprocal_rank(retrieved_source_files: List[str], expected_source_file: str) -> float:
    for rank, source_file in enumerate(retrieved_source_files, start=1):
        if source_file == expected_source_file:
            return 1.0 / rank
    return 0.0
