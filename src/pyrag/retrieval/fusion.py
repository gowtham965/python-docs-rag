from typing import Dict, List


def reciprocal_rank_fusion(ranked_lists: List[List[str]], k: int = 60) -> List[str]:
    scores: Dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, item_id in enumerate(ranked_list):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    return [
        item_id
        for item_id, _ in sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    ]
