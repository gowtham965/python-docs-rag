from pyrag.retrieval.fusion import reciprocal_rank_fusion


def test_rrf_favors_items_ranked_high_in_multiple_lists():
    list_a = ["x", "y", "z"]
    list_b = ["y", "x", "z"]
    fused = reciprocal_rank_fusion([list_a, list_b])
    assert fused[0] in ("x", "y")
    assert fused[-1] == "z"


def test_rrf_single_list_preserves_order():
    fused = reciprocal_rank_fusion([["a", "b", "c"]])
    assert fused == ["a", "b", "c"]


def test_rrf_computes_exact_score_with_known_values():
    # k=60 (default). list_a: "a" at rank 0, "b" at rank 1.
    # list_b: "b" at rank 0 only (a not present).
    # a's score = 1/(60+0+1) = 1/61 ≈ 0.016393
    # b's score = 1/(60+1+1) + 1/(60+0+1) = 1/62 + 1/61 ≈ 0.032914
    # b should clearly outrank a, and this only holds for the correct
    # +1 offset and k=60 — a wrong offset or k changes the outcome.
    fused = reciprocal_rank_fusion([["a", "b"], ["b"]])
    assert fused == ["b", "a"]
