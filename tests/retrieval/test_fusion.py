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
