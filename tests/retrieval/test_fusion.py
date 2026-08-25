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
    # k=1 (explicit, to make the +1 offset large relative to scores).
    # Correct formula: score = 1/(k+rank+1)
    #   list_1 = ["A", "B"]        -> A rank0, B rank1
    #   list_2 = ["z1", "z2", "B"] -> z1 rank0, z2 rank1, B rank2
    #   A  = 1/(1+0+1)                     = 0.5
    #   B  = 1/(1+1+1) + 1/(1+2+1)         = 0.33333 + 0.25   = 0.58333
    #   z1 = 1/(1+0+1)                     = 0.5
    #   z2 = 1/(1+1+1)                     = 0.33333
    #   -> B (0.58333) strictly beats A/z1 (0.5): fused[0] == "B"
    #
    # If the +1 offset were missing (bug: score = 1/(k+rank) instead):
    #   A  = 1/(1+0)             = 1.0
    #   B  = 1/(1+1) + 1/(1+2)   = 0.5 + 0.33333 = 0.83333
    #   -> A (1.0) would beat B (0.83333): fused[0] would be "A", not "B"
    # So this assertion only holds under the correct +1 offset.
    fused = reciprocal_rank_fusion([["A", "B"], ["z1", "z2", "B"]], k=1)
    assert fused[0] == "B"
