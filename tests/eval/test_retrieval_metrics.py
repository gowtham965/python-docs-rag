from pyrag.eval.retrieval_metrics import hit_rate_at_k, mean_reciprocal_rank


def test_hit_rate_at_k_true_when_expected_within_k():
    retrieved = ["library/os.rst", "library/stdtypes.rst", "library/re.rst"]
    assert hit_rate_at_k(retrieved, "library/stdtypes.rst", k=2) is True


def test_hit_rate_at_k_false_when_expected_outside_k():
    retrieved = ["library/os.rst", "library/stdtypes.rst", "library/re.rst"]
    assert hit_rate_at_k(retrieved, "library/re.rst", k=1) is False


def test_mean_reciprocal_rank_scores_by_position():
    retrieved = ["library/os.rst", "library/stdtypes.rst"]
    assert mean_reciprocal_rank(retrieved, "library/stdtypes.rst") == 0.5


def test_mean_reciprocal_rank_zero_when_not_found():
    retrieved = ["library/os.rst"]
    assert mean_reciprocal_rank(retrieved, "library/re.rst") == 0.0
