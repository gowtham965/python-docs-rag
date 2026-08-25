from pyrag.retrieval.embedder import Embedder


def test_embedder_returns_normalized_vectors():
    embedder = Embedder()
    vectors = embedder.embed(["hello world", "python programming"])
    assert vectors.shape[0] == 2
    assert vectors.shape[1] > 0
    norms = (vectors ** 2).sum(axis=1) ** 0.5
    assert all(abs(n - 1.0) < 1e-3 for n in norms)
