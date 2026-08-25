import numpy as np

from pyrag.retrieval.vector_store import VectorStore
from pyrag.models import Chunk


def test_vector_store_add_and_query_roundtrip(tmp_path):
    store = VectorStore(path=str(tmp_path / "chroma"))
    chunks = [
        Chunk(id="a", text="apples are fruit", source_file="f1", section_title="s1"),
        Chunk(id="b", text="cars have wheels", source_file="f2", section_title="s2"),
    ]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
    store.add(chunks, embeddings)

    result_ids = store.query(np.array([0.9, 0.1]), top_k=1)
    assert result_ids == ["a"]
