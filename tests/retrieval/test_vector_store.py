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


def test_vector_store_add_batches_inserts_above_max_batch_size(tmp_path):
    """Verify that VectorStore.add() correctly batches inserts when chunk count exceeds max_batch_size."""
    store = VectorStore(path=str(tmp_path / "chroma"))

    class FakeClient:
        def get_max_batch_size(self):
            return 2

    class FakeCollection:
        def __init__(self):
            self.calls = []

        def add(self, ids, embeddings, documents, metadatas):
            self.calls.append({
                "ids": ids,
                "embeddings": embeddings,
                "documents": documents,
                "metadatas": metadatas,
            })

    fake_collection = FakeCollection()
    store._client = FakeClient()
    store._collection = fake_collection

    # Create 5 chunks and embeddings
    chunks = [
        Chunk(id=str(i), text=f"text {i}", source_file="test_file.py", section_title="Test Section")
        for i in range(5)
    ]
    embeddings = np.array([[float(i)] for i in range(5)])

    # Call add() which should batch into 3 calls (sizes 2, 2, 1)
    store.add(chunks, embeddings)

    # Verify batching behavior
    assert len(fake_collection.calls) == 3, f"Expected 3 batches, got {len(fake_collection.calls)}"

    # Batch 1: items 0-1
    assert fake_collection.calls[0]["ids"] == ["0", "1"]
    assert fake_collection.calls[0]["embeddings"] == [[0.0], [1.0]]
    assert fake_collection.calls[0]["documents"] == ["text 0", "text 1"]
    assert len(fake_collection.calls[0]["metadatas"]) == 2

    # Batch 2: items 2-3
    assert fake_collection.calls[1]["ids"] == ["2", "3"]
    assert fake_collection.calls[1]["embeddings"] == [[2.0], [3.0]]
    assert fake_collection.calls[1]["documents"] == ["text 2", "text 3"]
    assert len(fake_collection.calls[1]["metadatas"]) == 2

    # Batch 3: item 4
    assert fake_collection.calls[2]["ids"] == ["4"]
    assert fake_collection.calls[2]["embeddings"] == [[4.0]]
    assert fake_collection.calls[2]["documents"] == ["text 4"]
    assert len(fake_collection.calls[2]["metadatas"]) == 1

    # Verify metadata alignment
    for call in fake_collection.calls:
        for i, (id_val, meta) in enumerate(zip(call["ids"], call["metadatas"])):
            assert meta["source_file"] == "test_file.py"
            assert meta["section_title"] == "Test Section"
