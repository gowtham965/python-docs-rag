from pyrag.config import load_config
from pyrag.ingestion.build_index import load_chunks
from pyrag.retrieval.embedder import Embedder
from pyrag.retrieval.vector_store import VectorStore
from pyrag.retrieval.bm25_store import BM25Store
from pyrag.retrieval.reranker import Reranker
from pyrag.retrieval.retriever import Retriever
from pyrag.generation.llm_client import build_llm_client
from pyrag.generation.pipeline import RagPipeline


def build_pipeline(chunks_path: str = "data/processed/chunks.json"):
    config = load_config()
    chunks = load_chunks(chunks_path)
    chunk_lookup = {c.id: c for c in chunks}

    embedder = Embedder(config.embedding_model)
    vector_store = VectorStore(path=config.chroma_path)
    bm25_store = BM25Store(chunks)
    reranker = Reranker(config.reranker_model)

    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
        bm25_store=bm25_store,
        reranker=reranker,
        chunk_lookup=chunk_lookup,
        relevance_threshold=config.relevance_threshold,
    )
    llm_client = build_llm_client(config)
    return RagPipeline(retriever=retriever, llm_client=llm_client), retriever
