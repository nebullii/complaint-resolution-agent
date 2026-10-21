"""Chroma vector store over regulation chunks, searched only within the parts
that apply to the predicted product."""

from dataclasses import dataclass

from src import config

COLLECTION = "regulations"
# bge models expect this prefix on queries (not on documents).
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@dataclass
class Passage:
    id: str
    citation: str
    heading: str
    text: str
    part: str
    source: str
    as_of: str
    score: float


class RegulationStore:
    def __init__(self, path=config.CHROMA_DIR, embedding_model=config.EMBEDDING_MODEL):
        import chromadb
        from sentence_transformers import SentenceTransformer

        self.client = chromadb.PersistentClient(path=str(path))
        self.collection = self.client.get_or_create_collection(
            COLLECTION, metadata={"hnsw:space": "cosine"})
        self.encoder = SentenceTransformer(embedding_model)

    def build(self, chunks, batch_size=256):
        """Replace the collection with these chunks (dicts from ecfr.to_dicts)."""
        self.client.delete_collection(COLLECTION)
        self.collection = self.client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            texts = [f"{c['heading']}\n{c['text']}" for c in batch]
            embeddings = self.encoder.encode(texts, normalize_embeddings=True).tolist()
            self.collection.add(
                ids=[c["id"] for c in batch],
                documents=[c["text"] for c in batch],
                embeddings=embeddings,
                metadatas=[{k: c[k] for k in ("citation", "heading", "part", "source", "as_of")}
                           for c in batch],
            )
        return self.collection.count()

    def search(self, query, parts, k=config.RETRIEVAL_TOP_K):
        embedding = self.encoder.encode([QUERY_PREFIX + query], normalize_embeddings=True).tolist()
        where = {"part": parts[0]} if len(parts) == 1 else {"part": {"$in": parts}}
        result = self.collection.query(query_embeddings=embedding, n_results=k, where=where)
        return [
            Passage(id=id_, text=doc, score=1 - dist, **meta)
            for id_, doc, meta, dist in zip(result["ids"][0], result["documents"][0],
                                            result["metadatas"][0], result["distances"][0])
        ]
