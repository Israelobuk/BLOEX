from __future__ import annotations

from typing import Any
from uuid import uuid4

from rapidfuzz import fuzz

from .persistence import AuditStore


class MemoryManager:
    def __init__(
        self,
        *,
        store: AuditStore,
        ollama_client,
        qdrant_url: str = "",
        collection_name: str = "bloex_analysis_memory",
    ):
        self.store = store
        self.ollama_client = ollama_client
        self.qdrant_url = qdrant_url.strip()
        self.collection_name = collection_name

    async def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        matches = self._rapidfuzz_search(query, limit=limit)
        semantic = await self._qdrant_search(query, limit=limit)
        combined = matches + semantic
        return sorted(combined, key=lambda item: item.get("score", 0), reverse=True)[:limit]

    def _rapidfuzz_search(self, query: str, limit: int) -> list[dict[str, Any]]:
        results = []
        for document in self.store.list_memory_documents(limit=100):
            request = document.get("request", {})
            text = f"{request.get('user_goal', '')} {document.get('final_prediction', '')}".strip()
            if not text:
                continue
            score = fuzz.token_set_ratio(query, text)
            if score >= 60:
                results.append(
                    {
                        "source": "rapidfuzz",
                        "score": round(float(score), 2),
                        "analysis_id": document["analysis_id"],
                        "final_prediction": document.get("final_prediction", ""),
                    }
                )
        return sorted(results, key=lambda item: item["score"], reverse=True)[:limit]

    async def _qdrant_search(self, query: str, limit: int) -> list[dict[str, Any]]:
        if not self.qdrant_url:
            return []
        embedding = await self.ollama_client.embedding(query)
        if not embedding:
            return []
        try:
            from qdrant_client import QdrantClient

            client = QdrantClient(url=self.qdrant_url, timeout=3)
            results = client.search(
                collection_name=self.collection_name,
                query_vector=embedding,
                limit=limit,
            )
            return [
                {
                    "source": "qdrant",
                    "score": round(float(result.score) * 100, 2),
                    "analysis_id": (result.payload or {}).get("analysis_id", ""),
                    "final_prediction": (result.payload or {}).get("final_prediction", ""),
                }
                for result in results
            ]
        except Exception:
            return []

    async def save_summary(self, *, analysis_id: str, text: str, final_prediction: str) -> bool:
        if not self.qdrant_url:
            return False
        embedding = await self.ollama_client.embedding(text)
        if not embedding:
            return False
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, PointStruct, VectorParams

            client = QdrantClient(url=self.qdrant_url, timeout=3)
            collections = client.get_collections().collections
            existing = {collection.name for collection in collections}
            if self.collection_name not in existing:
                client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=len(embedding), distance=Distance.COSINE),
                )
            client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=str(uuid4()),
                        vector=embedding,
                        payload={
                            "analysis_id": analysis_id,
                            "text": text,
                            "final_prediction": final_prediction,
                        },
                    )
                ],
            )
            return True
        except Exception:
            return False
