import asyncio

from core.memory import MemoryManager
from core.persistence import AuditStore


class OfflineSemanticClient:
    async def embedding(self, text):
        return None


def test_rapidfuzz_memory_finds_similar_prior_case(tmp_path):
    store = AuditStore(tmp_path / "history.db")
    store.initialize()
    store.save_analysis(
        analysis_id="old",
        request_payload={"user_goal": "Find churn risk in customer revenue"},
        final_prediction="Revenue drops with high tickets indicate churn risk.",
        final_confidence=0.8,
        root_node={
            "node_id": "n",
            "parent_id": None,
            "depth": 0,
            "branch_label": "root",
            "row_count": 1,
            "statistical_summary": {},
            "trigger_reason": "root",
            "llm_prompt": "",
            "llm_response": "",
            "confidence_score": 0.8,
            "children": [],
        },
        memory_used=False,
        status="completed",
        memory_matches=[],
    )

    manager = MemoryManager(store=store, ollama_client=OfflineSemanticClient(), qdrant_url="")
    matches = asyncio.run(manager.search("Find customer churn risk from revenue"))

    assert matches
    assert matches[0]["source"] == "rapidfuzz"
    assert matches[0]["score"] >= 60


def test_semantic_memory_offline_returns_no_error(tmp_path):
    store = AuditStore(tmp_path / "history.db")
    store.initialize()
    manager = MemoryManager(store=store, ollama_client=OfflineSemanticClient(), qdrant_url="http://127.0.0.1:6333")

    matches = asyncio.run(manager.search("anything"))

    assert isinstance(matches, list)
