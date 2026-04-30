import asyncio

from core.ollama_client import OllamaResult
from core.persistence import AuditStore
from core.workflow import run_analysis_workflow


class OfflineOllama:
    async def generate(self, prompt, *, json_mode=False):
        return OllamaResult(ok=False, text="", error="offline")

    async def embedding(self, text):
        return None


def test_langgraph_workflow_runs_full_analysis_with_fallback_llm(tmp_path):
    store = AuditStore(tmp_path / "history.db")
    request_payload = {
        "user_goal": "Find churn risk",
        "data": [
            {"region": "north", "revenue": 100, "tickets": 2},
            {"region": "north", "revenue": 120, "tickets": 3},
            {"region": "south", "revenue": 600, "tickets": 12},
            {"region": "south", "revenue": 700, "tickets": 13},
        ],
        "focus_columns": ["region", "revenue", "tickets"],
        "max_depth": 2,
        "min_rows": 2,
        "use_memory": True,
        "use_llm": True,
    }

    result = asyncio.run(
        run_analysis_workflow(
            analysis_id="analysis-1",
            request_payload=request_payload,
            store=store,
            ollama_client=OfflineOllama(),
            qdrant_url="",
        )
    )

    saved = store.get_analysis("analysis-1")

    assert result["analysis_id"] == "analysis-1"
    assert result["status"] == "completed"
    assert result["workflow"] in {"langgraph", "sequential"}
    assert result["root_node"]["llm_response"].startswith("Deterministic analysis only")
    assert saved["analysis_id"] == "analysis-1"
