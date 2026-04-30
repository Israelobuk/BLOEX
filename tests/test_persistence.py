from core.persistence import AuditStore


def test_audit_store_persists_analysis_nodes_and_history(tmp_path):
    db_path = tmp_path / "history.db"
    store = AuditStore(db_path)
    store.initialize()

    node = {
        "node_id": "node-1",
        "parent_id": None,
        "depth": 0,
        "branch_label": "root",
        "row_count": 2,
        "statistical_summary": {"numeric_means": {"revenue": 50.0}},
        "trigger_reason": "root analysis",
        "llm_prompt": "prompt",
        "llm_response": "response",
        "confidence_score": 0.7,
        "children": [],
    }

    store.save_analysis(
        analysis_id="analysis-1",
        request_payload={"user_goal": "goal"},
        final_prediction="prediction",
        final_confidence=0.7,
        root_node=node,
        memory_used=True,
        status="completed",
        memory_matches=[{"source": "rapidfuzz", "score": 91}],
    )

    saved = store.get_analysis("analysis-1")
    history = store.list_history()
    searchable = store.list_memory_documents()

    assert saved["analysis_id"] == "analysis-1"
    assert saved["root_node"]["statistical_summary"]["numeric_means"]["revenue"] == 50.0
    assert saved["memory_used"] is True
    assert history[0]["analysis_id"] == "analysis-1"
    assert searchable[0]["final_prediction"] == "prediction"


def test_audit_store_returns_none_for_unknown_analysis(tmp_path):
    store = AuditStore(tmp_path / "history.db")
    store.initialize()

    assert store.get_analysis("missing") is None
