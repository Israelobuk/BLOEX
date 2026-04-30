from fastapi.testclient import TestClient

from backend.main import app


def test_analyze_analysis_history_and_health_routes(monkeypatch, tmp_path):
    db_path = tmp_path / "history.db"
    monkeypatch.setenv("BBE_HISTORY_DB", str(db_path))
    monkeypatch.setenv("BBE_BASE_URL", "http://127.0.0.1:9")
    client = TestClient(app)

    payload = {
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
        "use_llm": False,
    }

    created = client.post("/api/analyze", json=payload)
    assert created.status_code == 200
    body = created.json()
    assert body["analysis_id"]
    assert body["status"] == "completed"
    assert body["root_node"]["children"]

    fetched = client.get(f"/api/analysis/{body['analysis_id']}")
    assert fetched.status_code == 200
    assert fetched.json()["analysis_id"] == body["analysis_id"]

    history = client.get("/api/history")
    assert history.status_code == 200
    assert history.json()["items"][0]["analysis_id"] == body["analysis_id"]

    health = client.get("/api/health")
    assert health.status_code == 200
    assert "predictiveEngine" in health.json()


def test_analysis_route_returns_404_for_missing_id(monkeypatch, tmp_path):
    monkeypatch.setenv("BBE_HISTORY_DB", str(tmp_path / "history.db"))
    client = TestClient(app)

    response = client.get("/api/analysis/missing")

    assert response.status_code == 404
