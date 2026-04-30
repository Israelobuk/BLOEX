from fastapi.testclient import TestClient

from backend.main import app


def test_explain_route_enriches_structured_context(monkeypatch, tmp_path):
    monkeypatch.setenv("BBE_HISTORY_DB", str(tmp_path / "history.db"))
    monkeypatch.setenv("BBE_BASE_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("BBE_SIGNAL_ENRICHMENT", "true")
    client = TestClient(app)

    response = client.post(
        "/api/explain",
        json={
            "question": "Why did the model say this result is strong?",
            "model_answer": "The model says the latest result is strong because the evidence looks consistent.",
            "context": "Run Alpha, score 91, 4 signals, outcome strong.\nRun Beta, score 62, 1 signal, outcome weak.\nRun Gamma, score 88, 3 signals, outcome strong.",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["signal_analysis"]["analysis_id"]
    assert "recursive signal read" in body["black_box_explanation"].lower()
    assert body["evidence_claims"]
