from fastapi.testclient import TestClient

from backend.main import app


def test_existing_routes_remain_mounted():
    client = TestClient(app)

    config_response = client.get("/api/config")
    explain_response = client.post("/api/explain", json={"question": "Q?", "model_answer": "", "context": ""})

    assert config_response.status_code in {200, 500}
    assert explain_response.status_code in {200, 500}
