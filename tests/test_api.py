from fastapi.testclient import TestClient

from src.api.main import create_app
from tests.conftest import FakeClassifier

COMPLAINT = "A collector keeps calling me about a debt I do not owe, at 555-123-4567."


def client_for(make_service, **kw):
    service, _ = make_service(**kw)
    return TestClient(create_app(service)), service


def test_classify_masks_and_logs(make_service, audit):
    client, _ = client_for(make_service)
    r = client.post("/classify", json={"text": COMPLAINT})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "debt_collection" and body["model_version"] == "fake-v1"
    logged = audit.events("classify-only")[0]["payload"]
    assert "555-123-4567" not in logged["text"] and logged["pii_masked"] == {"PHONE": 1}


def test_full_flow_over_http(make_service):
    client, _ = client_for(make_service)
    r = client.post("/complaints", json={"text": COMPLAINT})
    cid = r.json()["complaint_id"]
    assert r.json()["pending"]["type"] == "approval"

    # A triage answer is the wrong shape for an approval question.
    assert client.post(f"/complaints/{cid}/review",
                       json={"reviewer": "neha", "label": "mortgage"}).status_code == 422

    r = client.post(f"/complaints/{cid}/review", json={"reviewer": "neha", "decision": "approve"})
    assert r.json()["state"]["status"] == "approved"
    assert client.post(f"/complaints/{cid}/review",
                       json={"reviewer": "neha", "decision": "approve"}).status_code == 409
    assert [e["step"] for e in client.get(f"/complaints/{cid}/audit").json()][-1] == "approval"


def test_triage_over_http(make_service):
    client, _ = client_for(make_service, classifier=FakeClassifier(confidence=0.2))
    cid = client.post("/complaints", json={"text": COMPLAINT}).json()["complaint_id"]
    r = client.post(f"/complaints/{cid}/review", json={"reviewer": "neha", "label": "debt_collection"})
    assert r.json()["pending"]["type"] == "approval"


def test_unknown_complaint_is_404(make_service):
    client, _ = client_for(make_service)
    assert client.get("/complaints/nope").status_code == 404
    assert client.post("/complaints/nope/review",
                       json={"reviewer": "neha", "decision": "approve"}).status_code == 404


def test_too_short_complaint_is_rejected(make_service):
    client, _ = client_for(make_service)
    assert client.post("/classify", json={"text": "hi"}).status_code == 422
