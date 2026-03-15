import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

# ── Health & root tests ──────────────────────────────────
def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "running"

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"

def test_intents_endpoint():
    r = client.get("/intents")
    assert r.status_code == 200
    assert r.json()["total"] == 227

# ── Predict endpoint tests ───────────────────────────────
def test_predict_valid_input():
    r = client.post("/predict", json={"text": "check my balance"})
    assert r.status_code == 200
    data = r.json()
    assert "intent" in data
    assert "confidence" in data
    assert "session_id" in data

def test_predict_empty_text():
    r = client.post("/predict", json={"text": ""})
    assert r.status_code == 400

def test_predict_too_short():
    r = client.post("/predict", json={"text": "h"})
    assert r.status_code == 400

def test_predict_too_long():
    r = client.post("/predict", json={"text": "a" * 201})
    assert r.status_code == 400

def test_predict_creates_session():
    r = client.post("/predict", json={"text": "check my balance"})
    assert r.status_code == 200
    assert r.json()["session_id"] is not None

def test_predict_reuses_session():
    r1 = client.post("/predict", json={"text": "check my balance"})
    session_id = r1.json()["session_id"]
    r2 = client.post("/predict", json={
        "text": "transfer money",
        "session_id": session_id
    })
    assert r2.status_code == 200
    assert r2.json()["last_intent"] is not None

# ── Session history tests ────────────────────────────────
def test_get_history():
    r1 = client.post("/predict", json={"text": "check my balance"})
    session_id = r1.json()["session_id"]
    r2 = client.get(f"/history/{session_id}")
    assert r2.status_code == 200
    assert len(r2.json()["history"]) > 0

def test_delete_history():
    r1 = client.post("/predict", json={"text": "check my balance"})
    session_id = r1.json()["session_id"]
    r2 = client.delete(f"/history/{session_id}")
    assert r2.status_code == 200

def test_invalid_session_history():
    r = client.get("/history/invalid-session-id")
    assert r.status_code == 404