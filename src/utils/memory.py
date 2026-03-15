import redis
import json
import uuid
from datetime import datetime

# ── Connect to Redis ─────────────────────────────────────
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

SESSION_EXPIRY = 3600  # 1 hour

# ── Create new session ───────────────────────────────────
def create_session() -> str:
    session_id = str(uuid.uuid4())
    r.setex(
        f"session:{session_id}",
        SESSION_EXPIRY,
        json.dumps([])
    )
    return session_id

# ── Add message to session ───────────────────────────────
def add_message(session_id: str, role: str, text: str, intent: str = None, confidence: float = None):
    key = f"session:{session_id}"
    existing = r.get(key)
    messages = json.loads(existing) if existing else []

    message = {
        "role":      role,
        "text":      text,
        "timestamp": datetime.now().isoformat()
    }
    if intent:
        message["intent"]     = intent
        message["confidence"] = confidence

    messages.append(message)
    r.setex(key, SESSION_EXPIRY, json.dumps(messages))

# ── Get session history ──────────────────────────────────
def get_history(session_id: str) -> list:
    key = f"session:{session_id}"
    existing = r.get(key)
    return json.loads(existing) if existing else []

# ── Clear session ────────────────────────────────────────
def clear_session(session_id: str):
    r.delete(f"session:{session_id}")

# ── Get last intent from session ─────────────────────────
def get_last_intent(session_id: str) -> str:
    history = get_history(session_id)
    for msg in reversed(history):
        if msg.get("intent"):
            return msg["intent"]
    return None