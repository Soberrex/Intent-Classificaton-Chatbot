import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.memory import (
    create_session,
    add_message,
    get_history,
    clear_session,
    get_last_intent
)

def test_create_session():
    session_id = create_session()
    assert session_id is not None
    assert len(session_id) > 0

def test_add_and_get_message():
    session_id = create_session()
    add_message(session_id, "user", "hello")
    history = get_history(session_id)
    assert len(history) == 1
    assert history[0]["text"] == "hello"
    assert history[0]["role"] == "user"

def test_add_multiple_messages():
    session_id = create_session()
    add_message(session_id, "user", "hello")
    add_message(session_id, "bot", "Intent: greeting", intent="greeting", confidence=85.0)
    history = get_history(session_id)
    assert len(history) == 2

def test_get_last_intent():
    session_id = create_session()
    add_message(session_id, "bot", "Intent: greeting", intent="greeting", confidence=85.0)
    last = get_last_intent(session_id)
    assert last == "greeting"

def test_clear_session():
    session_id = create_session()
    add_message(session_id, "user", "hello")
    clear_session(session_id)
    history = get_history(session_id)
    assert len(history) == 0

def test_empty_session_history():
    session_id = create_session()
    history = get_history(session_id)
    assert history == []

def test_last_intent_empty_session():
    session_id = create_session()
    last = get_last_intent(session_id)
    assert last is None