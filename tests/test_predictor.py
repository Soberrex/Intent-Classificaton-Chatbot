import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.inference.predictor import predict, clean_text

# ── Text cleaning tests ──────────────────────────────────
def test_clean_text_lowercase():
    assert clean_text("HELLO WORLD") == "hello world"

def test_clean_text_special_chars():
    assert clean_text("hello!!!") == "hello"

def test_clean_text_extra_spaces():
    assert clean_text("hello   world") == "hello world"

def test_clean_text_numbers():
    assert clean_text("order 12345") == "order 12345"

# ── Prediction tests ─────────────────────────────────────
def test_predict_returns_dict():
    result = predict("check my account balance")
    assert isinstance(result, dict)

def test_predict_has_required_keys():
    result = predict("check my account balance")
    assert "intent" in result
    assert "confidence" in result
    assert "top3" in result
    assert "is_low_confidence" in result

def test_predict_confidence_range():
    result = predict("check my account balance")
    assert 0 <= result["confidence"] <= 100

def test_predict_top3_length():
    result = predict("check my account balance")
    assert len(result["top3"]) == 3

def test_predict_intent_is_string():
    result = predict("check my account balance")
    assert isinstance(result["intent"], str)

def test_predict_banking_intent():
    result = predict("my card is not working")
    assert result["intent"] == "card_not_working"

def test_predict_greeting_intent():
    result = predict("hello how are you")
    assert result["intent"] == "greeting"

def test_predict_low_confidence_flag():
    result = predict("xyzabc123")
    assert result["is_low_confidence"] == True

def test_predict_high_confidence_flag():
    result = predict("my card is not working")
    assert result["is_low_confidence"] == False