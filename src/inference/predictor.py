import os
import torch
import pickle
import re
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Config ──────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent.parent.parent
MODEL_PATH   = str(BASE_DIR / "models" / "bert-intent-model")
ENCODER_PATH = str(BASE_DIR / "data" / "processed" / "label_encoder.pkl")
MAX_LEN      = 64
DEVICE       = torch.device("cpu")

# ── Constants ────────────────────────────────────────────
MIN_TEXT_LENGTH      = 2
MAX_TEXT_LENGTH      = 200
CONFIDENCE_THRESHOLD = 10.0
OOS_INTENT           = "oos"
OOS_RESPONSE         = "I'm not sure I understand that. Could you rephrase or ask me something banking related?"

print("Loading model from:", MODEL_PATH)

# ── Load model, tokenizer, label encoder ────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
model     = model.to(DEVICE)
model.eval()

with open(ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

print(f"Model loaded! Classes: {len(label_encoder.classes_)}")

# ── Text cleaning ────────────────────────────────────────
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ── Predict function ─────────────────────────────────────
def predict(text: str) -> dict:
    cleaned  = clean_text(text)
    encoding = tokenizer(
        cleaned,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    input_ids      = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits  = outputs.logits

    probabilities               = torch.softmax(logits, dim=1)
    confidence, predicted_class = torch.max(probabilities, dim=1)

    predicted_label  = label_encoder.inverse_transform([predicted_class.item()])[0]
    confidence_score = round(confidence.item() * 100, 2)

    top3_probs, top3_indices = torch.topk(probabilities, 3, dim=1)
    top3 = [
        {
            "intent":     label_encoder.inverse_transform([idx.item()])[0],
            "confidence": round(prob.item() * 100, 2)
        }
        for prob, idx in zip(top3_probs[0], top3_indices[0])
    ]

    is_low_confidence = confidence_score < CONFIDENCE_THRESHOLD

    if predicted_label == OOS_INTENT or is_low_confidence:
        predicted_label = "out_of_scope"
        is_oos = True
    else:
        is_oos = False

    return {
        "input_text":        text,
        "cleaned_text":      cleaned,
        "intent":            predicted_label,
        "confidence":        confidence_score,
        "top3":              top3,
        "is_low_confidence": is_low_confidence,
        "is_oos":            is_oos,
        "response":          OOS_RESPONSE if is_oos else f"I detected your intent as {predicted_label}"
    }

# ── Quick test ───────────────────────────────────────────
if __name__ == "__main__":
    test_queries = [
        "I want to check my account balance",
        "how do I transfer money to someone",
        "my card is not working",
        "what are your opening hours",
        "I need to cancel my subscription"
    ]

    print("\n" + "="*50)
    print("INFERENCE TEST")
    print("="*50)

    for query in test_queries:
        result = predict(query)
        print(f"\nInput:      {result['input_text']}")
        print(f"Intent:     {result['intent']}")
        print(f"Confidence: {result['confidence']}%")
        print(f"Is OOS:     {result['is_oos']}")
        print(f"Response:   {result['response']}")
        print(f"Top 3:      {[(r['intent'], r['confidence']) for r in result['top3']]}")