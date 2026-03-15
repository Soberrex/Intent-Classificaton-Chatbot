import os
import sys
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.inference.predictor import predict, label_encoder
from src.utils.memory import create_session, add_message, get_history, clear_session, get_last_intent

# ── Lifespan ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("✅ Intent Chatbot API started!")
    print(f"✅ Model loaded with {len(label_encoder.classes_)} intents")
    yield
    print("API shutting down...")

# ── App ──────────────────────────────────────────────────
app = FastAPI(
    title="Intent Classification Chatbot API",
    description="BERT-based intent classification API with Redis memory",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── Models ───────────────────────────────────────────────
from typing import Optional

class PredictRequest(BaseModel):
    text:       str
    session_id: Optional[str] = None

class IntentResult(BaseModel):
    intent:     str
    confidence: float

class PredictResponse(BaseModel):
    session_id:        str
    input_text:        str
    cleaned_text:      str
    intent:            str
    confidence:        float
    top3:              list[IntentResult]
    last_intent:       Optional[str] = None
    is_low_confidence: bool = False

# ── Routes ───────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Intent Classification Chatbot API",
        "status":  "running",
        "intents": len(label_encoder.classes_)
    }

@app.get("/health")
def health():
    return {
        "status":  "healthy",
        "model":   "bert-base-uncased",
        "intents": len(label_encoder.classes_)
    }

@app.post("/predict", response_model=PredictResponse)
def predict_intent(request: PredictRequest):
    text = request.text.strip()

    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if len(text) < 2:
        raise HTTPException(status_code=400, detail="Text too short — minimum 2 characters")

    if len(text) > 200:
        raise HTTPException(status_code=400, detail="Text too long — maximum 200 characters")

    # Create session if not provided
    session_id = request.session_id or create_session()

    # Get last intent for context
    last_intent = get_last_intent(session_id)

    # Run prediction
    result = predict(request.text)

    # Save to Redis
    add_message(session_id, "user", request.text)
    add_message(
        session_id, "bot",
        f"Intent: {result['intent']}",
        intent=result["intent"],
        confidence=result["confidence"]
    )

    return PredictResponse(
        session_id=session_id,
        input_text=result["input_text"],
        cleaned_text=result["cleaned_text"],
        intent=result["intent"],
        confidence=result["confidence"],
        top3=[IntentResult(**r) for r in result["top3"]],
        last_intent=last_intent,
        is_low_confidence=result["is_low_confidence"]
    )

@app.get("/history/{session_id}")
def get_session_history(session_id: str):
    history = get_history(session_id)
    if not history:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "history": history}

@app.delete("/history/{session_id}")
def delete_session(session_id: str):
    clear_session(session_id)
    return {"message": "Session cleared"}

@app.get("/intents")
def get_intents():
    return {
        "total":   len(label_encoder.classes_),
        "intents": list(label_encoder.classes_)
    }