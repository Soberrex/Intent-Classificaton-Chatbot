# 🤖 Intent Classification Chatbot

A production-ready conversational AI chatbot that detects user intent using a fine-tuned BERT model.

## 🎯 Performance
- **Accuracy:** 88.04% on validation set
- **F1 Score:** 87.25%
- **Intents:** 227 across Banking77 + CLINC150 datasets
- **Tests:** 32/32 passing

## 🛠️ Tech Stack
- **Model:** BERT (bert-base-uncased) fine-tuned for sequence classification
- **Backend:** FastAPI + Uvicorn
- **Frontend:** Streamlit
- **Memory:** Redis (Memurai on Windows)
- **Training:** PyTorch + HuggingFace Transformers on Google Colab (T4 GPU)

## 📁 Project Structure
```
intent-classification-chatbot/
├── src/
│   ├── inference/predictor.py    ← BERT inference pipeline
│   └── utils/memory.py           ← Redis conversation memory
├── api/main.py                   ← FastAPI REST API
├── streamlit_app/app.py          ← Chat UI
├── notebooks/                    ← Training notebooks
├── tests/                        ← 32 pytest tests
└── requirements.txt
```

## 🚀 Quick Start

### 1 — Clone the repo
```bash
git clone https://github.com/Soberrex/Intent-Classificaton-Chatbot.git
cd Intent-Classificaton-Chatbot
```

### 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### 3 — Download the model
Download the trained model from Hugging Face:
👉 [RexTheSlasher/intent-classifier-bert](https://huggingface.co/RexTheSlasher/intent-classifier-bert)

Place it at: `models/bert-intent-model/`

### 4 — Start Redis
Install and start Memurai (Windows) or Redis

### 5 — Run the API
```bash
uvicorn api.main:app --reload --port 8000
```

### 6 — Run the UI
```bash
streamlit run streamlit_app/app.py
```

## 📊 Dataset
| Dataset | Intents | Train Samples |
|---|---|---|
| Banking77 | 77 | 10,003 |
| CLINC150 | 149 | 7,550 |
| **Combined** | **227** | **17,553** |

## 🔗 Live Demo
👉 [Live Demo Link] (coming soon)

## 📓 Training
See `notebooks/` for complete data preparation and model training pipeline.

## 🧪 Testing
```bash
pytest tests/ -v
```

## 📈 Training Progress
| Epoch | Val Accuracy | Val F1 |
|---|---|---|
| 5 | 84.4% | 82.9% |
| 8 | 87.1% | 86.2% |
| 10 | 87.7% | 86.9% |
| 12 | **88.0%** | **87.3%** |