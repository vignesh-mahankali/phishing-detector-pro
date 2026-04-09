# ============================================================
#   PHISHING DETECTOR PRO — FastAPI Backend
#   Run locally : uvicorn main:app --reload
#   Deploy      : Render.com (free tier)
# ============================================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import re
import os

# ============================================================
# APP SETUP
# ============================================================
app = FastAPI(
    title="Phishing Detector Pro",
    description="AI-powered phishing and scam message detector",
    version="1.0.0",
)

# Allow requests from your web UI (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # in production, replace with your UI's URL
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# LOAD MODEL ON STARTUP
# ============================================================
MODEL_PATH = "vignesh-mahankali/phishing-detector-pro"
import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
THRESHOLD  = 0.75                     # confidence cutoff

device    = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
model     = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH, weights_only=False)
model.to(device)
model.eval()

print(f"✅ Model loaded on {device.upper()}")
print(f"✅ API ready — threshold: {THRESHOLD}")


# ============================================================
# REQUEST / RESPONSE SCHEMAS
# ============================================================
class PredictRequest(BaseModel):
    text: str
    url:  str = ""   # optional URL field

class PredictResponse(BaseModel):
    label:      str    # "PHISHING" or "LEGITIMATE"
    confidence: float  # 0.0 to 1.0
    verdict:    str    # "danger" | "warning" | "safe"
    flags:      list   # list of reason strings
    is_phishing: bool


# ============================================================
# HELPER — RULE-BASED FLAGS (explains WHY it was flagged)
# ============================================================
def extract_flags(text: str, url: str, is_phishing: bool) -> list:
    flags = []
    lower = text.lower()

    if is_phishing:
        if any(w in lower for w in ["urgent", "immediately", "act now", "expire", "suspended", "limited"]):
            flags.append("Urgency language detected")
        if any(w in lower for w in ["won", "winner", "lottery", "prize", "million", "claim"]):
            flags.append("Prize or lottery scam pattern")
        if any(w in lower for w in ["verify", "confirm", "update your", "click here", "sign in"]):
            flags.append("Account verification request")
        if any(w in lower for w in ["password", "ssn", "bank details", "credit card", "wire transfer"]):
            flags.append("Sensitive information being requested")
        if re.search(r"http://|bit\.ly|tinyurl|\.xyz|\.tk|\.ml|\.ga", lower):
            flags.append("Suspicious or shortened URL detected")
        if url and re.search(r"http://|\.xyz|\.tk|\.ml", url.lower()):
            flags.append("Unsafe URL scheme detected")
        if any(w in lower for w in ["dear customer", "dear user", "dear member", "dear beneficiary"]):
            flags.append("Generic impersonal greeting")
        if not flags:
            flags.append("AI model detected phishing pattern")
    else:
        flags.append("No phishing indicators detected")
        if any(w in lower for w in ["order", "shipped", "delivered", "dispatch"]):
            flags.append("Looks like a legitimate order notification")
        if any(w in lower for w in ["otp", "debited", "credited", "balance", "transaction"]):
            flags.append("Looks like a legitimate bank notification")
        if any(w in lower for w in ["meeting", "assignment", "portal", "reminder"]):
            flags.append("Looks like a legitimate reminder")

    return flags


# ============================================================
# HELPER — CLEAN TEXT (same as training)
# ============================================================
def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\S+", "URL", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:512]


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def root():
    return {
        "status": "online",
        "model":  "DistilBERT Phishing Detector Pro",
        "accuracy": "98.94%",
        "f1_score": "98.57%",
        "trained_on": "28,301 messages",
    }


@app.get("/health")
def health():
    return {"status": "healthy", "device": device}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    # Clean input
    cleaned = clean_text(request.text)
    if not cleaned:
        return PredictResponse(
            label="LEGITIMATE",
            confidence=0.0,
            verdict="safe",
            flags=["Empty message"],
            is_phishing=False,
        )

    # Tokenize and run model
    inputs = tokenizer(
        cleaned,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs   = torch.softmax(outputs.logits, dim=-1)

    spam_prob   = probs[0][1].item()
    is_phishing = spam_prob >= THRESHOLD

    # Determine verdict tier
    if spam_prob >= 0.85:
        verdict = "danger"
    elif spam_prob >= 0.60:
        verdict = "warning"
    else:
        verdict = "safe"

    # Extract explanation flags
    flags = extract_flags(request.text, request.url, is_phishing)

    return PredictResponse(
        label       = "PHISHING" if is_phishing else "LEGITIMATE",
        confidence  = round(spam_prob, 4),
        verdict     = verdict,
        flags       = flags,
        is_phishing = is_phishing,
    )


# ============================================================
# BATCH PREDICT (for future dashboard feature)
# ============================================================
class BatchRequest(BaseModel):
    messages: list[str]

@app.post("/predict/batch")
def predict_batch(request: BatchRequest):
    results = []
    for text in request.messages[:20]:   # cap at 20 per batch
        r = predict(PredictRequest(text=text))
        results.append(r)
    return {"results": results, "total": len(results)}
