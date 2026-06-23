import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import pickle
import json
import time
import sqlite3
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import jwt

app = FastAPI(title="Spam Classifier API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config ─────────────────────────────────────────────────
SECRET_KEY = "spam-classifier-secret-key-2026"
ALGORITHM = "HS256"
TOKEN_EXPIRE_HOURS = 24

# ── Fake user DB (in production use real DB) ───────────────
USERS = {
    "farzeen": "password123",
    "admin": "admin123"
}

# ── Load model ─────────────────────────────────────────────
try:
    with open("model/pipeline.pkl", "rb") as f:
        model = pickle.load(f)
    print("✅ Model loaded")
except FileNotFoundError:
    model = None

security = HTTPBearer()

# ── Schemas ────────────────────────────────────────────────
class LoginRequest(BaseModel):
    username: str
    password: str

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    label: str
    confidence: float
    latency_ms: float

# ── JWT helpers ────────────────────────────────────────────
def create_token(username: str) -> str:
    payload = {
        "sub": username,
        "exp": datetime.utcnow() + timedelta(hours=TOKEN_EXPIRE_HOURS)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(
            credentials.credentials,
            SECRET_KEY,
            algorithms=[ALGORITHM]
        )
        return payload["sub"]
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# ── Routes ─────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model_ready": model is not None}

@app.post("/login")
def login(request: LoginRequest):
    if request.username not in USERS:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if USERS[request.username] != request.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_token(request.username)
    return {"access_token": token, "token_type": "bearer"}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, username: str = Depends(verify_token)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    start = time.perf_counter()
    proba = model.predict_proba([request.text])[0]
    classes = model.classes_
    pred_idx = proba.argmax()
    label = classes[pred_idx]
    confidence = float(proba[pred_idx])
    latency_ms = (time.perf_counter() - start) * 1000

    return PredictResponse(
        label=label,
        confidence=round(confidence, 4),
        latency_ms=round(latency_ms, 2)
    )

# ── Serve frontend ─────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")