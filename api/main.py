import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import pickle
import json
import time
import sqlite3
import httpx
import os
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import jwt

load_dotenv()

app = FastAPI(title="Spam Classifier API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SECRET_KEY = os.getenv("SECRET_KEY", "spam-classifier-secret-key-2026")
ALGORITHM = "HS256"
TOKEN_EXPIRE_HOURS = 24
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
REDIRECT_URI = "https://spam-classifier.duckdns.org/auth/google/callback"

USERS = {
    "farzeen": "password123",
    "admin": "admin123"
}

try:
    with open("model/pipeline.pkl", "rb") as f:
        model = pickle.load(f)
    print("✅ Model loaded")
except FileNotFoundError:
    model = None

security = HTTPBearer()

class LoginRequest(BaseModel):
    username: str
    password: str

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    label: str
    confidence: float
    latency_ms: float

def create_token(username: str) -> str:
    payload = {
        "sub": username,
        "exp": datetime.utcnow() + timedelta(hours=TOKEN_EXPIRE_HOURS)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        return payload["sub"]
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.get("/health")
def health():
    return {"status": "ok", "model_ready": model is not None}

@app.post("/login")
def login(request: LoginRequest):
    if request.username not in USERS or USERS[request.username] != request.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"access_token": create_token(request.username), "token_type": "bearer"}

# ── Google OAuth ──────────────────────────────────────────
@app.get("/auth/google")
def google_login():
    url = (
        "https://accounts.google.com/o/oauth2/v2/auth"
        f"?client_id={GOOGLE_CLIENT_ID}"
        f"&redirect_uri={REDIRECT_URI}"
        "&response_type=code"
        "&scope=openid email profile"
        "&access_type=offline"
        "&prompt=select_account"
    )
    return RedirectResponse(url)

@app.get("/auth/google/callback")
async def google_callback(code: str):
    async with httpx.AsyncClient() as client:
        # Exchange code for tokens
        token_res = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uri": REDIRECT_URI,
                "grant_type": "authorization_code",
            }
        )
        token_data = token_res.json()
        if "error" in token_data:
            raise HTTPException(400, token_data["error"])

        # Get user info
        userinfo_res = await client.get(
            "https://www.googleapis.com/oauth2/v3/userinfo",
            headers={"Authorization": f"Bearer {token_data['access_token']}"}
        )
        userinfo = userinfo_res.json()

    email = userinfo.get("email")
    name = userinfo.get("name", email)

    if not email:
        raise HTTPException(400, "Could not get email from Google")

    # Issue your own JWT
    payload = {
        "sub": email,
        "email": email,
        "name": name,
        "picture": userinfo.get("picture", ""),
        "exp": datetime.utcnow() + timedelta(hours=TOKEN_EXPIRE_HOURS)
    }
    your_jwt = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

    # Redirect to frontend with token
    return RedirectResponse(
        f"https://spam-classifier.duckdns.org/?token={your_jwt}"
    )

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

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")