import pickle
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

  # Pydantic: validates request data automatically
  # If the user sends {"text": 123} (number instead of string), FastAPI
  # rejects it with a clear error message before your code even runs



# ── App initialisation ────────────────────────────────────
app = FastAPI(
    title="Spam Classifier API",
    version="1.0.0",
    description="Classifies text as spam or ham"
)

  # FastAPI auto-generates docs at /docs (Swagger UI) and /redoc
  # The CI pipeline can hit /docs to verify the API is alive after deployment



# ── Load model at startup ─────────────────────────────────

  # This runs ONCE when the container starts — not on every request
  # Loading a model takes 50-200ms. If you loaded on every request,
  # your API would be extremely slow (500ms+ per prediction)


MODEL_PATH = "model/pipeline.pkl"

try:
    with open(MODEL_PATH, "rb") as f:  
# "rb" = read bytes

        model = pickle.load(f)
    print("✅ Model loaded successfully")
except FileNotFoundError:
    print("❌ Model file not found — train first with: python src/train.py")
    model = None
    
  # Don't crash the app — let /health report the issue
      # In CI, a missing model should cause a test failure, not a crash loop



# ── Request/Response schemas ──────────────────────────────
class PredictRequest(BaseModel):
    text: str                 
# Required: the email/SMS text


class PredictResponse(BaseModel):
    label: str                
# "spam" or "ham"

    confidence: float         
# Probability 0.0 → 1.0

    latency_ms: float         
# How long prediction took — for monitoring



# ── Health check endpoint ─────────────────────────────────
@app.get("/health")
def health():
    
  # CI/CD hits this endpoint after deployment to verify the app is alive
      # Load balancers also use this to route traffic away from unhealthy pods
      # Kubernetes liveness probe checks this every 10 seconds

    return {
        "status": "ok" if model is not None else "model_not_loaded",
        "model_ready": model is not None
    }


# ── Prediction endpoint ───────────────────────────────────
@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    
  # @app.post: this route handles POST requests to /predict
      # response_model: FastAPI validates output matches PredictResponse schema


    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
  # 503 = Service Unavailable. 503 tells the caller to retry later.
          # 500 would mean "I crashed" — different semantic meaning.


    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
        
  # 400 = Bad Request. Caller's fault — they sent bad data.


    start = time.perf_counter()
    
  # perf_counter: high-resolution timer (nanosecond precision)
      # More accurate than time.time() for measuring latency


    proba = model.predict_proba([request.text])[0]
    
  # predict_proba returns [[P(ham), P(spam)]] — a 2D array
      # [0] gets the first (only) row
      # Result: array([0.03, 0.97]) means 97% spam


    classes = model.classes_         
# ["ham", "spam"] — order from training

    pred_idx = proba.argmax()        
# Index of highest probability

    label    = classes[pred_idx]     
# "spam" or "ham"

    confidence = float(proba[pred_idx])

    latency_ms = (time.perf_counter() - start) * 1000

    return PredictResponse(
        label=label,
        confidence=round(confidence, 4),
        latency_ms=round(latency_ms, 2)
    )
