import pickle
import pytest
import pandas as pd
from sklearn.metrics import f1_score


# ── Fixture: loads model once, shared across tests ────────
@pytest.fixture(scope="module")
def model():
    
  # @pytest.fixture: special function that provides test data/objects
      # scope="module": the model loads once for all tests in this file
      # Without scope="module", it would reload the model for each test — slow

    with open("model/pipeline.pkl", "rb") as f:
        return pickle.load(f)


# ── Test 1: Model loads correctly ─────────────────────────
def test_model_loads(model):
    
  # Simplest test: does the model file exist and load without errors?
      # If training failed silently, this catches it immediately

    assert model is not None


# ── Test 2: Output format is correct ─────────────────────
def test_prediction_format(model):
    result = model.predict(["You have won a free prize!"])
    assert len(result) == 1
    assert result[0] in ["spam", "ham"]
    
  # Verifies the model returns the expected label format
      # If someone changes label encoding in preprocessing, this fails loudly



# ── Test 3: Minimum accuracy threshold ───────────────────
def test_model_accuracy(model):
    
  # This is the most important MLOps test: performance regression guard
      # If someone changes the model and accuracy drops below 90%, CI FAILS
      # This prevents "improved" code that actually degrades model quality

    df = pd.read_csv("data/processed/spam.csv")
    sample = df.sample(200, random_state=42)
    
  # Use a small random sample — fast enough for CI
      # random_state=42 makes the same 200 rows selected every run


    preds = model.predict(sample["text"])
    f1 = f1_score(sample["label"], preds, pos_label="spam")

    
  # The assertion below is the "quality gate"
      # In a real company, this threshold is set by the product team
      # "Our spam filter must catch at least 90% of spam (F1 ≥ 0.90)"

    assert f1 >= 0.90, f"F1 score {f1:.4f} is below threshold 0.90"


# ── Test 4: Obvious spam should be caught ─────────────────

def test_obvious_spam(model):
    spam_texts = [
        "FREE FREE FREE! Click now to claim your reward",
        "Congratulations! You won a £1000 Tesco gift card. Call now",
        "URGENT! You have won a guaranteed prize. Claim now",
    ]
    preds = model.predict(spam_texts)
    assert all(p == "spam" for p in preds)    