import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import pickle
import json

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ── Load hyperparameters from params.yaml ──────────────────
with open("params.yaml") as f:
    params = yaml.safe_load(f)

ALPHA        = params["model"]["alpha"]
MAX_FEATURES = params["features"]["max_features"]
TEST_SIZE    = params["data"]["test_size"]
RANDOM_STATE = params["data"]["random_state"]

# ── Load data ──────────────────────────────────────────────
df = pd.read_csv("data/processed/spam.csv")
X  = df["text"]
y  = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# ── Build pipeline ─────────────────────────────────────────
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=MAX_FEATURES)),
    ("clf",   MultinomialNB(alpha=ALPHA))
])

# ── Train with MLflow tracking ─────────────────────────────
mlflow.set_experiment("spam-classifier")

with mlflow.start_run():
    mlflow.log_params({
        "alpha": ALPHA,
        "max_features": MAX_FEATURES,
        "test_size": TEST_SIZE
    })

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, pos_label="spam")

    mlflow.log_metrics({"accuracy": acc, "f1_spam": f1})
    mlflow.sklearn.log_model(pipeline, "model")

    with open("model/pipeline.pkl", "wb") as f:
        pickle.dump(pipeline, f)

    print(f"Accuracy: {acc:.4f}  |  F1(spam): {f1:.4f}")
    print(classification_report(y_test, y_pred))

# ── Save metrics for DVC ───────────────────────────────────
with open("metrics.json", "w") as f:
    json.dump({"accuracy": float(acc), "f1_spam": float(f1)}, f)