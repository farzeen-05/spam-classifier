import mlflow                     
# Experiment tracker — records what we tried and what worked

import mlflow.sklearn             
# MLflow plugin that understands scikit-learn models

import pandas as pd
import yaml                        
# Reads params.yaml — keeps hyperparams out of code

import pickle                      
# Serialises (saves) the trained model to a .pkl file

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

  # TF-IDF: converts raw text → numeric matrix
  # TF (Term Frequency): how often a word appears in THIS email
  # IDF (Inverse Document Frequency): penalises words that appear in ALL emails
  # "the" → low IDF score (appears everywhere, not informative)
  # "prize" → high IDF score (rare in normal email, suspicious in spam)

from sklearn.naive_bayes import MultinomialNB

  # Naive Bayes: probabilistic classifier
  # P(spam | words) = P(words | spam) × P(spam) / P(words)
  # "Naive" because it assumes each word is independent (not true, but works well)

from sklearn.pipeline import Pipeline

  # Pipeline chains steps together: vectorize → classify
  # Critical for deployment: the same transformation runs on new data

from sklearn.metrics import accuracy_score, f1_score, classification_report


# ── Load hyperparameters from params.yaml ──────────────────
with open("params.yaml") as f:
    params = yaml.safe_load(f)

  # WHY: If params are hardcoded in train.py, DVC can't track them.
  # With params.yaml, DVC detects when params change and reruns training.
  # Also, the CI pipeline can inject different params for experimentation.


ALPHA = params["model"]["alpha"]         
# Naive Bayes smoothing param

MAX_FEATURES = params["features"]["max_features"]  
# TF-IDF vocabulary size

TEST_SIZE = params["data"]["test_size"]   
# Fraction of data for validation

RANDOM_STATE = params["data"]["random_state"]  
# Seed: same shuffle every run



# ── Load data ──────────────────────────────────────────────
df = pd.read_csv("data/processed/spam.csv")

  # Reads the PROCESSED data, not raw. Raw data is never touched after
  # initial download — this is the immutable source of truth.


X = df["text"]       
# Input: email/SMS text

y = df["label"]      
# Target: "spam" or "ham" (not spam)


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

  # train_test_split: separates data so we evaluate on data the model never saw
  # random_state makes the split deterministic: same seed = same split every run
  # This is critical for reproducibility in CI: you want the exact same evaluation



# ── Build pipeline ────────────────────────────────────────
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=MAX_FEATURES)),
    
  # Step 1: text → sparse matrix of TF-IDF scores

    ("clf",   MultinomialNB(alpha=ALPHA))
    
  # Step 2: matrix → probability(spam)

])

  # WHY A PIPELINE: When you call pipeline.predict("free prizes!"), it
  # automatically runs TF-IDF first, then classifier. Without this, you'd
  # manually call vectorizer.transform() every time — easy to forget in prod.



# ── Train with MLflow tracking ────────────────────────────
mlflow.set_experiment("spam-classifier")

  # Groups all runs under one experiment name in the MLflow UI


with mlflow.start_run():
    
  # Everything inside this block is recorded as one "run"
      # Each run gets a unique run_id you can compare against others


    # Log hyperparameters — what settings did we use?
    mlflow.log_params({
        "alpha": ALPHA,
        "max_features": MAX_FEATURES,
        "test_size": TEST_SIZE
    })
    
  # Now in MLflow UI you can ask: "Which alpha value gave the best F1?"


    # Train
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, pos_label="spam")
    
  # F1 is the right metric for imbalanced spam data:
      # 98% "ham" in dataset → a model that always says "ham" gets 98% accuracy
      # F1 punishes that: it's the harmonic mean of precision and recall


    # Log metrics — what results did we get?
    mlflow.log_metrics({"accuracy": acc, "f1_spam": f1})

    # Log model — save the trained model to MLflow artifact store
    mlflow.sklearn.log_model(pipeline, "model")
    
  # This saves the model so you can load ANY past run's model
      # Critical for rollback: if new model performs worse, load last good run


    # Also save locally for Docker container to pick up
    with open("model/pipeline.pkl", "wb") as f:
        pickle.dump(pipeline, f)
    
  # pickle serialises the Python object to bytes
      # "wb" = write bytes (not text)
      # The FastAPI container reads this file at startup


    print(f"Accuracy: {acc:.4f}  |  F1(spam): {f1:.4f}")
    print(classification_report(y_test, y_pred))