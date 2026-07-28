# 🛡️ Spam Classifier — End-to-End MLOps Pipeline

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green?logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Containerised-2496ED?logo=docker)
![Kubernetes](https://img.shields.io/badge/Kubernetes-k3s-326CE5?logo=kubernetes)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow)
![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-945DD6)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?logo=githubactions)
![AWS](https://img.shields.io/badge/AWS-EC2-FF9900?logo=amazonaws)
![License](https://img.shields.io/badge/License-MIT-yellow)

> A production-grade MLOps pipeline for SMS/email spam detection — from raw data to a live, authenticated web application with automated CI/CD, experiment tracking, and Kubernetes deployment on AWS EC2.

🔗 **Live Demo:** [https://spam-classifier.duckdns.org](https://spam-classifier.duckdns.org)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [ML Pipeline](#ml-pipeline)
- [API Endpoints](#api-endpoints)
- [CI/CD Pipeline](#cicd-pipeline)
- [MLOps Features](#mlops-features)
- [Authentication](#authentication)
- [Deployment](#deployment)
- [Results](#results)
- [Getting Started](#getting-started)

---

## Overview

This project demonstrates a complete MLOps workflow for a real-world text classification problem. It is not just a model — it is a fully automated, reproducible, and monitored production system.

**What it does:**
- Classifies SMS/email messages as **spam** or **ham** (legitimate)
- Serves predictions via a REST API with JWT authentication
- Provides a web frontend with username/password and Google OAuth login
- Automatically tests, builds, and deploys on every git push

**Dataset:** UCI SMS Spam Collection — 5,574 messages (5,169 after deduplication)

---

## Architecture

```
User Browser
     │
     ▼
https://spam-classifier.duckdns.org
     │
     ▼
Traefik Ingress (k3s) — SSL termination
     │
     ▼
FastAPI Container (Kubernetes Pod)
     │
     ├── /          → Frontend (HTML/CSS/JS)
     ├── /login     → JWT Authentication
     ├── /auth/google → Google OAuth
     └── /predict   → ML Inference
          │
          ▼
     scikit-learn Pipeline
     (TF-IDF → MultinomialNB)
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML | scikit-learn, TF-IDF, Naive Bayes |
| API | FastAPI, Uvicorn, Pydantic |
| Auth | JWT (PyJWT), Google OAuth 2.0 |
| Container | Docker |
| Orchestration | Kubernetes (k3s) |
| CI/CD | GitHub Actions |
| Experiment Tracking | MLflow |
| Data Versioning | DVC |
| Cloud | AWS EC2 |
| Reverse Proxy | Traefik, Nginx |
| SSL | Let's Encrypt (certbot) |
| DNS | DuckDNS |
| Frontend | HTML, CSS, JavaScript |

---

## Project Structure

```
spam-classifier/
├── api/
│   └── main.py              # FastAPI app — /predict, /login, /auth/google
├── src/
│   ├── train.py             # Training script with MLflow tracking
│   └── preprocess.py        # Data cleaning and feature engineering
├── static/
│   └── index.html           # Frontend with JWT + Google OAuth
├── tests/
│   ├── test_model.py        # Model quality tests (accuracy gate)
│   └── test_api.py          # API integration tests
├── .github/
│   └── workflows/
│       └── ci_cd.yml        # GitHub Actions pipeline
├── data/
│   ├── raw/                 # Original dataset (DVC tracked)
│   └── processed/           # Cleaned data (DVC tracked)
├── model/
│   └── pipeline.pkl         # Trained scikit-learn pipeline
├── Dockerfile               # Container definition
├── dvc.yaml                 # DVC pipeline stages
├── params.yaml              # Hyperparameters
└── requirements.txt         # Pinned dependencies
```

---

## ML Pipeline

### Feature Engineering
- **TF-IDF Vectorizer** — converts raw text to numeric matrix
  - `max_features=5000` — vocabulary limited to top 5000 words
  - Words like "FREE", "prize", "WIN" score high (rare in normal messages)
  - Common words like "the", "is" score low (appear everywhere)

### Model
- **Multinomial Naive Bayes** — probabilistic text classifier
  - `alpha=0.1` — Laplace smoothing (prevents zero probabilities)
  - Fast inference — under 15ms per prediction
  - Interpretable — can explain why a message is spam

### scikit-learn Pipeline
```python
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000)),
    ("clf",   MultinomialNB(alpha=0.1))
])
```

The pipeline ensures identical transformations during training and inference.

---

## API Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| GET | `/health` | None | Health check — used by Kubernetes liveness probe |
| POST | `/login` | None | Username/password login — returns JWT |
| GET | `/auth/google` | None | Redirect to Google OAuth |
| GET | `/auth/google/callback` | None | Google OAuth callback — sets auth cookie |
| POST | `/predict` | JWT | Classify text as spam or ham |
| GET | `/` | None | Serve frontend |

### Example Request
```bash
curl -X POST https://spam-classifier.duckdns.org/predict \
  -H "Authorization: Bearer <your_jwt_token>" \
  -H "Content-Type: application/json" \
  -d '{"text": "Congratulations! You have won a £1000 prize. Click now!"}'
```

### Example Response
```json
{
  "label": "spam",
  "confidence": 0.9987,
  "latency_ms": 12.4
}
```

---

## CI/CD Pipeline

Every push to `main` triggers a 3-stage automated pipeline:

```
git push
    │
    ▼
┌─────────────────────────────┐
│  JOB 1: TEST                │
│  - Install dependencies     │
│  - Preprocess data          │
│  - Train model              │
│  - Run 11 pytest tests      │
│  - Quality gate: F1 ≥ 0.90  │
└────────────┬────────────────┘
             │ (only if tests pass)
             ▼
┌─────────────────────────────┐
│  JOB 2: BUILD               │
│  - docker build             │
│  - docker push → ghcr.io   │
│  - Tag: latest + git SHA    │
└────────────┬────────────────┘
             │ (only if build passes)
             ▼
┌─────────────────────────────┐
│  JOB 3: DEPLOY              │
│  - SSH into EC2             │
│  - Pull new image           │
│  - Restart container        │
│  - Health check /health     │
└─────────────────────────────┘
```

**Quality Gate:** If model F1 drops below 0.90, CI fails and deployment is blocked.

---

## MLOps Features

### Experiment Tracking (MLflow)
Every training run automatically logs:
- **Parameters:** alpha, max_features, test_size
- **Metrics:** accuracy, F1 score
- **Artifacts:** trained model file

Experiment comparison that led to the best model:

| Run | Alpha | Accuracy | F1 (spam) |
|-----|-------|----------|-----------|
| 1 | 1.0 | 96.8% | 0.872 |
| 2 | 0.5 | 97.8% | 0.914 |
| 3 | **0.1** | **98.4%** | **0.939** |

### Data Versioning (DVC)
- Raw dataset tracked with MD5 hash — any data change is detected
- Pipeline stages defined in `dvc.yaml` — only changed stages rerun
- `dvc metrics diff` shows exact metric changes between experiments
- Full reproducibility — any past experiment can be recreated exactly

### Hyperparameter Management
All parameters stored in `params.yaml` — no hardcoded values in code:
```yaml
data:
  test_size: 0.2
  random_state: 42
features:
  max_features: 5000
model:
  alpha: 0.1
```

---

## Authentication

### JWT (Username/Password)
1. POST `/login` with credentials
2. Receive JWT token (24h expiry)
3. Include `Authorization: Bearer <token>` in all API calls

### Google OAuth 2.0
1. Click "Sign in with Google"
2. Redirect to Google authorization server
3. Google returns authorization code to `/auth/google/callback`
4. Backend exchanges code for user info
5. Backend generates JWT, sets secure cookie
6. User is redirected to app — logged in

---

## Deployment

**Infrastructure:**
- AWS EC2 t2.micro (Amazon Linux 2023)
- k3s Kubernetes — single-node cluster
- Traefik ingress controller
- Let's Encrypt SSL certificate
- DuckDNS for free domain

**Deployment flow:**
```bash
# GitHub Actions SSHs into EC2 and runs:
docker pull ghcr.io/farzeen-05/spam-classifier:latest
docker stop spam-api && docker rm spam-api
docker run -d --name spam-api -p 8001:8000 \
  --restart unless-stopped \
  ghcr.io/farzeen-05/spam-classifier:latest
curl -f http://localhost:8001/health
```

---

## Results

| Metric | Value |
|--------|-------|
| Dataset size | 5,169 messages |
| Train/Test split | 80% / 20% |
| Accuracy | **98.4%** |
| F1 Score (spam) | **0.939** |
| Precision (spam) | 98% |
| Recall (spam) | 90% |
| Inference latency | < 15ms |
| Tests passing | 11 / 11 |

---

## Getting Started

### Prerequisites
- Python 3.11+
- Docker
- Git

### Local Setup

```bash
# Clone the repository
git clone https://github.com/farzeen-05/spam-classifier.git
cd spam-classifier

# Install dependencies
pip install -r requirements.txt

# Download dataset and preprocess
python src/preprocess.py

# Train the model
python src/train.py

# Run tests
python -m pytest tests/ -v

# Start the API
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Run with Docker

```bash
docker build -t spam-classifier:v1 .
docker run -p 8000:8000 spam-classifier:v1
```

Open `http://localhost:8000` in your browser.

### Run DVC Pipeline

```bash
dvc repro          # Run full pipeline
dvc metrics show   # View current metrics
dvc metrics diff   # Compare with previous run
```

---

## Author

**Farzeen Abdul Khadir**
ECE Graduate | ML & Full-Stack Developer | MLOps & Cloud

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?logo=linkedin)](https://www.linkedin.com/in/farzeen-abdul-khadir-8921ba2a1)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?logo=github)](https://github.com/farzeen-05)

---

## License

This project is licensed under the MIT License.
