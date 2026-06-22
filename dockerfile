# ── STAGE 1: Base image ───────────────────────────────────
FROM python:3.11-slim

  # FROM: which image to start with — like choosing a base OS
  # python:3.11-slim: official Python image, "slim" = minimal Debian Linux
  # Slim saves ~400MB vs full image. Has no dev tools but runs Python fine.
  # Never use :latest — version pins the Python interpreter itself


# ── STAGE 2: Set working directory ───────────────────────
WORKDIR /app

  # All subsequent commands run inside /app
  # If /app doesn't exist, Docker creates it
  # Convention: most production containers use /app


# ── STAGE 3: Copy and install dependencies FIRST ─────────
COPY requirements.txt .

  # Copies requirements.txt from your machine → /app/requirements.txt in container
  # We do this BEFORE copying src/ — critical for Docker layer caching


RUN pip install --no-cache-dir -r requirements.txt

  # RUN executes a shell command during the build
  # --no-cache-dir: don't cache downloaded packages inside the image
  #   This saves ~200MB from the final image size
  # WHY COPY requirements BEFORE code:
  #   Docker builds in layers. If requirements.txt hasn't changed, Docker
  #   reuses the cached "pip install" layer — saves 2-3 minutes per build.
  #   If you COPY everything first, ANY code change forces pip reinstall.


# ── STAGE 4: Copy application code ───────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY api/ ./api/
COPY params.yaml .
COPY data/processed/ ./data/processed/
RUN mkdir -p model && python src/train.py

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
  # CMD: the default command when the container starts
  # uvicorn: ASGI server (like gunicorn but for async apps)
  # api.main:app = module "api/main.py", variable named "app"
  # --host 0.0.0.0: listen on ALL network interfaces inside the container
  #   If you use 127.0.0.1 (localhost), requests from outside the container
  #   cannot reach it — a very common beginner mistake
  # --port 8000: must match EXPOSE above
