# Web demo of the Trump table (Hugging Face Spaces compatible: listens on port 7860).
FROM python:3.11-slim

# Hugging Face runs containers as user 1000
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

# CPU-only torch (much smaller than the default CUDA build); the networks are tiny
RUN pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt

COPY --chown=user . .

EXPOSE 7860
# --demo: private in-memory stats per visitor; sessions are capped and closed when idle
CMD ["python", "-m", "webapp.server", "--demo", "--host", "0.0.0.0", "--port", "7860", "--no-browser", "--max-sessions", "10", "--idle-minutes", "30"]
