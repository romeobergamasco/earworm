# Earworm API image (Render free web service; any Docker host).
FROM python:3.12-slim

# Spaces run the container as user 1000
RUN useradd -m -u 1000 app
WORKDIR /app

COPY server/requirements.txt server/requirements.txt
RUN pip install --no-cache-dir -r server/requirements.txt

COPY earworm/ earworm/
COPY server/ server/
COPY web/ web/

USER app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
EXPOSE 7860
# SUPABASE_URL, SUPABASE_SECRET_KEY, ALLOWED_ORIGINS come from the host's settings.
# The host may impose the port through $PORT (Render does); 7860 otherwise.
CMD ["sh", "-c", "exec uvicorn server.main:app --host 0.0.0.0 --port ${PORT:-7860} --proxy-headers --forwarded-allow-ips '*'"]
