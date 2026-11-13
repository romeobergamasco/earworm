# Earworm server image (Hugging Face Spaces, Docker SDK).
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
# SUPABASE_URL and SUPABASE_SECRET_KEY come from the Space's secrets
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "7860", "--proxy-headers"]
