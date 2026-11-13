# server (Hugging Face Spaces)

API FastAPI (`server/app.py`) qui charge l'index depuis Supabase en tâche de fond, le recharge quand la version du catalogue change (`server/store.py`), et sert la PWA de `web/`.

En local (lit `.env`) :

```bash
.venv/Scripts/python.exe -m uvicorn server.main:app --port 7860
```

Puis http://localhost:7860/api/health et la doc interactive sur http://localhost:7860/api/docs.
