# Black Box Explainer

Black Box Explainer uses:

- `frontend/` for the React app
- `backend/` for the FastAPI API
- `explain/`, `llm/`, `utils/`, and `config.py` for the shared explainer pipeline

The active product is the black React UI, not the older Streamlit surface.

## Local Dev

Start Ollama:

```powershell
ollama serve
```

Pull the model once if needed:

```powershell
ollama pull phi3:mini
```

Run the backend:

```powershell
cd C:\Users\isobu\OneDrive\Desktop\Projects\blackbox_explainer\backend
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Run the frontend:

```powershell
cd C:\Users\isobu\OneDrive\Desktop\Projects\blackbox_explainer\frontend
npm install
npm run dev
```

Then open:

```text
http://127.0.0.1:5173
```

## Deploy Shape

- `frontend` -> Vercel
- `backend + Ollama` -> one Render Docker service

The backend owns the model connection. The browser only talks to the FastAPI API.

## Files To Put On GitHub

Keep these:

- `frontend/`
- `backend/`
- `explain/`
- `llm/`
- `utils/`
- `config.py`
- `Dockerfile`
- `docker-start.sh`
- `.dockerignore`
- `render.yaml`
- `README.md`
- `.gitignore`
- `.env.example`

Do not commit these:

- `.venv/`
- `backend/.venv/`
- `frontend/node_modules/`
- `frontend/dist/`
- `frontend/.vite-cache/`
- `.env`
- `__pycache__/`
- `db.sqlite3`

## Render Deployment

This repo is now set up for a Docker-based backend on Render.

What the backend container does:

1. starts `ollama serve`
2. waits for Ollama to be ready
3. pulls `BBE_MODEL`
4. starts `uvicorn main:app`

Use the included [render.yaml](render.yaml) as a Blueprint.

Important env vars on Render:

```text
BBE_BASE_URL=http://127.0.0.1:11434
BBE_MODEL=phi3:mini
BBE_TEMPERATURE=0.1
BBE_MAX_TOKENS=640
BBE_TIMEOUT_SECONDS=60
BBE_CRITIQUE_PASS=false
FRONTEND_URL=https://your-vercel-site.vercel.app
CORS_ALLOW_ORIGINS=https://your-vercel-site.vercel.app
```

Note:

- This setup is heavier than a normal Python web service.
- A persistent disk is strongly recommended if you do not want Ollama to re-pull the model on cold starts or redeploys.

## Vercel Deployment

Set the Vercel project root to `frontend/`.

Use:

```text
Build command: npm run build
Output directory: dist
```

Add this env var on Vercel:

```text
VITE_API_BASE_URL=https://your-render-backend.onrender.com
```

## Notes

- `app.py` still exists, but it is not the main app path.
- `frontend/src/App.rewrite.jsx` is not part of the active app.
