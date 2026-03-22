# Black Box Explainer

Black Box Explainer is a Streamlit app for explaining why an Ollama-served model produced a given answer.

It is designed to work cleanly as a hosted website backed by any reachable Ollama server.

## Recommended Fast Model

Default fast model:
- `phi3:mini`

If you want the fastest local experience, pull it first:

```powershell
ollama pull phi3:mini
```

## Environment Variables

Use `.env` locally or Streamlit secrets in production.

```env
BBE_BASE_URL=https://your-ollama-server.example.com
BBE_MODEL=phi3:mini
BBE_TEMPERATURE=0.1
BBE_MAX_TOKENS=320
BBE_TIMEOUT_SECONDS=35
BBE_CRITIQUE_PASS=false
```

## Local Run

```powershell
cd C:\Users\isobu\OneDrive\Desktop\Projects\blackbox_explainer
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## One-Click Launch on Windows

From the project folder, run:

```powershell
.\start-blackbox-explainer.ps1
```

Or double-click:

```text
open-blackbox-explainer.cmd
```

The launcher uses the local `.venv`, starts Streamlit, and opens the app at `http://127.0.0.1:8501`.
It also starts `ollama serve` automatically when Ollama is not already running and points the app at `http://127.0.0.1:11434`.

## Current Behavior

- simple Streamlit frontend
- server URL + model settings
- answer + black box + evidence + context tabs
- follow-up chat using the same Ollama model
