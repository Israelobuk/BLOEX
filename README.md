# Black Box Explainer

Black Box Explainer is an LLM answer analysis tool built to help users inspect, question, and better understand AI-generated responses.

Instead of treating an LLM answer as something to blindly trust, the project is designed to break the response down into a more interpretable format:

- what the answer is saying
- why the model likely landed on that answer
- what assumptions the answer depends on
- where the reasoning may still be weak
- what follow-up questions a user should ask next

The goal is to make black-box model behavior easier to evaluate, especially when a user wants to understand whether an answer sounds plausible, overconfident, incomplete, or weakly supported.

## What The Project Does

The app takes:

- the original user question
- the LLM answer being reviewed

and returns a structured analysis that helps the user inspect the response rather than simply accept it at face value.

The output is organized into a few core views:

- `Answer`: a direct audit-style read of the response
- `Why The Model Said It`: a plain-language explanation of what the model likely focused on
- `Supporting Context`: extracted support signals and evidence claims when available
- `Gaps & Next Questions`: assumptions, uncertainty, and suggested follow-up questions

## Why It Matters

Large language models can produce answers that sound polished even when the reasoning is incomplete, weakly supported, or overly confident.

This project explores a practical interpretability workflow for AI outputs by helping users:

- inspect how an answer was formed
- identify weak reasoning or unsupported claims
- see what the answer appears to rely on
- ask sharper follow-up questions

## Current Scope

This is a local-first project designed around a local Ollama backend.

The full workflow works best when run on your own machine with a local model. Public deployment is possible for the frontend, but hosted Ollama inference on free-tier infrastructure is memory-constrained, so the most reliable version of the project is the local setup described below.

## Local Setup

### 1. Start Ollama

If Ollama is not already running:

```powershell
ollama serve
```

Pull the model once if needed:

```powershell
ollama pull tinyllama:latest
```

### 2. Run the backend

```powershell
cd backend
.\.venv\Scripts\activate
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### 3. Run the frontend

```powershell
cd frontend
npm run dev
```

### 4. Open the app

```text
http://127.0.0.1:5173
```

## Example Workflow

1. Paste the original question that was asked to the LLM.
2. Paste the exact LLM answer you want to inspect.
3. Run the explainer.
4. Review the result across the explanation tabs.
5. Use the follow-up section to pressure-test weak points in the response.

## Project Structure

```text
blackbox_explainer/
  backend/      API layer
  frontend/     User interface
  explain/      Core explanation and analysis pipeline
  llm/          LLM client interfaces
  utils/        Shared utilities
  config.py     Shared configuration
```

## Main Ideas Behind The Project

- LLM answers should be inspectable, not just readable.
- Interpretability tools should help a user challenge an answer, not just restate it.
- Structured analysis makes model behavior easier to evaluate than a plain paragraph response.

## Status

The active version of this project is the React frontend plus Python backend workflow in:

- `frontend/`
- `backend/`
- `explain/`
- `llm/`
- `utils/`

Older files from earlier experiments may still exist in the repo, but they are not the main app path.

## GitHub Notes

Files and folders that belong in the repo:

- `frontend/`
- `backend/`
- `explain/`
- `llm/`
- `utils/`
- `config.py`
- `README.md`
- `.gitignore`

Files and folders that should not be committed:

- `.venv/`
- `backend/.venv/`
- `frontend/node_modules/`
- `frontend/dist/`
- `frontend/.vite-cache/`
- `.env`
- `__pycache__/`
- `db.sqlite3`
