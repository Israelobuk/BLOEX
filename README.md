# BLOEX

BLOEX is a web app for auditing whether an answer actually holds up.

Paste the original question, the answer you want to inspect, and any supporting context. BLOEX turns that into a readable review of what the answer gets right, what it assumes, what the evidence actually supports, and where the response is still weak.

The product is built to make black-box answers easier to inspect without forcing users to dig through prompt chains, hidden assumptions, or raw model output.

## What BLOEX Does

- reviews a pasted answer against the user's original question
- checks whether the answer is supported by the context you provide
- separates the result into verdict, reasoning, support, and risk
- runs fully on your machine with a local model through Ollama
- keeps a lightweight predictive engine in the same repo for structured analysis workflows

## Product Flow

1. Enter the original question.
2. Paste the answer you want to inspect.
3. Add optional context, notes, quotes, or source material.
4. Run the audit.
5. Review the result across:
   - `Answer`
   - `Why The Model Said It`
   - `Supporting Context`
   - `Gaps & Risks`

## What Makes It Different

Most answer-review tools stop at a single generated verdict.

BLOEX is designed to do more than that. It separates the result into clear parts, checks the answer against the context you provide, and routes the backend through a more controlled workflow before anything is shown in the UI.

That keeps the front end simple while making the underlying audit process more deliberate and more resilient.

## Tech Stack

### Frontend

- React 18
- Vite 5
- Plain CSS

### Backend

- FastAPI
- Uvicorn
- Pydantic-style FastAPI models
- Python Dotenv
- Requests

### Model Runtime

- Ollama

### Orchestration and Analysis

- LangGraph for stateful audit orchestration
- Pandas for structured signal analysis
- RapidFuzz for fuzzy matching
- aiohttp for concurrent model calls in the predictive engine

### Persistence

- SQLite for local history and cached analysis state

### Testing

- Pytest
- HTTPX

## How It Works

### Explain Flow

The main audit path lives under `explain/` and is served through `/api/explain`.

LangGraph controls the workflow around Ollama:

1. `generate_primary`
2. `evaluate_quality`
3. `repair_output`
4. `finalize`

Ollama handles inference. LangGraph handles state, routing, retries, and completion logic.

### Predictive Engine

The repo also includes a lightweight structured analysis engine under `api/` and `core/`.

It supports:

- recursive branch analysis
- SQLite-backed history
- fuzzy matching support
- model-assisted branch explanations

Available endpoints:

- `POST /api/analyze`
- `GET /api/analysis/{analysis_id}`
- `GET /api/history`
- `GET /api/health`

## Product Experience

BLOEX is meant to feel simple from the user side:

- drop in a question, an answer, and optional context
- get back a readable audit instead of a blob of model confidence
- move between verdict, reasoning, support, and risk without leaving the page
- keep the product focused on answer quality, not prompt tinkering

The app is built to present a clean interface on top of a more careful backend workflow.

## Deployment

This repo is structured as a deployable web product:

- a React frontend for the main user experience
- a FastAPI backend for audit requests and analysis routes
- a model runtime through Ollama
- a stateful orchestration layer through LangGraph
- lightweight persistence through SQLite

Docker and Render configuration are already included in the repo for deployment workflows.

## Repository Layout

```text
BLOEX/
  frontend/     Web interface
  backend/      FastAPI application entrypoint
  explain/      Audit prompts, parsing, orchestration, output shaping
  api/          Predictive analysis routes and schemas
  core/         Analysis logic, persistence, memory, model helpers
  data/         Local runtime data
  llm/          Model client layer
  tests/        Test coverage
  utils/        Shared helpers
```

## Summary

- Ollama provides the inference layer.
- LangGraph manages the stateful audit workflow around that inference.
- SQLite keeps the application state lightweight.
- The predictive engine remains part of the repo, but the main product surface is the audit experience.


