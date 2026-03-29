from __future__ import annotations

from dataclasses import replace
import os
import sys
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import load_from_env
from explain.pipeline import ExplainerPipeline, build_fallback_result
from llm import create_client


load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(Path(__file__).resolve().parent / ".env")

FOLLOWUP_SYSTEM_PROMPT = """
You are a helpful assistant who explains things clearly to real users.
Rules:
1) Start with a direct answer first.
2) Use plain, natural language instead of formal or academic wording.
3) Explain things like you are talking to a smart user, not writing a paper.
4) If you use a technical term, explain it simply.
5) Explain what the model seems to be focusing on, missing, overstating, or understating when relevant.
6) Use examples only when they make the answer easier to understand.
7) If the follow-up is unrelated to the context, answer directly.
""".strip()

MODEL_OPTIONS = [
    {"value": "llama3.2:latest", "label": "Llama 3.2", "description": "Balanced default. Closer to the working Ollama setup used in the market analyzer project."},
    {"value": "phi3:mini", "label": "Phi-3 Mini", "description": "Fast and lightweight, but more fragile for longer structured outputs."},
    {"value": "llama3.1:8b", "label": "Llama 3.1 8B", "description": "More capable and detailed. Better when you want stronger reasoning and fuller writeups."},
    {"value": "gpt-oss:120b", "label": "GPT-OSS 120B", "description": "Hosted-scale reasoning model. Best when you want a stronger cloud backend with more depth."},
]


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _cors_origins() -> list[str]:
    raw = os.getenv("CORS_ALLOW_ORIGINS", "").strip()
    if raw:
        if raw == "*":
            return ["*"]
        return [origin.strip() for origin in raw.split(",") if origin.strip()]
    frontend_url = os.getenv("FRONTEND_URL", "").strip()
    if frontend_url:
        return [frontend_url]
    return ["*"]


class ExplainRequest(BaseModel):
    question: str = Field(min_length=1)
    model_answer: str = ""
    context: str = ""
    model: str | None = None


class FollowupRequest(BaseModel):
    question: str = Field(min_length=1)
    model_answer: str = ""
    context: str = ""
    followup: str = Field(min_length=1)
    model: str | None = None


def _load_settings():
    settings = load_from_env()
    base_url = settings.base_url.strip() or "http://127.0.0.1:11434"
    model = settings.model.strip() or "llama3.2:latest"
    return replace(settings, base_url=base_url, model=model)


def _selected_model(requested_model: str | None, default_model: str) -> str:
    allowed_models = {option["value"] for option in MODEL_OPTIONS}
    candidate = (requested_model or default_model or "").strip()
    if not candidate:
        raise HTTPException(status_code=500, detail="Model is not configured on the backend.")
    if candidate not in allowed_models:
        raise HTTPException(status_code=400, detail=f"Unsupported model '{candidate}'.")
    return candidate


def _build_client(model: str):
    settings = _load_settings()
    if not settings.base_url:
        raise HTTPException(status_code=500, detail="Model service URL is not configured on the backend.")
    return create_client(
        base_url=settings.base_url,
        model=model,
        api_key=settings.api_key,
        timeout_seconds=settings.timeout_seconds,
    )


app = FastAPI(title="Black Box Explainer API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    settings = _load_settings()
    model = _selected_model(None, settings.model)
    client = _build_client(model)
    ready, status = client.healthcheck()
    return {
        "ok": ready,
        "status": status,
        "serverUrlLocked": True,
        "serverLabel": settings.base_url,
        "selectedModel": model,
        "models": MODEL_OPTIONS,
        "critiquePass": settings.critique_pass,
    }


@app.get("/api/config")
def config():
    settings = _load_settings()
    return {
        "serverUrlLocked": True,
        "serverLabel": settings.base_url,
        "selectedModel": settings.model,
        "models": MODEL_OPTIONS,
    }


@app.post("/api/explain")
def explain(request: ExplainRequest):
    settings = _load_settings()
    model = _selected_model(request.model, settings.model)
    client = _build_client(model)
    pipeline = ExplainerPipeline(client)

    try:
        result = pipeline.run(
            question=request.question.strip(),
            model_answer=request.model_answer.strip(),
            context=request.context,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            critique_pass=settings.critique_pass,
        )
    except HTTPException:
        raise
    except Exception as exc:
        result = build_fallback_result(
            question=request.question.strip(),
            model_answer=request.model_answer.strip(),
            context=request.context,
            backend_meta=client.metadata(),
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            error_message=str(exc),
        )

    result["selected_model"] = model
    return result


@app.post("/api/followup")
def followup(request: FollowupRequest):
    settings = _load_settings()
    model = _selected_model(request.model, settings.model)
    client = _build_client(model)

    messages = [
        {"role": "system", "content": FOLLOWUP_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Original question:\n{request.question.strip()}\n\n"
                + (
                    f"Model answer being audited:\n{request.model_answer.strip()}\n\n"
                    if request.model_answer.strip()
                    else ""
                )
                + (
                f"Context:\n{request.context[:2000]}\n\n"
                f"Follow-up question:\n{request.followup.strip()}"
                )
            ),
        },
    ]

    try:
        reply = client.chat(
            messages=messages,
            temperature=settings.temperature,
            max_tokens=min(560, settings.max_tokens),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"The follow-up request failed: {exc}") from exc

    return {"reply": reply, "selected_model": model}
