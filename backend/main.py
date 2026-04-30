from __future__ import annotations

from dataclasses import replace
from copy import deepcopy
import asyncio
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path
from uuid import uuid4
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import load_from_env
from api.routes import router as predictive_router, predictive_health
from core.context_parser import parse_structured_context
from core.logic_engine import build_analysis_tree
from core.ollama_client import AsyncOllamaClient
from core.persistence import AuditStore, DEFAULT_DB_PATH
from core.signal_enrichment import enrich_explainer_result_with_signals
from core.workflow import run_analysis_workflow
from explain.pipeline import ExplainerPipeline, build_fallback_result
from llm import create_client


load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(Path(__file__).resolve().parent / ".env")
MODEL_OPTIONS = [
    {"value": "tinyllama:latest", "label": "TinyLlama", "description": "Smallest deployment-safe option. Best for getting the hosted explainer to run reliably."},
    {"value": "phi3:mini", "label": "Phi-3 Mini", "description": "Faster and lighter than Llama 3.2, with better quality than TinyLlama when memory allows."},
    {"value": "llama3.2:latest", "label": "Llama 3.2", "description": "Stronger writing quality, but too heavy for many free hosted instances."},
    {"value": "llama3.1:8b", "label": "Llama 3.1 8B", "description": "More capable and detailed. Better when you want stronger reasoning and fuller writeups."},
    {"value": "gpt-oss:120b", "label": "GPT-OSS 120B", "description": "Hosted-scale reasoning model. Best when you want a stronger cloud backend with more depth."},
]

CACHE_DIR = PROJECT_ROOT / ".cache"
CACHE_FILE = CACHE_DIR / "explain_cache.json"
EXPLAIN_CACHE_MAX_ITEMS = int(os.getenv("BBE_EXPLAIN_CACHE_MAX_ITEMS", "400"))
EXPLAIN_CACHE_TTL_SECONDS = int(os.getenv("BBE_EXPLAIN_CACHE_TTL_SECONDS", str(60 * 60 * 24 * 14)))
CACHE_SCHEMA_VERSION = "v4"


class ExplainCache:
    def __init__(self, path: Path, max_items: int, ttl_seconds: int):
        self.path = path
        self.max_items = max(1, max_items)
        self.ttl_seconds = max(0, ttl_seconds)
        self.lock = threading.Lock()
        self.items: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        try:
            if not self.path.exists():
                return
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return
            raw_items = data.get("items", {})
            if not isinstance(raw_items, dict):
                return
            self.items = raw_items
            self._prune_locked()
        except Exception:
            self.items = {}

    def _save_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"items": self.items}
        self.path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")

    def _is_fresh(self, created_at: float) -> bool:
        if self.ttl_seconds <= 0:
            return True
        return (time.time() - created_at) <= self.ttl_seconds

    def _prune_locked(self) -> None:
        now = time.time()
        # Drop expired first.
        expired = [k for k, v in self.items.items() if not self._is_fresh(float(v.get("created_at", now)))]
        for key in expired:
            self.items.pop(key, None)
        # Then cap size by oldest first.
        if len(self.items) > self.max_items:
            ordered = sorted(self.items.items(), key=lambda kv: float(kv[1].get("created_at", 0)))
            for key, _ in ordered[: len(self.items) - self.max_items]:
                self.items.pop(key, None)

    def get(self, key: str) -> dict | None:
        with self.lock:
            record = self.items.get(key)
            if not record:
                return None
            created_at = float(record.get("created_at", 0))
            if not self._is_fresh(created_at):
                self.items.pop(key, None)
                self._save_locked()
                return None
            # Touch for freshness ordering without changing semantics.
            record["last_hit_at"] = time.time()
            self.items[key] = record
            self._save_locked()
            value = record.get("value")
            return deepcopy(value) if isinstance(value, dict) else None

    def set(self, key: str, value: dict) -> None:
        with self.lock:
            now = time.time()
            self.items[key] = {"created_at": now, "last_hit_at": now, "value": deepcopy(value)}
            self._prune_locked()
            self._save_locked()


def _cache_key_for_explain(
    *,
    model: str,
    question: str,
    model_answer: str,
    context: str,
    temperature: float,
    max_tokens: int,
    critique_pass: bool,
) -> str:
    payload = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "model": model,
        "question": question.strip(),
        "model_answer": model_answer.strip(),
        "context": context.strip(),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "critique_pass": critique_pass,
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


EXPLAIN_CACHE = ExplainCache(
    path=CACHE_FILE,
    max_items=EXPLAIN_CACHE_MAX_ITEMS,
    ttl_seconds=EXPLAIN_CACHE_TTL_SECONDS,
)


def _pandas_signal_hint(context: str) -> str:
    structured = parse_structured_context(context)
    if not structured:
        return ""
    try:
        root = build_analysis_tree(
            structured["data"],
            user_goal="Summarize structured evidence signals for answer auditing.",
            focus_columns=structured.get("focus_columns") or [],
            max_depth=1,
            min_rows=2,
        )
    except Exception:
        return ""

    summary = root.get("statistical_summary", {})
    parts = [f"rows={root.get('row_count', 0)}"]
    means = summary.get("numeric_means", {})
    missing = summary.get("missing_percentage", {})
    outliers = summary.get("outlier_signals", {})
    correlations = summary.get("correlations", {})
    if means:
        parts.append("means=" + ", ".join(f"{k}:{v}" for k, v in list(means.items())[:4]))
    if missing:
        parts.append("missing=" + ", ".join(f"{k}:{v}%" for k, v in list(missing.items())[:4]))
    if outliers:
        parts.append("outliers=" + ", ".join(list(outliers.keys())[:4]))
    if correlations:
        parts.append("correlations=" + ", ".join(f"{k}:{v}" for k, v in list(correlations.items())[:4]))
    trigger = str(root.get("trigger_reason") or "").strip()
    if trigger:
        parts.append(f"trigger={trigger}")
    return " | ".join(parts)


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value.strip())
    except Exception:
        return default


def _should_retry_stronger_model() -> bool:
    return _parse_bool(os.getenv("BBE_STRONGER_MODEL_RETRY"), True)


def _should_enable_signal_enrichment() -> bool:
    return _parse_bool(os.getenv("BBE_SIGNAL_ENRICHMENT"), False)


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


def _load_settings():
    settings = load_from_env()
    base_url = settings.base_url.strip() or "http://127.0.0.1:11434"
    model = settings.model.strip() or "tinyllama:latest"
    return replace(settings, base_url=base_url, model=model)


def _selected_model(requested_model: str | None, default_model: str) -> str:
    allowed_models = {option["value"] for option in MODEL_OPTIONS}
    candidate = (requested_model or default_model or "").strip()
    if not candidate:
        raise HTTPException(status_code=500, detail="Model is not configured on the backend.")
    if candidate not in allowed_models:
        raise HTTPException(status_code=400, detail=f"Unsupported model '{candidate}'.")
    return candidate


def _installed_models(settings) -> list[str]:
    try:
        response = requests.get(
            f"{settings.base_url}/api/tags",
            timeout=min(max(settings.timeout_seconds, 5), 30),
        )
        response.raise_for_status()
        data = response.json()
        return [item.get("name", "").strip() for item in data.get("models", []) if isinstance(item, dict) and item.get("name")]
    except Exception:
        return []


def _try_pull_model(settings, model: str) -> bool:
    try:
        response = requests.post(
            f"{settings.base_url}/api/pull",
            json={"model": model, "stream": False},
            timeout=max(settings.timeout_seconds, 120),
        )
        return response.ok
    except Exception:
        return False


def _resolve_runtime_model(settings, requested_model: str | None = None) -> tuple[str, list[str]]:
    preferred = _selected_model(requested_model, settings.model)
    installed = _installed_models(settings)
    if installed:
        if preferred in installed:
            return preferred, installed
        allowed = {option["value"] for option in MODEL_OPTIONS}
        for name in installed:
            if name in allowed:
                return name, installed
        return installed[0], installed

    # Nothing local installed yet: try pulling default tiny model once.
    if _try_pull_model(settings, "tinyllama:latest"):
        installed = _installed_models(settings)
        if "tinyllama:latest" in installed:
            return "tinyllama:latest", installed
    return preferred, installed


def _next_stronger_model(current: str, installed: list[str]) -> str | None:
    # Prefer a stronger local model when tiny outputs are low quality.
    preference = ["phi3:mini", "llama3.2:latest", "llama3.1:8b", "gpt-oss:120b"]
    for candidate in preference:
        if candidate != current and candidate in installed:
            return candidate
    return None


def _result_is_weak(result: dict) -> bool:
    answer = str(result.get("answer", "") or "").strip().lower()
    black_box = str(result.get("black_box_explanation", "") or "").strip().lower()
    confidence_reason = str(result.get("confidence_reason", "") or "").strip().lower()
    evidence = result.get("evidence_claims") or []
    assumptions = [str(item).strip().lower() for item in (result.get("assumptions") or [])]
    uncertainty = result.get("uncertainty") or []
    uncertainty_lines = [str(item).strip().lower() for item in uncertainty]
    fallback_mode = bool(result.get("fallback_mode"))
    weak_phrases = [
        "directionally useful, but it still needs review",
        "full reasoning trace was incomplete",
        "fallback review",
        "a plain-english audit verdict",
        "a direct answer to the question",
        "using as much detail as needed",
        "using as much detail as the context supports",
        "a plain-english explanation of that confidence",
    ]
    return (
        fallback_mode
        or any(p in answer for p in weak_phrases)
        or any(p in black_box for p in weak_phrases)
        or any(p in confidence_reason for p in weak_phrases)
        or any("one meaningful hidden assumption" in item for item in assumptions)
        or any("meaningful hidden assumption or interpretation step" in item for item in assumptions)
        or any("a meaningful caveat about where the answer may be too strong" in item for item in uncertainty_lines)
        or any("one caveat saying where the answer may be too strong" in item for item in uncertainty_lines)
        or len(evidence) == 0
        or len(uncertainty) == 0
    )


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


def _maybe_enrich_with_recursive_signals(result: dict, request: ExplainRequest) -> dict:
    structured = parse_structured_context(request.context)
    if not structured:
        return result

    settings = _load_settings()
    analysis_id = str(uuid4())
    store = AuditStore(Path(os.getenv("BBE_HISTORY_DB", str(DEFAULT_DB_PATH))))
    client = AsyncOllamaClient(
        base_url=settings.base_url,
        model=settings.model,
        api_key=settings.api_key,
        timeout_seconds=settings.timeout_seconds,
        embedding_model=os.getenv("BBE_EMBEDDING_MODEL", settings.model).strip(),
    )
    payload = {
        "user_goal": (
            "Explain which structured evidence signals the model answer appears to rely on, "
            "miss, or overstate."
        ),
        "data": structured["data"],
        "focus_columns": structured["focus_columns"],
        "max_depth": 2,
        "min_rows": 2,
        "use_memory": True,
        "use_llm": True,
    }
    try:
        signal_result = asyncio.run(
            run_analysis_workflow(
                analysis_id=analysis_id,
                request_payload=payload,
                store=store,
                ollama_client=client,
                qdrant_url=os.getenv("BBE_QDRANT_URL", "").strip(),
            )
        )
    except Exception as exc:
        enriched = dict(result)
        uncertainty = list(enriched.get("uncertainty") or [])
        uncertainty.append(f"Recursive signal analysis was skipped because structured evidence failed to process: {exc}")
        enriched["uncertainty"] = uncertainty[:4]
        return enriched

    signal_result["analysis_id"] = analysis_id
    return enrich_explainer_result_with_signals(result, signal_result)

app = FastAPI(title="Black Box Explainer API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(predictive_router)


@app.get("/api/health")
async def health():
    settings = _load_settings()
    model, installed = _resolve_runtime_model(settings)
    client = _build_client(model)
    ready, status = client.healthcheck()
    return {
        "ok": ready,
        "status": status,
        "serverUrlLocked": True,
        "serverLabel": settings.base_url,
        "selectedModel": model,
        "timeoutSeconds": settings.timeout_seconds,
        "maxTokens": settings.max_tokens,
        "models": MODEL_OPTIONS,
        "installedModels": installed,
        "critiquePass": settings.critique_pass,
        "predictiveEngine": await predictive_health(),
    }


@app.get("/api/config")
def config():
    settings = _load_settings()
    model, installed = _resolve_runtime_model(settings)
    return {
        "serverUrlLocked": True,
        "serverLabel": settings.base_url,
        "selectedModel": model,
        "models": MODEL_OPTIONS,
        "installedModels": installed,
    }


@app.post("/api/explain")
def explain(request: ExplainRequest):
    started_at = time.time()
    settings = _load_settings()
    model, installed = _resolve_runtime_model(settings, request.model)
    client = _build_client(model)
    pipeline = ExplainerPipeline(client)
    question = request.question.strip()
    model_answer = request.model_answer.strip()
    context = request.context
    has_structured_context = parse_structured_context(context) is not None
    signal_hint = _pandas_signal_hint(context)
    store = AuditStore(Path(os.getenv("BBE_HISTORY_DB", str(DEFAULT_DB_PATH))))
    cache_key = _cache_key_for_explain(
        model=model,
        question=question,
        model_answer=model_answer,
        context=context,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        critique_pass=settings.critique_pass,
    )
    cached = EXPLAIN_CACHE.get(cache_key)
    if cached is not None and not cached.get("fallback_mode") and not _result_is_weak(cached):
        cached["selected_model"] = model
        cached["cached"] = True
        if _should_enable_signal_enrichment():
            return _maybe_enrich_with_recursive_signals(cached, request)
        return cached
    sqlite_cached = store.get_explain_cache(cache_key, EXPLAIN_CACHE_TTL_SECONDS)
    if isinstance(sqlite_cached, dict) and not sqlite_cached.get("fallback_mode") and not _result_is_weak(sqlite_cached):
        sqlite_cached["selected_model"] = model
        sqlite_cached["cached"] = True
        EXPLAIN_CACHE.set(cache_key, sqlite_cached)
        if _should_enable_signal_enrichment():
            return _maybe_enrich_with_recursive_signals(sqlite_cached, request)
        return sqlite_cached

    try:
        use_agentic_explainer = _parse_bool(os.getenv("BBE_AGENTIC_EXPLAINER"), True)
        if use_agentic_explainer:
            result = pipeline.run_agentic(
                question=question,
                model_answer=model_answer,
                context=context,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
                critique_pass=settings.critique_pass,
                completion_retry=settings.completion_retry,
                max_agent_loops=max(1, min(_parse_int(os.getenv("BBE_AGENTIC_LOOPS"), 2), 4)),
                signal_hint=signal_hint,
            )
        else:
            result = pipeline.run(
                question=question,
                model_answer=model_answer,
                context=context,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
                critique_pass=settings.critique_pass,
                completion_retry=settings.completion_retry,
                signal_hint=signal_hint,
            )
    except HTTPException:
        raise
    except Exception as exc:
        result = build_fallback_result(
            question=question,
            model_answer=model_answer,
            context=context,
            backend_meta=client.metadata(),
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            error_message=str(exc),
        )

    # Stronger-model retry is optional because it can double end-to-end latency.
    if _should_retry_stronger_model() and _result_is_weak(result):
        stronger = _next_stronger_model(model, installed)
        if stronger:
            stronger_client = _build_client(stronger)
            stronger_pipeline = ExplainerPipeline(stronger_client)
            try:
                if _parse_bool(os.getenv("BBE_AGENTIC_EXPLAINER"), True):
                    improved = stronger_pipeline.run_agentic(
                        question=question,
                        model_answer=model_answer,
                        context=context,
                        temperature=settings.temperature,
                        max_tokens=settings.max_tokens,
                        critique_pass=settings.critique_pass,
                        completion_retry=settings.completion_retry,
                        max_agent_loops=max(1, min(_parse_int(os.getenv("BBE_AGENTIC_LOOPS"), 2), 4)),
                        signal_hint=signal_hint,
                    )
                else:
                    improved = stronger_pipeline.run(
                        question=question,
                        model_answer=model_answer,
                        context=context,
                        temperature=settings.temperature,
                        max_tokens=settings.max_tokens,
                        critique_pass=settings.critique_pass,
                        completion_retry=settings.completion_retry,
                        signal_hint=signal_hint,
                    )
                if not _result_is_weak(improved):
                    result = improved
                    model = stronger
            except Exception:
                pass

    result["selected_model"] = model
    result["cached"] = False
    result["response_ms"] = int((time.time() - started_at) * 1000)
    if not result.get("fallback_mode") and not _result_is_weak(result):
        EXPLAIN_CACHE.set(cache_key, result)
        store.set_explain_cache(cache_key, result)
    if _should_enable_signal_enrichment():
        result = _maybe_enrich_with_recursive_signals(result, request)
    return result

