from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, HTTPException

from config import load_from_env
from core.ollama_client import AsyncOllamaClient
from core.persistence import AuditStore, DEFAULT_DB_PATH
from core.workflow import run_analysis_workflow

from .schemas import AnalyzeRequest, AnalyzeResponse, HistoryResponse


router = APIRouter()


def _store() -> AuditStore:
    return AuditStore(Path(os.getenv("BBE_HISTORY_DB", str(DEFAULT_DB_PATH))))


def _qdrant_url() -> str:
    return os.getenv("BBE_QDRANT_URL", "").strip()


def _ollama_client() -> AsyncOllamaClient:
    settings = load_from_env()
    return AsyncOllamaClient(
        base_url=settings.base_url,
        model=settings.model,
        api_key=settings.api_key,
        timeout_seconds=settings.timeout_seconds,
        embedding_model=os.getenv("BBE_EMBEDDING_MODEL", settings.model).strip(),
    )


@router.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    analysis_id = str(uuid4())
    store = _store()
    client = _ollama_client()
    try:
        result = await run_analysis_workflow(
            analysis_id=analysis_id,
            request_payload=request.model_dump(),
            store=store,
            ollama_client=client,
            qdrant_url=_qdrant_url(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "analysis_id": analysis_id,
        "final_prediction": result["final_prediction"],
        "final_confidence": result["final_confidence"],
        "root_node": result["root_node"],
        "memory_used": result.get("memory_used", False),
        "status": "completed",
    }


@router.get("/api/analysis/{analysis_id}", response_model=AnalyzeResponse)
def get_analysis(analysis_id: str):
    saved = _store().get_analysis(analysis_id)
    if saved is None:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    return {
        "analysis_id": saved["analysis_id"],
        "final_prediction": saved["final_prediction"],
        "final_confidence": saved["final_confidence"],
        "root_node": saved["root_node"],
        "memory_used": saved["memory_used"],
        "status": saved["status"],
    }


@router.get("/api/history", response_model=HistoryResponse)
def history(limit: int = 25):
    return {"items": _store().list_history(limit=max(1, min(limit, 100)))}


async def predictive_health() -> dict:
    store_health = _store().health()
    client = _ollama_client()
    ollama_health = await client.health()
    qdrant = {"ok": False, "configured": bool(_qdrant_url())}
    if _qdrant_url():
        try:
            from qdrant_client import QdrantClient

            QdrantClient(url=_qdrant_url(), timeout=3).get_collections()
            qdrant = {"ok": True, "configured": True}
        except Exception as exc:
            qdrant = {"ok": False, "configured": True, "error": str(exc)}
    return {
        "sqlite": store_health,
        "ollama": ollama_health,
        "qdrant": qdrant,
    }
