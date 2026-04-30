from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "history.db"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _loads(value: str | None, default):
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


class AuditStore:
    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)

    def _connect(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS analyses (
                    analysis_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    final_prediction TEXT NOT NULL,
                    final_confidence REAL NOT NULL,
                    memory_used INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    root_node_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS analysis_nodes (
                    node_id TEXT PRIMARY KEY,
                    analysis_id TEXT NOT NULL,
                    parent_id TEXT,
                    depth INTEGER NOT NULL,
                    branch_label TEXT NOT NULL,
                    row_count INTEGER NOT NULL,
                    statistical_summary_json TEXT NOT NULL,
                    trigger_reason TEXT NOT NULL,
                    llm_prompt TEXT NOT NULL,
                    llm_response TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    FOREIGN KEY (analysis_id) REFERENCES analyses(analysis_id)
                );

                CREATE TABLE IF NOT EXISTS memory_matches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    score REAL NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (analysis_id) REFERENCES analyses(analysis_id)
                );

                CREATE TABLE IF NOT EXISTS explain_cache (
                    cache_key TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    response_json TEXT NOT NULL
                );
                """
            )

    def save_analysis(
        self,
        *,
        analysis_id: str,
        request_payload: dict,
        final_prediction: str,
        final_confidence: float,
        root_node: dict,
        memory_used: bool,
        status: str,
        memory_matches: list[dict],
    ) -> None:
        self.initialize()
        created_at = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO analyses
                (analysis_id, created_at, request_json, final_prediction, final_confidence, memory_used, status, root_node_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    analysis_id,
                    created_at,
                    _dumps(request_payload),
                    final_prediction,
                    float(final_confidence),
                    int(memory_used),
                    status,
                    _dumps(root_node),
                ),
            )
            conn.execute("DELETE FROM analysis_nodes WHERE analysis_id = ?", (analysis_id,))
            for node in self._flatten_nodes(root_node):
                conn.execute(
                    """
                    INSERT OR REPLACE INTO analysis_nodes
                    (node_id, analysis_id, parent_id, depth, branch_label, row_count, statistical_summary_json,
                     trigger_reason, llm_prompt, llm_response, confidence_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        node["node_id"],
                        analysis_id,
                        node.get("parent_id"),
                        int(node["depth"]),
                        node["branch_label"],
                        int(node["row_count"]),
                        _dumps(node.get("statistical_summary", {})),
                        node.get("trigger_reason", ""),
                        node.get("llm_prompt", ""),
                        node.get("llm_response", ""),
                        float(node.get("confidence_score", 0)),
                    ),
                )
            conn.execute("DELETE FROM memory_matches WHERE analysis_id = ?", (analysis_id,))
            for match in memory_matches:
                conn.execute(
                    """
                    INSERT INTO memory_matches (analysis_id, source, score, payload_json, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        analysis_id,
                        str(match.get("source", "unknown")),
                        float(match.get("score", 0)),
                        _dumps(match),
                        created_at,
                    ),
                )

    def get_analysis(self, analysis_id: str) -> dict | None:
        self.initialize()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM analyses WHERE analysis_id = ?",
                (analysis_id,),
            ).fetchone()
            if row is None:
                return None
            matches = conn.execute(
                "SELECT payload_json FROM memory_matches WHERE analysis_id = ? ORDER BY id ASC",
                (analysis_id,),
            ).fetchall()
        return {
            "analysis_id": row["analysis_id"],
            "created_at": row["created_at"],
            "request": _loads(row["request_json"], {}),
            "final_prediction": row["final_prediction"],
            "final_confidence": row["final_confidence"],
            "root_node": _loads(row["root_node_json"], {}),
            "memory_used": bool(row["memory_used"]),
            "memory_matches": [_loads(match["payload_json"], {}) for match in matches],
            "status": row["status"],
        }

    def list_history(self, limit: int = 25) -> list[dict]:
        self.initialize()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT analysis_id, created_at, request_json, final_prediction, final_confidence, memory_used, status
                FROM analyses
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        return [
            {
                "analysis_id": row["analysis_id"],
                "created_at": row["created_at"],
                "request": _loads(row["request_json"], {}),
                "final_prediction": row["final_prediction"],
                "final_confidence": row["final_confidence"],
                "memory_used": bool(row["memory_used"]),
                "status": row["status"],
            }
            for row in rows
        ]

    def list_memory_documents(self, limit: int = 100) -> list[dict]:
        self.initialize()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT analysis_id, request_json, final_prediction, final_confidence
                FROM analyses
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        return [
            {
                "analysis_id": row["analysis_id"],
                "request": _loads(row["request_json"], {}),
                "final_prediction": row["final_prediction"],
                "final_confidence": row["final_confidence"],
            }
            for row in rows
        ]

    def health(self) -> dict:
        try:
            self.initialize()
            with self._connect() as conn:
                conn.execute("SELECT 1").fetchone()
            return {"ok": True, "path": str(self.db_path)}
        except Exception as exc:
            return {"ok": False, "path": str(self.db_path), "error": str(exc)}

    def get_explain_cache(self, cache_key: str, ttl_seconds: int) -> dict | None:
        self.initialize()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT created_at, response_json FROM explain_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
            if row is None:
                return None
            created_at = row["created_at"]
            if ttl_seconds > 0:
                try:
                    then = datetime.fromisoformat(created_at)
                    now = datetime.now(timezone.utc)
                    if (now - then).total_seconds() > ttl_seconds:
                        conn.execute("DELETE FROM explain_cache WHERE cache_key = ?", (cache_key,))
                        return None
                except Exception:
                    conn.execute("DELETE FROM explain_cache WHERE cache_key = ?", (cache_key,))
                    return None
            return _loads(row["response_json"], None)

    def set_explain_cache(self, cache_key: str, response_payload: dict) -> None:
        self.initialize()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO explain_cache (cache_key, created_at, response_json)
                VALUES (?, ?, ?)
                """,
                (cache_key, _utc_now(), _dumps(response_payload)),
            )

    def _flatten_nodes(self, node: dict) -> list[dict]:
        nodes = [node]
        for child in node.get("children", []):
            nodes.extend(self._flatten_nodes(child))
        return nodes
