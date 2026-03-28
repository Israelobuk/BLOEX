from datetime import datetime, timezone
from typing import Any, Dict, List


def build_trace_log(
    backend_meta: Dict[str, Any],
    temperature: float,
    max_tokens: int,
    steps: List[str],
    raw_preview: str = "",
) -> Dict[str, Any]:
    # Keep a small in-memory trace to explain how the answer was generated.
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": backend_meta.get("model"),
        "backend": backend_meta.get("client"),
        "base_url": backend_meta.get("base_url"),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "steps_run": steps,
        "raw_output_preview": raw_preview,
    }
