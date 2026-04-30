from __future__ import annotations

from typing import Any

from .logic_engine import iter_nodes


def _strongest_nodes(root_node: dict[str, Any], limit: int = 3) -> list[dict[str, Any]]:
    nodes = [node for node in iter_nodes(root_node) if node.get("trigger_reason")]
    return sorted(nodes, key=lambda node: float(node.get("confidence_score", 0)), reverse=True)[:limit]


def _signal_note(signal_result: dict[str, Any]) -> str:
    root = signal_result.get("root_node", {})
    strongest = _strongest_nodes(root, limit=2)
    lines = []
    for node in strongest:
        response = str(node.get("llm_response") or "").strip()
        reason = str(node.get("trigger_reason") or "").strip()
        branch = str(node.get("branch_label") or "root")
        if response:
            lines.append(response)
        elif reason:
            lines.append(f"The recursive signal read flagged `{branch}` because of {reason}.")
    if not lines:
        lines.append(str(signal_result.get("final_prediction") or "No strong structured signal was found."))
    return " ".join(lines)


def _build_signal_claim(signal_result: dict[str, Any]) -> dict[str, Any]:
    root = signal_result.get("root_node", {})
    summary = root.get("statistical_summary", {})
    means = summary.get("numeric_means", {})
    correlations = summary.get("correlations", {})
    pieces = []
    if means:
        pieces.append("means " + ", ".join(f"{key}={value}" for key, value in list(means.items())[:3]))
    if correlations:
        pieces.append("correlations " + ", ".join(f"{key}={value}" for key, value in list(correlations.items())[:2]))
    quote = "; ".join(pieces) or str(signal_result.get("final_prediction", "Structured signal analysis completed."))
    return {
        "claim": "Recursive signal read from structured context",
        "support_reason": "BLOEX found table-like or repeated numeric evidence and recursively checked which signals line up with the model answer.",
        "quote": quote,
        "start": None,
        "end": None,
        "verified": False,
        "signal_based": True,
    }


def enrich_explainer_result_with_signals(result: dict[str, Any], signal_result: dict[str, Any]) -> dict[str, Any]:
    if result.get("signal_analysis"):
        return result

    enriched = dict(result)
    signal_note = _signal_note(signal_result)
    prior_black_box = str(enriched.get("black_box_explanation") or "").strip()
    enriched["black_box_explanation"] = (
        f"{prior_black_box}\n\nRecursive signal read: {signal_note}".strip()
    )

    claims = list(enriched.get("evidence_claims") or [])
    claims.append(_build_signal_claim(signal_result))
    enriched["evidence_claims"] = claims[:4]

    uncertainty = list(enriched.get("uncertainty") or [])
    final_confidence = signal_result.get("final_confidence", 0)
    uncertainty.append(
        "The recursive signal read is not proof; it is a structured check of the evidence patterns behind the model answer."
    )
    enriched["uncertainty"] = uncertainty[:4]

    confidence_reason = str(enriched.get("confidence_reason") or "").strip()
    signal_confidence = f"The structured signals produced a recursive confidence score of {final_confidence}."
    enriched["confidence_reason"] = (
        f"{confidence_reason} {signal_confidence} This combines text grounding with structured signals where available.".strip()
    )
    enriched["signal_analysis"] = {
        "analysis_id": signal_result.get("analysis_id"),
        "final_prediction": signal_result.get("final_prediction"),
        "final_confidence": final_confidence,
        "root_node": signal_result.get("root_node"),
        "memory_used": signal_result.get("memory_used", False),
        "status": signal_result.get("status", "completed"),
    }
    return enriched
