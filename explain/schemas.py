from typing import Any, Dict, List
import re


ALLOWED_CONFIDENCE = {"low", "medium", "high"}


def default_result() -> Dict[str, Any]:
    return {
        "answer": "",
        "black_box_explanation": "",
        "assumptions": [],
        "evidence_claims": [],
        "answer_support": [],
        "overreach_flags": [],
        "overreach_summary": "",
        "confidence_breakdown": {},
        "uncertainty": [],
        "confidence": "",
        "confidence_reason": "",
        "followups": [],
    }


def to_string_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip() and str(x).strip().upper() != 'NONE']
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.upper() == 'NONE':
        return []
    return [text]


def normalize_followups(value: Any) -> List[str]:
    items = to_string_list(value)
    normalized: List[str] = []

    for item in items:
        clean = " ".join(item.split())
        match = re.search(r"([^?]*\?)", clean)
        question = match.group(1).strip() if match else ""

        if not question:
            continue
        if len(question) > 180:
            continue

        normalized.append(question)

    return normalized


def normalize_result(raw: Dict[str, Any]) -> Dict[str, Any]:
    out = default_result()

    out["answer"] = str(raw.get("answer", "")).strip()
    out["black_box_explanation"] = str(raw.get("black_box_explanation", "")).strip()
    out["assumptions"] = to_string_list(raw.get("assumptions"))[:3]
    out["uncertainty"] = to_string_list(raw.get("uncertainty"))[:3]
    out["followups"] = normalize_followups(raw.get("followups"))[:3]

    confidence = str(raw.get("confidence", "")).strip().lower()
    out["confidence"] = confidence if confidence in ALLOWED_CONFIDENCE else ""
    out["confidence_reason"] = str(raw.get("confidence_reason", "")).strip()

    claims = raw.get("evidence_claims", [])
    clean_claims: List[Dict[str, Any]] = []
    if isinstance(claims, list):
        for item in claims[:3]:
            if not isinstance(item, dict):
                continue
            claim = str(item.get("claim", "")).strip()
            support_reason = str(item.get("support_reason", "")).strip()
            quote = str(item.get("quote", "")).strip()
            start = item.get("start") if isinstance(item.get("start"), int) else None
            end = item.get("end") if isinstance(item.get("end"), int) else None

            if claim or quote or support_reason:
                clean_claims.append(
                    {
                        "claim": claim,
                        "support_reason": support_reason,
                        "quote": quote,
                        "start": start,
                        "end": end,
                        "verified": False,
                    }
                )

    out["evidence_claims"] = clean_claims
    return out
