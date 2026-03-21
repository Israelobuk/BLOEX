from typing import Any, Dict, List, Optional, Tuple
import html
import re

try:
    from rapidfuzz import fuzz
except ImportError:
    fuzz = None


SMART_DOUBLE_LEFT = chr(0x201C)
SMART_DOUBLE_RIGHT = chr(0x201D)
SMART_SINGLE_RIGHT = chr(0x2019)


def get_quote_position(context: str, quote: str) -> Tuple[Optional[int], Optional[int]]:
    if not quote:
        return None, None

    start = context.find(quote)
    if start != -1:
        return start, start + len(quote)

    start = context.lower().find(quote.lower())
    if start != -1:
        return start, start + len(quote)

    def norm(text: str) -> str:
        text = text.replace(SMART_DOUBLE_LEFT, '"').replace(SMART_DOUBLE_RIGHT, '"').replace(SMART_SINGLE_RIGHT, "'")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    norm_context = norm(context)
    norm_quote = norm(quote)
    start = norm_context.lower().find(norm_quote.lower())
    if start != -1:
        first_word = next((w for w in norm_quote.split(" ") if w), "")
        if first_word:
            raw_start = context.lower().find(first_word.lower())
            if raw_start != -1:
                return raw_start, min(len(context), raw_start + len(quote))

    if fuzz is not None:
        try:
            align = fuzz.partial_ratio_alignment(quote, context, score_cutoff=88)
            if align is not None:
                return align.dest_start, align.dest_end
        except Exception:
            pass

    return None, None


def verify_evidence_claims(result: Dict[str, Any], context: str) -> Dict[str, Any]:
    checked: List[Dict[str, Any]] = []

    for claim in result.get("evidence_claims", []):
        quote = str(claim.get("quote", "")).strip()
        start, end = get_quote_position(context, quote)

        if start is None or end is None:
            checked.append(
                {
                    "claim": claim.get("claim", ""),
                    "support_reason": claim.get("support_reason", ""),
                    "quote": quote,
                    "start": None,
                    "end": None,
                    "verified": False,
                }
            )
        else:
            checked.append(
                {
                    "claim": claim.get("claim", ""),
                    "support_reason": claim.get("support_reason", ""),
                    "quote": context[start:end],
                    "start": start,
                    "end": end,
                    "verified": True,
                }
            )

    result["evidence_claims"] = checked
    return result


def _keyword_tokens(text: str) -> set:
    stop = {
        "the", "a", "an", "is", "are", "was", "were", "be", "to", "of", "in", "on", "for",
        "and", "or", "it", "this", "that", "with", "as", "at", "by", "from", "why", "what",
        "how", "when", "where", "who", "which", "does", "do", "did", "can", "could", "would",
        "should", "will", "you", "your", "i", "we", "they", "he", "she", "them", "his", "her",
    }
    words = []
    for raw in text.lower().split():
        clean = "".join(ch for ch in raw if ch.isalnum())
        if len(clean) >= 3 and clean not in stop:
            words.append(clean)
    return set(words)


def add_question_relevance(result: Dict[str, Any], question: str) -> Dict[str, Any]:
    q_tokens = _keyword_tokens(question)
    for claim in result.get("evidence_claims", []):
        claim_text = str(claim.get("claim", ""))
        quote_text = str(claim.get("quote", ""))
        c_tokens = _keyword_tokens(claim_text + " " + quote_text)
        overlap = len(q_tokens.intersection(c_tokens))
        claim["question_relevance"] = "relevant" if overlap > 0 else "weak"
    return result


def adjust_confidence(result: Dict[str, Any]) -> Dict[str, Any]:
    claims = result.get("evidence_claims", [])
    if not claims:
        result["confidence"] = "low"
        result["confidence_type"] = "unsupported"
        result["confidence_basis"] = "No usable evidence claim was extracted from the context."
        result["confidence_reason"] = "This low confidence is based on the lack of verifiable evidence in the provided context."
        return result

    verified = sum(1 for c in claims if c.get("verified"))
    total = len(claims)
    weak_relevance = sum(1 for c in claims if c.get("question_relevance") == "weak")

    if verified == 0:
        result["confidence"] = "low"
    elif verified < total and result.get("confidence") == "high":
        result["confidence"] = "medium"

    if weak_relevance == len(claims) and len(claims) > 0:
        result["confidence"] = "low"

    confidence = str(result.get("confidence", "")).strip().lower() or "low"

    if verified == total and weak_relevance == 0:
        confidence_type = "well-supported"
        basis = f"Based on {verified}/{total} verified evidence claims with direct relevance to the question."
    elif verified > 0:
        confidence_type = "partially-supported"
        basis = (
            f"Based on {verified}/{total} verified evidence claims"
            f" and {weak_relevance} weakly relevant claim{'s' if weak_relevance != 1 else ''}."
        )
    else:
        confidence_type = "weakly-supported"
        basis = "Based on unverified or weakly relevant evidence rather than direct, validated support."

    existing_reason = str(result.get("confidence_reason", "")).strip()
    mentioned = re.search(r"\b(low|medium|high)\b", existing_reason.lower())
    if not existing_reason or (mentioned and mentioned.group(1) != confidence):
        result["confidence_reason"] = (
            f"This {confidence} confidence is {basis[0].lower() + basis[1:]}"
        )

    result["confidence_type"] = confidence_type
    result["confidence_basis"] = basis

    return result


def build_highlighted_context(context: str, evidence_claims: List[Dict[str, Any]]) -> str:
    spans: List[Tuple[int, int]] = []
    for claim in evidence_claims:
        start = claim.get("start")
        end = claim.get("end")
        if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(context):
            spans.append((start, end))

    if not spans:
        return html.escape(context)

    spans.sort()
    merged: List[List[int]] = []
    for start, end in spans:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    parts: List[str] = []
    cursor = 0
    for start, end in merged:
        parts.append(html.escape(context[cursor:start]))
        parts.append("<mark>" + html.escape(context[start:end]) + "</mark>")
        cursor = end

    parts.append(html.escape(context[cursor:]))
    return "".join(parts)
