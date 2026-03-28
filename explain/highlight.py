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

DIRECT_SUPPORT_THRESHOLD = 0.78
PARTIAL_SUPPORT_THRESHOLD = 0.48
WEAK_SUPPORT_THRESHOLD = 0.28


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
        support_probe = " ".join(
            part
            for part in [
                quote,
                str(claim.get("claim", "")).strip(),
                str(claim.get("support_reason", "")).strip(),
            ]
            if part
        ).strip()
        matched_context, support_score = _best_context_support(support_probe, context)
        support_label = _support_label(support_score)

        if start is None or end is None:
            checked.append(
                {
                    "claim": claim.get("claim", ""),
                    "support_reason": claim.get("support_reason", ""),
                    "quote": quote,
                    "start": None,
                    "end": None,
                    "verified": False,
                    "matched_context": matched_context,
                    "context_support_score": support_score,
                    "context_support_label": support_label,
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
                    "matched_context": context[start:end],
                    "context_support_score": 1.0,
                    "context_support_label": "direct",
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


def _normalize_match_text(text: str) -> str:
    text = text.replace(SMART_DOUBLE_LEFT, '"').replace(SMART_DOUBLE_RIGHT, '"').replace(SMART_SINGLE_RIGHT, "'")
    return re.sub(r"\s+", " ", text).strip()


def _split_context_units(text: str) -> List[str]:
    units: List[str] = []
    for block in re.split(r"[\r\n]+", text):
        block = block.strip()
        if not block:
            continue
        for part in re.split(r"(?<=[.!?])\s+", block):
            clean = part.strip()
            if len(clean) >= 20:
                units.append(clean)
    if not units and text.strip():
        units.append(text.strip())
    return units


def _fuzzy_score(left: str, right: str) -> float:
    if not left or not right or fuzz is None:
        return 0.0
    try:
        return max(
            fuzz.partial_ratio(left, right) / 100.0,
            fuzz.token_set_ratio(left, right) / 100.0,
        )
    except Exception:
        return 0.0


def _token_overlap_score(left: str, right: str) -> float:
    left_tokens = _keyword_tokens(left)
    right_tokens = _keyword_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens.intersection(right_tokens))
    return overlap / max(1, min(len(left_tokens), len(right_tokens)))


def _best_context_support(text: str, context: str) -> Tuple[str, float]:
    probe = _normalize_match_text(text)
    if not probe:
        return "", 0.0

    best_unit = ""
    best_score = 0.0

    for unit in _split_context_units(context):
        normalized_unit = _normalize_match_text(unit)
        if not normalized_unit:
            continue
        if probe.lower() in normalized_unit.lower() or normalized_unit.lower() in probe.lower():
            return unit, 1.0

        token_score = _token_overlap_score(probe, normalized_unit)
        fuzzy_score = _fuzzy_score(probe, normalized_unit)
        score = max(token_score, fuzzy_score * 0.92, (token_score * 0.55) + (fuzzy_score * 0.45))

        if score > best_score:
            best_score = score
            best_unit = unit

    return best_unit, round(best_score, 3)


def _support_label(score: float) -> str:
    if score >= DIRECT_SUPPORT_THRESHOLD:
        return "direct"
    if score >= PARTIAL_SUPPORT_THRESHOLD:
        return "partial"
    if score >= WEAK_SUPPORT_THRESHOLD:
        return "weak"
    return "none"


def add_question_relevance(result: Dict[str, Any], question: str) -> Dict[str, Any]:
    q_tokens = _keyword_tokens(question)
    for claim in result.get("evidence_claims", []):
        probe = " ".join(
            [
                str(claim.get("claim", "")),
                str(claim.get("quote", "")),
                str(claim.get("support_reason", "")),
                str(claim.get("matched_context", "")),
            ]
        ).strip()
        c_tokens = _keyword_tokens(probe)
        overlap = len(q_tokens.intersection(c_tokens)) / max(1, min(len(q_tokens), len(c_tokens) or 1)) if q_tokens else 0.0
        fuzzy_overlap = _fuzzy_score(question, probe)
        relevance_score = max(overlap, fuzzy_overlap * 0.85)
        claim["question_relevance_score"] = round(relevance_score, 2)
        claim["question_relevance"] = "relevant" if relevance_score >= 0.22 else "weak"
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
    direct_grounded = sum(1 for c in claims if c.get("verified") or c.get("context_support_label") == "direct")
    partial_grounded = sum(1 for c in claims if c.get("verified") or c.get("context_support_label") in {"direct", "partial"})
    relevant = sum(1 for c in claims if c.get("question_relevance") == "relevant")
    weak_relevance = sum(1 for c in claims if c.get("question_relevance") == "weak")
    overreach_count = len(result.get("overreach_flags", []))

    grounding_ratio = partial_grounded / max(1, total)
    direct_ratio = direct_grounded / max(1, total)
    relevance_ratio = relevant / max(1, total)
    overreach_penalty = min(0.35, overreach_count * 0.15)
    score = (grounding_ratio * 0.45) + (direct_ratio * 0.35) + (relevance_ratio * 0.2) - overreach_penalty

    if partial_grounded == 0 or relevant == 0:
        confidence = "low"
    elif score >= 0.8:
        confidence = "high"
    elif score >= 0.45:
        confidence = "medium"
    else:
        confidence = "low"

    result["confidence"] = confidence

    if partial_grounded == total and relevant == total and overreach_count == 0:
        confidence_type = "well-supported"
        basis = f"Based on {direct_grounded}/{total} directly grounded evidence claims that line up with the question."
    elif partial_grounded > 0:
        confidence_type = "partially-supported"
        basis = (
            f"Based on {partial_grounded}/{total} evidence claims that found meaningful context support"
            f" and {relevant}/{total} that clearly connect back to the question."
        )
    else:
        confidence_type = "weakly-supported"
        basis = "Based on weak context matching or evidence that does not connect cleanly back to the question."

    existing_reason = str(result.get("confidence_reason", "")).strip()
    mentioned = re.search(r"\b(low|medium|high)\b", existing_reason.lower())
    if not existing_reason or (mentioned and mentioned.group(1) != confidence):
        result["confidence_reason"] = (
            f"This {confidence} confidence is {basis[0].lower() + basis[1:]}"
        )

    result["confidence_type"] = confidence_type
    result["confidence_basis"] = basis

    return result


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part and part.strip()]


def detect_answer_overreach(result: Dict[str, Any], question: str, context: str) -> Dict[str, Any]:
    answer_sentences = _split_sentences(str(result.get("answer", "")))
    claims = result.get("evidence_claims", [])
    supported_claims = [
        claim for claim in claims
        if claim.get("question_relevance") == "relevant" and claim.get("context_support_label") in {"direct", "partial"}
    ]

    question_tokens = _keyword_tokens(question)

    strong_language = re.compile(
        r"\b(always|never|guarantee(?:s|d)?|prove(?:s|d)?|clearly|definitely|certainly|must|only|all|none|cannot fail|without doubt)\b",
        re.IGNORECASE,
    )

    answer_support: List[Dict[str, Any]] = []
    overreach_flags: List[Dict[str, Any]] = []

    for sentence in answer_sentences:
        sentence_tokens = _keyword_tokens(sentence)
        token_count = max(len(sentence_tokens), 1)
        question_overlap = len(sentence_tokens.intersection(question_tokens)) / token_count if question_tokens else 0.0
        has_strong_language = bool(strong_language.search(sentence))
        matched_context, support_score = _best_context_support(sentence, context)
        support_label = _support_label(support_score)
        matched_context_has_strong_language = bool(strong_language.search(matched_context))

        if support_label == "direct":
            label = "supported"
            reason = "This part of the answer closely matches supporting language in the provided context."
        elif support_label == "partial":
            label = "inferred"
            reason = "This part of the answer looks supported in broad meaning, but not as a close match to the source wording."
        elif support_label == "weak" or question_overlap >= 0.2:
            label = "unsupported"
            reason = "This part of the answer may be directionally reasonable, but the context support is still thin."
        else:
            label = "unsupported"
            reason = "This part of the answer does not line up clearly with the strongest support found in the context."

        if has_strong_language and label == "unsupported":
            label = "overreach"
            reason = "This part of the answer sounds stronger or more absolute than the context support allows."
        elif has_strong_language and label == "inferred" and support_score < 0.6 and not matched_context_has_strong_language:
            label = "overreach"
            reason = "This part of the answer sounds stronger or more absolute than the context support allows."
        elif not supported_claims and label == "unsupported":
            reason = "The app could not find strong enough context support to treat this part of the answer as grounded."

        support_item = {
            "sentence": sentence,
            "label": label,
            "support_ratio": round(support_score, 2),
            "context_ratio": round(support_score, 2),
            "matched_context": matched_context,
            "reason": reason,
        }
        answer_support.append(support_item)

        if label == "overreach":
            overreach_flags.append(
                {
                    "sentence": sentence,
                    "reason": reason,
                }
            )

    if not answer_support and result.get("answer"):
        answer_support = [
            {
                "sentence": str(result.get("answer", "")).strip(),
                "label": "inferred",
                "support_ratio": 0.0,
                "context_ratio": 0.0,
                "reason": "The answer could not be split cleanly into claim-sized sentences, so it is treated as a single interpretation block.",
            }
        ]

    if not overreach_flags:
        summary = "No strong overreach detected in the answer."
    elif len(overreach_flags) == 1:
        summary = "1 answer claim may go beyond what the context clearly justifies."
    else:
        summary = f"{len(overreach_flags)} answer claims may go beyond what the context clearly justifies."

    result["answer_support"] = answer_support
    result["overreach_flags"] = overreach_flags
    result["overreach_summary"] = summary
    return result


def build_confidence_breakdown(result: Dict[str, Any]) -> Dict[str, Any]:
    claims = result.get("evidence_claims", [])
    verified = sum(1 for claim in claims if claim.get("verified"))
    context_backed = sum(
        1 for claim in claims
        if claim.get("verified") or claim.get("context_support_label") in {"direct", "partial"}
    )
    total = len(claims)
    relevant = sum(1 for claim in claims if claim.get("question_relevance") == "relevant")
    weak_relevance = sum(1 for claim in claims if claim.get("question_relevance") == "weak")
    overreach_count = len(result.get("overreach_flags", []))
    confidence = str(result.get("confidence", "")).strip().lower() or "low"

    scale_meaning = {
        "high": "High means the answer is strongly grounded in the context, closely tied to the question, and not stretching past the support very much.",
        "medium": "Medium means there is real support, but there are still gaps, assumptions, or places where the answer reaches beyond the strongest support.",
        "low": "Low means the answer is weakly grounded, loosely tied to the question, or stronger than the context really allows.",
    }

    formula_steps = [
        {
            "label": "Context Grounding",
            "value": f"{context_backed}/{total}" if total else "0/0",
            "detail": "The app checks whether the explanation points to parts of the context that meaningfully support the answer, even when the wording is paraphrased.",
        },
        {
            "label": "Exact Quote Match",
            "value": f"{verified}/{total}" if total else "0/0",
            "detail": "It separately tracks how many evidence quotes matched the context word-for-word.",
        },
        {
            "label": "Question Relevance",
            "value": f"{relevant} relevant, {weak_relevance} weak",
            "detail": "It then checks whether that support actually helps answer the user question, not just whether it appears somewhere in the text.",
        },
        {
            "label": "Answer Overreach",
            "value": str(overreach_count),
            "detail": "It also checks whether parts of the answer sound stronger or more absolute than the context support justifies.",
        },
        {
            "label": "Final Confidence",
            "value": confidence.upper(),
            "detail": "The final confidence is a qualitative score based on context grounding, quote verification, question relevance, and whether the answer seems to overstate the context.",
        },
    ]

    result["confidence_breakdown"] = {
        "score": confidence,
        "what_it_means": scale_meaning.get(confidence, scale_meaning["low"]),
        "formula_steps": formula_steps,
        "plain_english": (
            "Confidence is not a probability. It is the app's judgment of how well the answer is backed by the provided context and how cleanly that support matches the question."
        ),
    }
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
