from typing import Any, Dict, List
import requests

from explain.prompts import SYSTEM_PROMPT, build_plaintext_fallback_prompt
from explain.schemas import default_result, normalize_result
from explain.highlight import verify_evidence_claims, add_question_relevance, adjust_confidence, detect_answer_overreach, build_confidence_breakdown, build_highlighted_context
from utils.logging import build_trace_log
from utils.text import chunk_text


MAX_CONTEXT_CHARS_FOR_MODEL = 600
FAST_RETRY_CONTEXT_CHARS = 320
FAST_RETRY_MAX_TOKENS = 120


def _first_sentence(text: str) -> str:
    clean = " ".join(str(text or "").split()).strip()
    if not clean:
        return ""
    for sep in ".!?":
        if sep in clean:
            return clean.split(sep, 1)[0].strip()
    return clean[:180].strip()


def _best_followup(question: str) -> str:
    clean = " ".join(str(question or "").split()).strip()
    if not clean:
        return "What evidence would make this answer more trustworthy?"
    if clean.endswith("?"):
        return f"What evidence best supports the answer to: {clean}"
    return f"What evidence best supports the answer to: {clean}?"


def build_fallback_result(
    question: str,
    context: str,
    model_answer: str,
    backend_meta: Dict[str, Any],
    temperature: float,
    max_tokens: int,
    error_message: str,
) -> Dict[str, Any]:
    answer_text = " ".join(str(model_answer or "").split()).strip()
    lead = _first_sentence(answer_text)
    context_text = " ".join(str(context or "").split()).strip()

    result = default_result()
    result["answer"] = answer_text or "No model answer was provided to analyze."
    result["audit_verdict"] = (
        "The full explainer model failed, so this is a lightweight fallback review of the pasted answer."
    )
    result["black_box_explanation"] = (
        f"The pasted answer appears to focus most on '{lead}'." if lead else
        "The pasted answer could not be deeply analyzed because the model backend failed during generation."
    )
    result["assumptions"] = [
        "This fallback review assumes the pasted answer reflects the main claim the user wants inspected."
    ]
    result["uncertainty"] = [
        "This is a fallback result, not a full model-generated explanation, so treat it as a rough audit only.",
        "Without a stable model response, the app cannot reliably break down supporting vs unsupported reasoning.",
    ]
    if not context_text:
        result["uncertainty"].append(
            "No source context was provided, so the app cannot verify whether the answer is actually grounded."
        )

    result["confidence"] = "low"
    result["confidence_reason"] = (
        "Confidence is low because the explainer model failed during generation, so this result is only a fallback audit."
    )
    result["followups"] = [_best_followup(question)]

    if context_text:
        quote = _first_sentence(context_text)[:220]
        if quote:
            result["evidence_claims"] = [
                {
                    "claim": "Closest available context snippet from fallback mode",
                    "support_reason": "The model-backed extraction failed, so the app surfaced a basic source snippet instead.",
                    "quote": quote,
                    "start": None,
                    "end": None,
                    "verified": False,
                }
            ]

    result = verify_evidence_claims(result, context)
    result = add_question_relevance(result, question)
    if model_answer.strip():
        original_answer = result.get("answer", "")
        result["audited_answer"] = model_answer
        result["answer"] = model_answer
        result = detect_answer_overreach(result, question, context)
        result["answer"] = original_answer
    else:
        result = detect_answer_overreach(result, question, context)
    result = adjust_confidence(result)
    result = build_confidence_breakdown(result)
    result["highlighted_context"] = build_highlighted_context(context, result.get("evidence_claims", []))
    result["trace_log"] = build_trace_log(
        backend_meta=backend_meta,
        temperature=temperature,
        max_tokens=max_tokens,
        steps=["fallback_mode"],
        raw_preview=str(error_message)[:500],
    )
    result["fallback_mode"] = True
    result["fallback_error"] = error_message
    return result


def parse_plaintext_fallback(text: str) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {
        "assumptions": [],
        "uncertainty": [],
        "followups": [],
        "evidence_claims": [],
    }
    current_key = ""

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip().lower().replace("-", "_").replace(" ", "_")
            value = value.strip()
            if key in {
                "answer",
                "black_box",
                "quote",
                "assumption",
                "uncertainty",
                "confidence",
                "confidence_reason",
                "followup",
                "follow_up",
            }:
                current_key = key
            else:
                if current_key:
                    key = current_key
                    value = line
                else:
                    current_key = ""
                    continue
        elif current_key:
            key = current_key
            value = line
        else:
            continue

        if key == "answer":
            parsed["answer"] = f"{parsed.get('answer', '')} {value}".strip()
        elif key == "black_box":
            parsed["black_box_explanation"] = f"{parsed.get('black_box_explanation', '')} {value}".strip()
        elif key == "quote":
            if not parsed["evidence_claims"]:
                parsed["evidence_claims"] = [
                    {
                        "claim": "Quoted support from the model output",
                        "support_reason": "This was the main quote the model chose to justify the answer.",
                        "quote": value,
                        "start": None,
                        "end": None,
                        "verified": False,
                    }
                ]
            else:
                parsed["evidence_claims"][0]["quote"] = f"{parsed['evidence_claims'][0].get('quote', '')} {value}".strip()
        elif key == "assumption":
            parsed["assumptions"].append(value)
        elif key == "uncertainty":
            parsed["uncertainty"].append(value)
        elif key == "confidence":
            lowered = value.lower()
            if lowered.startswith("high"):
                parsed["confidence"] = "high"
            elif lowered.startswith("medium"):
                parsed["confidence"] = "medium"
            elif lowered.startswith("low"):
                parsed["confidence"] = "low"
            remainder = value.split(" ", 1)[1].strip(" -:") if " " in value else ""
            if remainder and not parsed.get("confidence_reason"):
                parsed["confidence_reason"] = remainder
        elif key == "confidence_reason":
            parsed["confidence_reason"] = value
        elif key in {"followup", "follow_up"}:
            parsed["followups"].append(value)

    return parsed


def result_is_complete(result: Dict[str, Any]) -> bool:
    return all(
        [
            (result.get("answer") or "").strip(),
            (result.get("black_box_explanation") or "").strip(),
            (result.get("confidence") or "").strip(),
            (result.get("confidence_reason") or "").strip(),
            result.get("evidence_claims"),
            result.get("assumptions"),
            result.get("uncertainty"),
            result.get("followups"),
        ]
    )


def merge_results(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)

    for key in ["answer", "black_box_explanation", "confidence", "confidence_reason"]:
        if not merged.get(key) and patch.get(key):
            merged[key] = patch[key]

    for key in ["assumptions", "uncertainty", "followups", "evidence_claims"]:
        if not merged.get(key) and patch.get(key):
            merged[key] = patch[key]

    return merged


def missing_fields(result: Dict[str, Any]) -> List[str]:
    missing: List[str] = []

    if not (result.get("answer") or "").strip():
        missing.append("ANSWER")
    if not (result.get("black_box_explanation") or "").strip():
        missing.append("BLACK_BOX")
    if not result.get("evidence_claims"):
        missing.append("QUOTE")
    if not result.get("assumptions"):
        missing.append("ASSUMPTION")
    if not result.get("uncertainty"):
        missing.append("UNCERTAINTY")
    if not (result.get("confidence") or "").strip():
        missing.append("CONFIDENCE")
    if not (result.get("confidence_reason") or "").strip():
        missing.append("CONFIDENCE_REASON")
    if not result.get("followups"):
        missing.append("FOLLOWUP")

    return missing


def patch_from_field_text(field: str, text: str) -> Dict[str, Any]:
    clean = " ".join(text.strip().split())
    if not clean:
        return {}

    parsed = parse_plaintext_fallback(clean)
    normalized = normalize_result(parsed)
    if any(normalized.get(key) for key in ["answer", "black_box_explanation", "confidence", "confidence_reason", "assumptions", "uncertainty", "followups", "evidence_claims"]):
        return normalized

    if field == "ANSWER":
        return normalize_result({"answer": clean})
    if field == "BLACK_BOX":
        return normalize_result({"black_box_explanation": clean})
    if field == "QUOTE":
        return normalize_result(
            {
                "evidence_claims": [
                    {
                        "claim": "Quoted support from the model output",
                        "support_reason": "This was the main quote the model chose to justify the answer.",
                        "quote": clean,
                        "start": None,
                        "end": None,
                        "verified": False,
                    }
                ]
            }
        )
    if field == "ASSUMPTION":
        return normalize_result({"assumptions": [clean]})
    if field == "UNCERTAINTY":
        return normalize_result({"uncertainty": [clean]})
    if field == "CONFIDENCE":
        return normalize_result({"confidence": clean})
    if field == "CONFIDENCE_REASON":
        return normalize_result({"confidence_reason": clean})
    if field == "FOLLOWUP":
        return normalize_result({"followups": [clean]})

    return {}


def fill_missing_defaults(result: Dict[str, Any], question: str, context: str, model_answer: str = "") -> Dict[str, Any]:
    patched = dict(result)

    if not (patched.get("answer") or "").strip():
        if model_answer.strip():
            patched["answer"] = "The pasted answer may be directionally useful, but it still needs review."
        else:
            patched["answer"] = "The source does not support a confident answer yet."

    if not (patched.get("black_box_explanation") or "").strip():
        if model_answer.strip():
            patched["black_box_explanation"] = (
                "The model seems to have leaned on the main claim in the pasted answer, "
                "but the full reasoning trace was incomplete."
            )
        else:
            patched["black_box_explanation"] = (
                "The model seems to have focused on the most obvious part of the source, "
                "but the full reasoning trace was incomplete."
            )

    if not patched.get("assumptions"):
        patched["assumptions"] = [
            "This review assumes the pasted answer reflects the main claim the user wants checked."
            if model_answer.strip()
            else "This review assumes the source contains the main information needed to answer the question."
        ]

    if not patched.get("uncertainty"):
        patched["uncertainty"] = [
            "Some parts may still be under-supported because the model returned an incomplete explanation."
        ]

    if not patched.get("followups"):
        patched["followups"] = [_best_followup(question)]

    if not (patched.get("confidence") or "").strip():
        patched["confidence"] = "low"

    if not (patched.get("confidence_reason") or "").strip():
        patched["confidence_reason"] = "Confidence is limited because the model returned incomplete structured output."

    if not patched.get("evidence_claims") and context.strip():
        quote = _first_sentence(context)[:220]
        if quote:
            patched["evidence_claims"] = [
                {
                    "claim": "Closest available source snippet",
                    "support_reason": "The model did not return a full evidence extraction, so the app surfaced a simple source snippet.",
                    "quote": quote,
                    "start": None,
                    "end": None,
                    "verified": False,
                }
            ]

    return normalize_result(patched)


class ExplainerPipeline:
    def __init__(self, client):
        self.client = client

    def _chat(self, messages: List[Dict[str, str]], temperature: float, max_tokens: int, timeout_seconds: int | None = None) -> str:
        return self.client.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
        )

    def _run_plaintext_chat(self, question: str, context: str, temperature: float, max_tokens: int, steps: List[str], model_answer: str = "") -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_plaintext_fallback_prompt(question, context, model_answer)},
        ]
        try:
            return self._chat(messages, temperature=temperature, max_tokens=max_tokens)
        except requests.ReadTimeout:
            steps.append("llm_fast_retry_call")
            retry_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_plaintext_fallback_prompt(question, context[:FAST_RETRY_CONTEXT_CHARS], model_answer)},
            ]
            return self._chat(
                retry_messages,
                temperature=0.0,
                max_tokens=min(max_tokens, FAST_RETRY_MAX_TOKENS),
                timeout_seconds=max(self.client.timeout_seconds, 35),
            )

    def _run_completion_retry(self, question: str, context: str, prior_output: str, missing: List[str], model_answer: str = "") -> str:
        answer_hint = (
            "ANSWER: a direct answer to the question in 1 to 3 simple sentences"
            if not model_answer.strip()
            else "ANSWER: a plain-English audit verdict in 1 to 3 simple sentences saying whether the model answer holds up against the context"
        )
        black_box_hint = (
            "BLACK_BOX: explain in plain English how the model likely connected the context to the answer, what it emphasized, and what it may have glossed over in 2 to 3 simple sentences"
            if not model_answer.strip()
            else "BLACK_BOX: explain in plain English where the model likely focused, what it overweighted or missed, and why that produced the model answer in 2 to 3 simple sentences"
        )
        followup_hint = (
            "FOLLOWUP: one short follow-up question, under 14 words, ending with a question mark, that would help a user test or improve the answer"
            if not model_answer.strip()
            else "FOLLOWUP: one short follow-up question, under 14 words, ending with a question mark, that would help a user test or improve the model answer"
        )
        field_hints = {
            "ANSWER": answer_hint,
            "BLACK_BOX": black_box_hint,
            "QUOTE": "QUOTE: an exact quote copied verbatim from the context",
            "ASSUMPTION": "ASSUMPTION: one meaningful hidden assumption or interpretation step in 1 simple sentence",
            "UNCERTAINTY": "UNCERTAINTY: one caveat saying where the answer may be too strong, too weak, or under-supported in 1 simple sentence",
            "CONFIDENCE": "CONFIDENCE: low, medium, or high",
            "CONFIDENCE_REASON": "CONFIDENCE_REASON: a short plain-English explanation of that confidence in 1 sentence",
            "FOLLOWUP": followup_hint,
        }
        missing_lines = "\n".join(field_hints[field] for field in missing)
        model_answer_block = f"MODEL_ANSWER:\n{model_answer}\n\n" if model_answer.strip() else ""
        retry_messages = [
            {
                "role": "system",
                "content": (
                    "Follow the requested plain-text format exactly. "
                    "No JSON. No markdown. Return only the missing labels. "
                    "Use plain, everyday language and avoid academic wording."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"QUESTION:\n{question}\n\n"
                    f"{model_answer_block}"
                    f"CONTEXT:\n{context[:FAST_RETRY_CONTEXT_CHARS]}\n\n"
                    "The prior output was incomplete. Fill only the missing labels below.\n"
                    "Return exactly one line per label.\n"
                    "Do not add any other text.\n\n"
                    f"{missing_lines}"
                ),
            },
        ]
        return self._chat(
            retry_messages,
            temperature=0.0,
            max_tokens=FAST_RETRY_MAX_TOKENS,
            timeout_seconds=max(self.client.timeout_seconds, 35),
        )

    def run(
        self,
        question: str,
        context: str,
        temperature: float,
        max_tokens: int,
        critique_pass: bool = False,
        model_answer: str = "",
    ) -> Dict[str, Any]:
        steps = ["llm_primary_call", "parse_plaintext", "verify_evidence", "detect_answer_overreach", "adjust_confidence", "build_confidence_breakdown"]
        raw_text = ""
        context_for_model = context
        trimmed_context = False

        chunks = chunk_text(context, max_chars=MAX_CONTEXT_CHARS_FOR_MODEL, overlap=0)
        if chunks:
            context_for_model = chunks[0]
            trimmed_context = len(context_for_model) < len(context)

        raw_text = self._run_plaintext_chat(
            question=question,
            context=context_for_model,
            temperature=temperature,
            max_tokens=max_tokens,
            steps=steps,
            model_answer=model_answer,
        )
        result = normalize_result(parse_plaintext_fallback(raw_text))

        if not result_is_complete(result):
            steps.append("llm_completion_retry_call")
            retry_text = self._run_completion_retry(question, context_for_model, raw_text, missing_fields(result), model_answer=model_answer)
            retry_result = normalize_result(parse_plaintext_fallback(retry_text))
            raw_text = raw_text + "\n" + retry_text
            result = merge_results(result, retry_result)

        if not result_is_complete(result):
            steps.append("fill_missing_defaults")
            result = fill_missing_defaults(result, question, context, model_answer=model_answer)

        result = verify_evidence_claims(result, context)
        result = add_question_relevance(result, question)
        if model_answer.strip():
            original_answer = result.get("answer", "")
            result["audited_answer"] = model_answer
            result["audit_verdict"] = original_answer
            result["answer"] = model_answer
            result = detect_answer_overreach(result, question, context)
            result["answer"] = original_answer
        else:
            result = detect_answer_overreach(result, question, context)
        result = adjust_confidence(result)
        result = build_confidence_breakdown(result)

        result["highlighted_context"] = build_highlighted_context(context, result.get("evidence_claims", []))
        result["trace_log"] = build_trace_log(
            backend_meta=self.client.metadata(),
            temperature=temperature,
            max_tokens=max_tokens,
            steps=steps,
            raw_preview=raw_text[:500] if raw_text else "",
        )
        return result
