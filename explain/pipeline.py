from typing import Any, Dict, List
import requests

from explain.prompts import SYSTEM_PROMPT, build_plaintext_fallback_prompt
from explain.schemas import default_result, normalize_result
from explain.highlight import verify_evidence_claims, add_question_relevance, adjust_confidence, build_highlighted_context
from utils.logging import build_trace_log
from utils.text import chunk_text


MAX_CONTEXT_CHARS_FOR_MODEL = 1800
FAST_RETRY_CONTEXT_CHARS = 700
FAST_RETRY_MAX_TOKENS = 420
FINAL_RETRY_MAX_TOKENS = 320
FIELD_RETRY_MAX_TOKENS = 220


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

    def _run_plaintext_chat(self, question: str, context: str, temperature: float, max_tokens: int, steps: List[str]) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_plaintext_fallback_prompt(question, context)},
        ]
        try:
            return self._chat(messages, temperature=temperature, max_tokens=max_tokens)
        except requests.ReadTimeout:
            steps.append("llm_fast_retry_call")
            retry_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_plaintext_fallback_prompt(question, context[:FAST_RETRY_CONTEXT_CHARS])},
            ]
            return self._chat(
                retry_messages,
                temperature=0.0,
                max_tokens=min(max_tokens, FAST_RETRY_MAX_TOKENS),
                timeout_seconds=max(self.client.timeout_seconds, 35),
            )

    def _run_completion_retry(self, question: str, context: str, prior_output: str, missing: List[str]) -> str:
        field_hints = {
            "ANSWER": "ANSWER: a clear answer in 2 to 4 sentences",
            "BLACK_BOX": "BLACK_BOX: explain where the model likely focused, what it overweighted or missed, and why that produced the answer in 3 to 5 sentences",
            "QUOTE": "QUOTE: an exact quote copied verbatim from the context",
            "ASSUMPTION": "ASSUMPTION: one meaningful hidden assumption or interpretation step in 1 to 2 sentences",
            "UNCERTAINTY": "UNCERTAINTY: one caveat saying where the answer may be too strong, too weak, or under-supported in 1 to 2 sentences",
            "CONFIDENCE": "CONFIDENCE: low, medium, or high",
            "CONFIDENCE_REASON": "CONFIDENCE_REASON: a short explanation of that confidence in 1 to 2 sentences",
            "FOLLOWUP": "FOLLOWUP: one question that tests the model's interpretation or challenges its emphasis",
        }
        missing_lines = "\n".join(field_hints[field] for field in missing)
        retry_messages = [
            {"role": "system", "content": "Follow the requested plain-text format exactly. No JSON. No markdown. Return only the missing labels."},
            {
                "role": "user",
                "content": (
                    f"QUESTION:\n{question}\n\n"
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

    def _run_final_retry(self, question: str, context: str, missing: List[str]) -> str:
        field_hints = {
            "ANSWER": "ANSWER: a clear answer in 2 to 4 sentences",
            "BLACK_BOX": "BLACK_BOX: explain where the model likely focused, what it overweighted or missed, and why that produced the answer in 3 to 5 sentences",
            "QUOTE": "QUOTE: an exact quote copied verbatim from the context",
            "ASSUMPTION": "ASSUMPTION: one meaningful hidden assumption or interpretation step in 1 to 2 sentences",
            "UNCERTAINTY": "UNCERTAINTY: one caveat saying where the answer may be too strong, too weak, or under-supported in 1 to 2 sentences",
            "CONFIDENCE": "CONFIDENCE: low, medium, or high",
            "CONFIDENCE_REASON": "CONFIDENCE_REASON: a short explanation of that confidence in 1 to 2 sentences",
            "FOLLOWUP": "FOLLOWUP: one question that tests the model's interpretation or challenges its emphasis",
        }
        missing_lines = "\n".join(field_hints[field] for field in missing)
        retry_messages = [
            {"role": "system", "content": "Answer with only labeled plain text. No JSON. No markdown. No extra text."},
            {
                "role": "user",
                "content": (
                    f"Use the context only.\n\nQUESTION:\n{question}\n\nCONTEXT:\n{context[:FAST_RETRY_CONTEXT_CHARS]}\n\n"
                    "Fill every missing label below.\n"
                    "Return exactly one line per label.\n"
                    "Do not add any other text.\n\n"
                    f"{missing_lines}"
                ),
            },
        ]
        return self._chat(
            retry_messages,
            temperature=0.0,
            max_tokens=FINAL_RETRY_MAX_TOKENS,
            timeout_seconds=max(self.client.timeout_seconds, 35),
        )

    def _run_single_field_retry(self, question: str, context: str, field: str) -> str:
        prompts = {
            "ANSWER": "Write a clear direct answer in 2 to 4 sentences using only the context. No label. No markdown.",
            "BLACK_BOX": "Write a diagnostic explanation in 3 to 5 sentences showing where the model likely focused, what it may have overweighted or missed, and why that produced the answer. No label. No markdown.",
            "QUOTE": "Write an exact quote copied verbatim from the context. No label. No markdown.",
            "ASSUMPTION": "Write one meaningful hidden assumption or interpretation step the answer depends on in 1 to 2 sentences. No label. No markdown.",
            "UNCERTAINTY": "Write one meaningful caveat in 1 to 2 sentences saying where the answer may be too strong, too weak, or under-supported. No label. No markdown.",
            "CONFIDENCE": "Reply with one word only: low, medium, or high.",
            "CONFIDENCE_REASON": "Write a short explanation of the confidence in 1 to 2 sentences using only the context. No label. No markdown.",
            "FOLLOWUP": "Write one question using only the context that tests the model's interpretation or challenges its emphasis. No label. No markdown.",
        }
        retry_messages = [
            {"role": "system", "content": "Use only the provided context. Be clear, specific, and sufficiently detailed."},
            {
                "role": "user",
                "content": (
                    f"QUESTION:\n{question}\n\n"
                    f"CONTEXT:\n{context[:FAST_RETRY_CONTEXT_CHARS]}\n\n"
                    f"{prompts[field]}"
                ),
            },
        ]
        return self._chat(
            retry_messages,
            temperature=0.0,
            max_tokens=FIELD_RETRY_MAX_TOKENS,
            timeout_seconds=max(self.client.timeout_seconds, 35),
        )

    def run(
        self,
        question: str,
        context: str,
        temperature: float,
        max_tokens: int,
        critique_pass: bool = False,
    ) -> Dict[str, Any]:
        steps = ["llm_primary_call", "parse_plaintext", "verify_evidence", "adjust_confidence"]
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
        )
        result = normalize_result(parse_plaintext_fallback(raw_text))

        if not result_is_complete(result):
            steps.append("llm_completion_retry_call")
            retry_text = self._run_completion_retry(question, context_for_model, raw_text, missing_fields(result))
            retry_result = normalize_result(parse_plaintext_fallback(retry_text))
            raw_text = raw_text + "\n" + retry_text
            result = merge_results(result, retry_result)

        if not result_is_complete(result):
            steps.append("llm_final_retry_call")
            final_retry_text = self._run_final_retry(question, context_for_model, missing_fields(result))
            final_retry_result = normalize_result(parse_plaintext_fallback(final_retry_text))
            raw_text = raw_text + "\n" + final_retry_text
            result = merge_results(result, final_retry_result)

        if not result_is_complete(result):
            steps.append("llm_single_field_retry_call")
            for field in missing_fields(result):
                field_text = self._run_single_field_retry(question, context_for_model, field)
                raw_text = raw_text + "\n" + field_text
                result = merge_results(result, patch_from_field_text(field, field_text))

        if not result_is_complete(result):
            raise RuntimeError("Model returned incomplete structured output.")

        result = verify_evidence_claims(result, context)
        result = add_question_relevance(result, question)
        result = adjust_confidence(result)

        result["highlighted_context"] = build_highlighted_context(context, result.get("evidence_claims", []))
        result["trace_log"] = build_trace_log(
            backend_meta=self.client.metadata(),
            temperature=temperature,
            max_tokens=max_tokens,
            steps=steps,
            raw_preview=raw_text[:500] if raw_text else "",
        )
        return result
