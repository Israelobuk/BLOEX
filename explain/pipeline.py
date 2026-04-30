from typing import Any, Dict, List, TypedDict
import re
import requests

from explain.prompts import SYSTEM_PROMPT, build_plaintext_fallback_prompt
from explain.schemas import default_result, normalize_result
from explain.highlight import verify_evidence_claims, add_question_relevance, adjust_confidence, detect_answer_overreach, build_confidence_breakdown, build_highlighted_context
from utils.logging import build_trace_log
from utils.text import chunk_text


MAX_CONTEXT_CHARS_FOR_MODEL = 420
FAST_RETRY_CONTEXT_CHARS = 220
FAST_RETRY_MAX_TOKENS = 90
LABEL_PATTERN = re.compile(
    r"\b(ANSWER|BLACK_BOX|QUOTE|ASSUMPTION|UNCERTAINTY|CONFIDENCE_REASON|CONFIDENCE)\s*:",
    re.IGNORECASE,
)


class OllamaAgentClient:
    def __init__(self, client):
        self._client = client

    def chat(
        self,
        *,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        timeout_seconds: int | None = None,
    ) -> str:
        return self._client.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
        )


def _first_sentence(text: str) -> str:
    clean = " ".join(str(text or "").split()).strip()
    if not clean:
        return ""
    for sep in ".!?":
        if sep in clean:
            return clean.split(sep, 1)[0].strip()
    return clean[:180].strip()


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
        "evidence_claims": [],
    }
    current_key = ""
    matches = list(LABEL_PATTERN.finditer(text))
    if matches:
        label_map = {
            "answer": "answer",
            "black_box": "black_box_explanation",
            "quote": "quote",
            "assumption": "assumption",
            "uncertainty": "uncertainty",
            "confidence": "confidence",
            "confidence_reason": "confidence_reason",
        }
        for index, match in enumerate(matches):
            raw_key = match.group(1).lower()
            key = label_map.get(raw_key)
            if not key:
                continue
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            value = " ".join(text[start:end].strip().split())
            if not value:
                continue
            if key == "answer":
                parsed["answer"] = f"{parsed.get('answer', '')} {value}".strip()
            elif key == "black_box_explanation":
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
        return parsed

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

    return parsed


def _clean_instruction_echo(result: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = dict(result)
    marker = re.compile(r"\b(BLACK_BOX|QUOTE|ASSUMPTION|UNCERTAINTY|CONFIDENCE_REASON|CONFIDENCE|QUESTION)\s*:", re.IGNORECASE)
    banned_fragments = [
        "return plain text in exactly this format",
        "a plain-english audit verdict in 1 to 3 clear sentences",
        "explain in plain english how the model likely connected",
        "explain in plain english where the model likely focused",
        "exact quote copied verbatim from the context",
        "one meaningful hidden assumption or interpretation step in 1 clear sentence",
        "one meaningful caveat about where the answer may be too strong, too weak, or insufficiently supported in 1 clear sentence",
        "a short, plain-english explanation of that confidence in 1 sentence",
    ]
    for field in ["answer", "black_box_explanation", "confidence_reason"]:
        value = str(cleaned.get(field, "") or "").strip()
        if not value:
            continue
        # Drop template/instruction leakage from weak-model outputs.
        hit = marker.search(value)
        if hit:
            value = value[: hit.start()].strip()
        lowered = value.lower()
        if any(fragment in lowered for fragment in banned_fragments):
            value = ""
        cleaned[field] = value
    return cleaned


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
        ]
    )


def merge_results(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)

    for key in ["answer", "black_box_explanation", "confidence", "confidence_reason"]:
        if not merged.get(key) and patch.get(key):
            merged[key] = patch[key]

    for key in ["assumptions", "uncertainty", "evidence_claims"]:
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

    return missing


def patch_from_field_text(field: str, text: str) -> Dict[str, Any]:
    clean = " ".join(text.strip().split())
    if not clean:
        return {}

    parsed = parse_plaintext_fallback(clean)
    normalized = normalize_result(parsed)
    if any(normalized.get(key) for key in ["answer", "black_box_explanation", "confidence", "confidence_reason", "assumptions", "uncertainty", "evidence_claims"]):
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

    return {}


class ExplainerPipeline:
    def __init__(self, client):
        self.client = client
        self.agent_client = OllamaAgentClient(client)

    def _chat(self, messages: List[Dict[str, str]], temperature: float, max_tokens: int, timeout_seconds: int | None = None) -> str:
        return self.agent_client.chat(
            messages=messages,
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

    def _compose_context_with_signals(self, context: str, signal_hint: str) -> str:
        signal_text = " ".join(str(signal_hint or "").split()).strip()
        if not signal_text:
            return context
        return f"{context}\n\nSTRUCTURED_SIGNALS:\n{signal_text}".strip()

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
        field_hints = {
            "ANSWER": answer_hint,
            "BLACK_BOX": black_box_hint,
            "QUOTE": "QUOTE: an exact quote copied verbatim from the context",
            "ASSUMPTION": "ASSUMPTION: one meaningful hidden assumption or interpretation step in 1 simple sentence",
            "UNCERTAINTY": "UNCERTAINTY: one caveat saying where the answer may be too strong, too weak, or under-supported in 1 simple sentence",
            "CONFIDENCE": "CONFIDENCE: low, medium, or high",
            "CONFIDENCE_REASON": "CONFIDENCE_REASON: a short plain-English explanation of that confidence in 1 sentence",
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
        completion_retry: bool = False,
        model_answer: str = "",
        signal_hint: str = "",
    ) -> Dict[str, Any]:
        steps = ["llm_primary_call", "parse_plaintext", "verify_evidence", "detect_answer_overreach", "adjust_confidence", "build_confidence_breakdown"]
        raw_text = ""
        effective_context = self._compose_context_with_signals(context, signal_hint)
        context_for_model = effective_context
        trimmed_context = False

        chunks = chunk_text(effective_context, max_chars=MAX_CONTEXT_CHARS_FOR_MODEL, overlap=0)
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
        result = _clean_instruction_echo(result)

        if completion_retry and not result_is_complete(result):
            steps.append("llm_completion_retry_call")
            retry_text = self._run_completion_retry(question, context_for_model, raw_text, missing_fields(result), model_answer=model_answer)
            retry_result = normalize_result(parse_plaintext_fallback(retry_text))
            retry_result = _clean_instruction_echo(retry_result)
            raw_text = raw_text + "\n" + retry_text
            result = merge_results(result, retry_result)

        if not result_is_complete(result):
            raise ValueError("Model returned incomplete structured output after retry.")

        result = verify_evidence_claims(result, effective_context)
        result = add_question_relevance(result, question)
        if model_answer.strip():
            original_answer = result.get("answer", "")
            result["audited_answer"] = model_answer
            result["audit_verdict"] = original_answer
            result["answer"] = model_answer
            result = detect_answer_overreach(result, question, effective_context)
            result["answer"] = original_answer
        else:
            result = detect_answer_overreach(result, question, effective_context)
        result = adjust_confidence(result)
        result = build_confidence_breakdown(result)

        result["highlighted_context"] = build_highlighted_context(effective_context, result.get("evidence_claims", []))
        result["trace_log"] = build_trace_log(
            backend_meta=self.client.metadata(),
            temperature=temperature,
            max_tokens=max_tokens,
            steps=steps,
            raw_preview=raw_text[:500] if raw_text else "",
        )
        return result

    def _finalize_result(self, result: Dict[str, Any], *, question: str, context: str, model_answer: str) -> Dict[str, Any]:
        finalized = dict(result)
        finalized = verify_evidence_claims(finalized, context)
        finalized = add_question_relevance(finalized, question)
        if model_answer.strip():
            original_answer = finalized.get("answer", "")
            finalized["audited_answer"] = model_answer
            finalized["audit_verdict"] = original_answer
            finalized["answer"] = model_answer
            finalized = detect_answer_overreach(finalized, question, context)
            finalized["answer"] = original_answer
        else:
            finalized = detect_answer_overreach(finalized, question, context)
        finalized = adjust_confidence(finalized)
        finalized = build_confidence_breakdown(finalized)
        finalized["highlighted_context"] = build_highlighted_context(context, finalized.get("evidence_claims", []))
        return finalized

    def _judge_quality(self, *, question: str, model_answer: str, result: Dict[str, Any]) -> bool:
        if not result_is_complete(result):
            return False
        answer = str(result.get("answer") or "")
        explanation = str(result.get("black_box_explanation") or "")
        if len(answer.strip()) < 25 or len(explanation.strip()) < 30:
            return False
        if not model_answer.strip():
            return True
        judge_prompt = (
            "Return one token only: PASS or FAIL.\n"
            "PASS only if the audit clearly evaluates the model answer against the question with concrete reasoning.\n\n"
            f"Question:\n{question}\n\n"
            f"Model answer under audit:\n{model_answer[:1200]}\n\n"
            f"Audit answer:\n{answer[:1200]}\n\n"
            f"Audit explanation:\n{explanation[:1200]}\n"
        )
        try:
            judged = self._chat(
                [{"role": "user", "content": judge_prompt}],
                temperature=0.0,
                max_tokens=8,
                timeout_seconds=max(self.client.timeout_seconds, 20),
            )
            return "PASS" in str(judged or "").upper()
        except Exception:
            return True

    def run_agentic(
        self,
        question: str,
        context: str,
        temperature: float,
        max_tokens: int,
        critique_pass: bool = False,
        completion_retry: bool = False,
        model_answer: str = "",
        max_agent_loops: int = 2,
        signal_hint: str = "",
    ) -> Dict[str, Any]:
        class AgentState(TypedDict, total=False):
            attempts: int
            steps: List[str]
            raw_text: str
            parsed: Dict[str, Any]
            missing: List[str]
            quality_ok: bool

        effective_context = self._compose_context_with_signals(context, signal_hint)

        def _trimmed_context() -> str:
            chunks = chunk_text(effective_context, max_chars=MAX_CONTEXT_CHARS_FOR_MODEL, overlap=0)
            return chunks[0] if chunks else effective_context

        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError:
            return self.run(
                question=question,
                    context=context,
                    signal_hint=signal_hint,
                    temperature=temperature,
                    max_tokens=max_tokens,
                critique_pass=critique_pass,
                completion_retry=completion_retry,
                model_answer=model_answer,
            )

        context_for_model = _trimmed_context()

        def generate_primary(state: AgentState) -> AgentState:
            steps = list(state.get("steps", []))
            steps.append("agent_generate_primary")
            raw = self._run_plaintext_chat(
                question=question,
                context=context_for_model,
                temperature=temperature,
                max_tokens=max_tokens,
                steps=steps,
                model_answer=model_answer,
            )
            parsed = _clean_instruction_echo(normalize_result(parse_plaintext_fallback(raw)))
            return {
                "attempts": int(state.get("attempts", 0)) + 1,
                "steps": steps,
                "raw_text": raw,
                "parsed": parsed,
                "missing": missing_fields(parsed),
            }

        def evaluate_quality(state: AgentState) -> AgentState:
            steps = list(state.get("steps", []))
            steps.append("agent_evaluate_quality")
            parsed = dict(state.get("parsed", {}))
            missing = list(state.get("missing", []))
            quality_ok = not missing and self._judge_quality(
                question=question,
                model_answer=model_answer,
                result=parsed,
            )
            return {"steps": steps, "quality_ok": quality_ok}

        def repair_output(state: AgentState) -> AgentState:
            steps = list(state.get("steps", []))
            steps.append("agent_repair_output")
            parsed = dict(state.get("parsed", {}))
            raw_text = str(state.get("raw_text", ""))
            missing = list(state.get("missing", []))
            if completion_retry and missing:
                retry_text = self._run_completion_retry(
                    question=question,
                    context=context_for_model,
                    prior_output=raw_text,
                    missing=missing,
                    model_answer=model_answer,
                )
                patch = _clean_instruction_echo(normalize_result(parse_plaintext_fallback(retry_text)))
                parsed = merge_results(parsed, patch)
                raw_text = f"{raw_text}\n{retry_text}".strip()
            return {
                "attempts": int(state.get("attempts", 0)) + 1,
                "steps": steps,
                "raw_text": raw_text,
                "parsed": parsed,
                "missing": missing_fields(parsed),
            }

        def finalize(state: AgentState) -> AgentState:
            steps = list(state.get("steps", []))
            steps.append("agent_finalize")
            parsed = dict(state.get("parsed", {}))
            if not result_is_complete(parsed):
                raise ValueError("Model returned incomplete structured output after retry.")
            result = self._finalize_result(parsed, question=question, context=effective_context, model_answer=model_answer)
            result["trace_log"] = build_trace_log(
                backend_meta=self.client.metadata(),
                temperature=temperature,
                max_tokens=max_tokens,
                steps=steps,
                raw_preview=str(state.get("raw_text", ""))[:500],
            )
            return {"parsed": result, "steps": steps}

        def route_after_quality(state: AgentState) -> str:
            if state.get("quality_ok"):
                return "finalize"
            if int(state.get("attempts", 0)) >= max(1, max_agent_loops):
                return "finalize"
            return "repair_output"

        def route_after_repair(state: AgentState) -> str:
            return "evaluate_quality"

        graph = StateGraph(AgentState)
        graph.add_node("generate_primary", generate_primary)
        graph.add_node("evaluate_quality", evaluate_quality)
        graph.add_node("repair_output", repair_output)
        graph.add_node("finalize", finalize)
        graph.add_edge(START, "generate_primary")
        graph.add_edge("generate_primary", "evaluate_quality")
        graph.add_conditional_edges("evaluate_quality", route_after_quality, {"finalize": "finalize", "repair_output": "repair_output"})
        graph.add_conditional_edges("repair_output", route_after_repair, {"evaluate_quality": "evaluate_quality"})
        graph.add_edge("finalize", END)

        compiled = graph.compile()
        state = compiled.invoke({"attempts": 0, "steps": []})
        return dict(state.get("parsed", {}))
