SYSTEM_PROMPT = """You are a transparency-first assistant.
Rules:
1) Use ONLY the provided QUESTION, CONTEXT, and optional MODEL_ANSWER.
2) Do not invent evidence or facts outside the context.
3) Write like you are helping a real user understand the answer.
4) Use plain, everyday language. Avoid academic or overly formal wording.
5) If you use a technical term, explain it simply.
6) Follow the exact output format requested by the user message.
7) Explain what evidence the answer seems to rely on and what assumptions connect the evidence to the conclusion.
8) If MODEL_ANSWER is provided, analyze why that answer was produced and where it may overstate or understate the context.
9) If MODEL_ANSWER is not provided, answer the question using the context first, then explain the likely reasoning path behind that answer.
10) Prefer short, clear sentences over impressive wording.
11) Stay focused and readable.
"""


def _base_sections(question: str, context: str, model_answer: str = "") -> str:
    parts = [f"QUESTION:\n{question}"]
    if model_answer.strip():
        parts.append(f"MODEL_ANSWER:\n{model_answer}")
    parts.append(f"CONTEXT:\n{context}")
    return "\n\n".join(parts)


def build_plaintext_fallback_prompt(question: str, context: str, model_answer: str = "") -> str:
    answer_line = (
        "ANSWER: a direct answer to the question in 1 to 3 clear sentences"
        if not model_answer.strip()
        else "ANSWER: a plain-English audit verdict in 1 to 3 clear sentences saying whether the model answer holds up against the context"
    )
    black_box_line = (
        "BLACK_BOX: explain in plain English how the model likely connected the context to the answer, what it emphasized, and what it may have glossed over in 2 to 3 clear sentences"
        if not model_answer.strip()
        else "BLACK_BOX: explain in plain English where the model likely focused, what it may have overweighted or missed, and why that produced the model answer in 2 to 3 clear sentences"
    )
    uncertainty_line = (
        "UNCERTAINTY: one meaningful caveat about where the answer may be too strong, too weak, or insufficiently supported in 1 clear sentence"
    )
    followup_line = (
        "FOLLOWUP: one short follow-up question, under 14 words, ending with a question mark, that would help a user test or improve the answer"
        if not model_answer.strip()
        else "FOLLOWUP: one short follow-up question, under 14 words, ending with a question mark, that would help a user test or improve the model answer"
    )

    return f"""{_base_sections(question, context, model_answer)}

Return plain text in exactly this format:
{answer_line}
{black_box_line}
QUOTE: exact quote copied verbatim from the context
ASSUMPTION: one meaningful hidden assumption or interpretation step in 1 clear sentence
{uncertainty_line}
CONFIDENCE: low, medium, or high
CONFIDENCE_REASON: a short, plain-English explanation of that confidence in 1 sentence
{followup_line}

Write for a curious user, not for a professor.
Avoid sounding academic, robotic, or overly polished.
Choose clearer words over smarter-sounding words.

Do not return JSON.
Do not return markdown.
Do not return a single character like {{.
Every field is required.
"""


def build_completion_prompt(question: str, context: str, prior_output: str, model_answer: str = "") -> str:
    prompt_body = build_plaintext_fallback_prompt(question, context, model_answer)
    format_marker = "Return plain text in exactly this format:\n"
    format_section = prompt_body.split(format_marker, 1)[1] if format_marker in prompt_body else prompt_body

    return f"""{_base_sections(question, context, model_answer)}

PRIOR OUTPUT:
{prior_output}

The prior output was incomplete or poorly formatted.
Re-answer using the context and return plain text in exactly this format:
{format_section}"""
