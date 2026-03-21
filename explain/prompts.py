SYSTEM_PROMPT = """You are a transparency-first assistant.
Rules:
1) Use ONLY the provided CONTEXT to reason about the answer.
2) Do not invent evidence or facts outside the context.
3) Give fuller explanations when the context supports them.
4) Follow the exact output format requested by the user message.
5) Diagnose how the model likely interpreted the context, not just what it concluded.
6) In the black-box explanation, say what evidence the model likely relied on, what it emphasized too much or too little, and why that would produce the answer.
7) Call out hidden assumptions, weak support, and places where the answer may overstate or understate the context.
8) Prefer substance over brevity, but stay focused and readable.
"""


def build_plaintext_fallback_prompt(question: str, context: str) -> str:
    return f"""QUESTION:
{question}

CONTEXT:
{context}

Return plain text in exactly this format:
ANSWER: a clear direct answer in 2 to 4 sentences
BLACK_BOX: explain where the model likely focused, what it may have overweighted or missed, and why that produced the answer in 3 to 5 sentences
QUOTE: exact quote copied verbatim from the context
ASSUMPTION: one meaningful hidden assumption or interpretation step in 1 to 2 sentences
UNCERTAINTY: one meaningful caveat that says where the answer may be too strong, too weak, or insufficiently supported in 1 to 2 sentences
CONFIDENCE: low, medium, or high
CONFIDENCE_REASON: a short explanation of that confidence in 1 to 2 sentences
FOLLOWUP: one thoughtful question that tests the model's interpretation or challenges its emphasis

Do not return JSON.
Do not return markdown.
Do not return a single character like {{.
Every field is required.
"""


def build_completion_prompt(question: str, context: str, prior_output: str) -> str:
    return f"""QUESTION:
{question}

CONTEXT:
{context}

PRIOR OUTPUT:
{prior_output}

The prior output was incomplete or poorly formatted.
Re-answer using the context and return plain text in exactly this format:
ANSWER: a clear direct answer in 2 to 4 sentences
BLACK_BOX: explain where the model likely focused, what it may have overweighted or missed, and why that produced the answer in 3 to 5 sentences
QUOTE: exact quote copied verbatim from the context
ASSUMPTION: one meaningful hidden assumption or interpretation step in 1 to 2 sentences
UNCERTAINTY: one meaningful caveat that says where the answer may be too strong, too weak, or insufficiently supported in 1 to 2 sentences
CONFIDENCE: low, medium, or high
CONFIDENCE_REASON: a short explanation of that confidence in 1 to 2 sentences
FOLLOWUP: one thoughtful question that tests the model's interpretation or challenges its emphasis

Do not return JSON.
Do not return markdown.
Every field is required.
"""
