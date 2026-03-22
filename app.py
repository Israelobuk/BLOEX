from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from config import load_from_env
from explain.pipeline import ExplainerPipeline
from llm import create_client


FOLLOWUP_SYSTEM_PROMPT = """
You are a serious technical assistant.
Rules:
1) Be direct and practical.
2) Start with a direct answer first.
3) Give enough detail to fully explain the answer.
4) Explain what the model seems to be focusing on, missing, overstating, or understating when relevant.
5) Use clear reasoning and examples when they help.
6) If the follow-up is unrelated to the context, answer directly.
"""


st.set_page_config(page_title="Black Box Explainer", layout="wide")

MODEL_OPTIONS = [
    ("phi3:mini", "Phi-3 Mini"),
    ("llama3.1:8b", "Llama 3.1 8B"),
    ("gpt-oss:120b", "GPT-OSS 120B"),
]

MODEL_DESCRIPTIONS = {
    "phi3:mini": "Fast and lightweight. Good for shorter explanations and quicker response times.",
    "llama3.1:8b": "More capable and detailed. Better when you want stronger reasoning and fuller writeups.",
    "gpt-oss:120b": "Hosted-scale reasoning model. Best when you want a stronger cloud backend with more depth.",
}


def init_state():
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "last_question" not in st.session_state:
        st.session_state.last_question = ""
    if "last_context" not in st.session_state:
        st.session_state.last_context = ""
    if "followup_chat_history" not in st.session_state:
        st.session_state.followup_chat_history = []


def get_streamlit_secrets() -> dict:
    try:
        return st.secrets.to_dict()
    except Exception:
        try:
            return dict(st.secrets)
        except Exception:
            return {}


def inject_font_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');

        :root {
            --bbe-bg: #f4f7fb;
            --bbe-panel: rgba(255, 255, 255, 0.78);
            --bbe-panel-strong: #ffffff;
            --bbe-border: rgba(15, 23, 42, 0.12);
            --bbe-border-strong: rgba(56, 189, 248, 0.42);
            --bbe-ink: #0f172a;
            --bbe-muted: #5b6b81;
            --bbe-accent: #38bdf8;
            --bbe-accent-deep: #0ea5e9;
            --bbe-shadow: 0 18px 48px rgba(15, 23, 42, 0.08);
            --bbe-shadow-soft: 0 10px 28px rgba(15, 23, 42, 0.06);
        }

        .stApp,
        [data-testid="stAppViewContainer"] {
            background: var(--bbe-bg);
            color: var(--bbe-ink);
            font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
        }

        html, body, [class*="css"] {
            font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
        }

        .block-container {
            padding-top: 2.3rem;
            padding-bottom: 2.75rem;
        }

        h1, h2, h3 {
            font-family: "Space Grotesk", "IBM Plex Sans", sans-serif;
            letter-spacing: -0.04em;
            color: var(--bbe-ink);
        }

        h1 {
            font-size: clamp(2.8rem, 4.8vw, 4.3rem);
            line-height: 0.95;
            margin-bottom: 0.55rem;
        }

        [data-testid="stCaptionContainer"],
        [data-testid="stMarkdownContainer"] p,
        label,
        .stTextInput label,
        .stTextArea label,
        .stSelectbox label {
            color: var(--bbe-muted);
        }

        [data-testid="stMetric"] {
            background: var(--bbe-panel);
            border: 1px solid var(--bbe-border);
            border-radius: 24px;
            padding: 1rem 1.1rem;
            box-shadow: var(--bbe-shadow-soft);
            backdrop-filter: blur(12px);
        }

        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"] {
            color: var(--bbe-ink);
        }

        [data-testid="stMetricValue"] {
            font-family: "Space Grotesk", "IBM Plex Sans", sans-serif;
            letter-spacing: -0.05em;
        }

        [data-testid="stExpander"] {
            background: var(--bbe-panel);
            border: 1px solid var(--bbe-border);
            border-radius: 24px;
            box-shadow: var(--bbe-shadow);
            backdrop-filter: blur(16px);
            overflow: hidden;
        }

        [data-testid="stExpander"] summary {
            font-family: "Space Grotesk", "IBM Plex Sans", sans-serif;
            color: var(--bbe-ink);
            letter-spacing: -0.03em;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--bbe-panel);
            border: 1px solid var(--bbe-border) !important;
            border-radius: 28px !important;
            box-shadow: var(--bbe-shadow);
            backdrop-filter: blur(16px);
        }

        .stTextInput input,
        .stTextArea textarea,
        .stSelectbox [data-baseweb="select"] > div,
        .stChatInput textarea {
            background: rgba(255, 255, 255, 0.92) !important;
            border: 1px solid rgba(148, 163, 184, 0.26) !important;
            border-radius: 20px !important;
            color: var(--bbe-ink) !important;
            box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.35);
            transition: border-color 160ms ease, box-shadow 160ms ease, transform 160ms ease;
        }

        .stTextInput input:focus,
        .stTextArea textarea:focus,
        .stChatInput textarea:focus {
            border-color: var(--bbe-border-strong) !important;
            box-shadow:
                0 0 0 4px rgba(56, 189, 248, 0.12),
                inset 0 0 0 1px rgba(255, 255, 255, 0.42) !important;
            transform: translateY(-1px);
        }

        .stTextInput input:disabled {
            color: #718197 !important;
            opacity: 1 !important;
        }

        .stButton > button,
        .stDownloadButton > button {
            background: #0f172a;
            color: #f8fafc;
            border: 1px solid rgba(56, 189, 248, 0.26);
            border-radius: 999px;
            padding: 0.7rem 1.35rem;
            font-family: "Space Grotesk", "IBM Plex Sans", sans-serif;
            letter-spacing: -0.02em;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.12);
            transition:
                transform 150ms ease,
                box-shadow 180ms ease,
                border-color 180ms ease,
                background-color 180ms ease;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            background: #111f36;
            border-color: rgba(56, 189, 248, 0.45);
            box-shadow: 0 16px 34px rgba(14, 165, 233, 0.18);
            transform: translateY(-1px);
        }

        .stButton > button:active,
        .stDownloadButton > button:active {
            transform: translateY(0) scale(0.985);
        }

        [data-testid="stTabs"] [role="tablist"] {
            gap: 0.6rem;
            border-bottom: 1px solid rgba(15, 23, 42, 0.10);
            padding-bottom: 0.35rem;
            margin-bottom: 0.7rem;
        }

        [data-testid="stTabs"] [role="tab"] {
            position: relative;
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 999px;
            padding: 0.72rem 1.35rem 0.95rem;
            min-height: 3.15rem;
            background: rgba(255, 255, 255, 0.76);
            color: var(--bbe-muted);
            box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.18);
            transition:
                color 140ms ease,
                background-color 180ms ease,
                border-color 180ms ease,
                transform 140ms ease,
                box-shadow 180ms ease;
        }

        [data-testid="stTabs"] [role="tab"]:hover {
            color: var(--bbe-ink);
            border-color: rgba(56, 189, 248, 0.28);
            box-shadow: 0 10px 22px rgba(15, 23, 42, 0.08);
            transform: translateY(-1px);
        }

        [data-testid="stTabs"] [role="tab"]:active {
            transform: translateY(0) scale(0.98);
        }

        [data-testid="stTabs"] [role="tab"]::after {
            content: "";
            position: absolute;
            left: 1rem;
            right: 1rem;
            bottom: 0.34rem;
            height: 3px;
            border-radius: 999px;
            background: var(--bbe-accent);
            transform: scaleX(0.32);
            transform-origin: center;
            opacity: 0;
            transition: transform 180ms ease, opacity 180ms ease;
        }

        [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
            color: var(--bbe-ink);
            background: rgba(255, 255, 255, 0.96);
            border-color: rgba(56, 189, 248, 0.36);
            box-shadow:
                0 14px 34px rgba(15, 23, 42, 0.10),
                0 0 0 1px rgba(56, 189, 248, 0.10);
            animation: tab-select-pop 180ms ease;
        }

        [data-testid="stTabs"] [role="tab"][aria-selected="true"]::after {
            transform: scaleX(1);
            opacity: 1;
        }

        [data-testid="stAlert"] {
            border-radius: 20px;
            border: 1px solid var(--bbe-border);
            box-shadow: var(--bbe-shadow-soft);
        }

        [data-testid="stChatMessage"] {
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 24px;
            box-shadow: var(--bbe-shadow-soft);
            backdrop-filter: blur(14px);
        }

        [data-testid="stMarkdownContainer"] blockquote {
            border-left: 2px solid rgba(56, 189, 248, 0.55);
            background: rgba(56, 189, 248, 0.05);
            border-radius: 0 18px 18px 0;
            padding: 0.85rem 1rem;
        }

        hr {
            border-color: rgba(15, 23, 42, 0.08);
        }

        @keyframes tab-select-pop {
            0% { transform: scale(0.985); }
            60% { transform: scale(1.015); }
            100% { transform: scale(1); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_runtime_settings(defaults):
    base_url = defaults.base_url.strip()
    st.text_input(
        "Server URL",
        value=base_url,
        placeholder="https://your-ollama-server.example.com",
        disabled=True,
    )
    st.caption("Server URL is locked by the app configuration.")

    model_options = []
    for value, label in MODEL_OPTIONS:
        if value not in model_options:
            model_options.append(value)
    if defaults.model and defaults.model not in model_options:
        model_options.insert(0, defaults.model)

    selected_model = defaults.model if defaults.model in model_options else model_options[0]
    model = st.selectbox(
        "Model",
        options=model_options,
        index=model_options.index(selected_model),
        format_func=lambda value: next((label for option, label in MODEL_OPTIONS if option == value), value),
    )
    if model in MODEL_DESCRIPTIONS:
        st.caption(MODEL_DESCRIPTIONS[model])
    return {
        "base_url": base_url.strip(),
        "model": model.strip(),
        "api_key": defaults.api_key.strip(),
        "temperature": float(defaults.temperature),
        "max_tokens": int(defaults.max_tokens),
        "timeout_seconds": int(defaults.timeout_seconds),
        "critique_pass": bool(defaults.critique_pass),
    }


def get_backend_status(settings: dict):
    if not settings["base_url"]:
        return False, "Model service URL is not configured."
    if not settings["model"]:
        return False, "Model name is required."

    try:
        client = create_client(
            base_url=settings["base_url"],
            model=settings["model"],
            api_key=settings.get("api_key", ""),
            timeout_seconds=settings["timeout_seconds"],
        )
        return client.healthcheck()
    except Exception as exc:
        return False, str(exc)


def render_result(result: dict):
    claims = result.get("evidence_claims", [])
    verified_count = sum(1 for claim in claims if claim.get("verified"))
    question_support_count = sum(
        1 for claim in claims
        if claim.get("verified") and claim.get("question_relevance") == "relevant"
    )
    confidence_reason = result.get("confidence_reason", "")
    confidence_type = str(result.get("confidence_type", "")).replace("-", " ").title()
    confidence_basis = result.get("confidence_basis", "")

    st.subheader("Result")
    c1, c2, c3 = st.columns(3)
    c1.metric("Confidence", (result.get("confidence") or "low").upper())
    c1.caption("Scale: LOW, MEDIUM, HIGH")
    if confidence_type:
        c1.caption(f"Type: {confidence_type}")
    if confidence_basis:
        c1.caption(confidence_basis)
    else:
        c1.caption("Based on evidence support, coverage, and interpretation stability.")
    c2.metric("Question Support", f"{question_support_count}/{len(claims)}")
    c2.caption("Counts evidence that is both verified in the context and relevant to the question.")
    c3.write("**Confidence reason**")
    c3.write(confidence_reason)

    tab_answer, tab_blackbox, tab_evidence, tab_context, tab_risks = st.tabs(
        ["Answer", "Black Box", "Evidence", "Context", "Risks & Follow-ups"]
    )

    with tab_answer:
        st.write(result.get("answer", ""))

    with tab_blackbox:
        st.write(result.get("black_box_explanation", ""))

    with tab_evidence:
        for claim in claims:
            st.write(f"**{claim.get('claim', 'Evidence')}**")
            st.write(f"> {claim.get('quote', '')}")
            if claim.get("support_reason"):
                st.caption(claim.get("support_reason", ""))
            st.write("---")

    with tab_context:
        st.markdown(result.get("highlighted_context", ""), unsafe_allow_html=True)

    with tab_risks:
        left, right = st.columns(2)
        with left:
            st.write("**Assumptions**")
            for item in result.get("assumptions", []):
                st.markdown(f"- {item}")
            st.write("**Uncertainty / What Could Be Wrong**")
            for item in result.get("uncertainty", []):
                st.markdown(f"- {item}")
        with right:
            st.write("**Helpful What-If Questions**")
            for item in result.get("followups", []):
                st.markdown(f"- {item}")


def render_chat(settings: dict):
    st.subheader("Talk to the Model")

    if st.button("Clear chat"):
        st.session_state.followup_chat_history = []
        st.rerun()

    for msg in st.session_state.followup_chat_history:
        role = "user" if msg.get("role") == "user" else "assistant"
        with st.chat_message(role):
            st.write(msg.get("content", ""))

    chat_input = st.chat_input("Ask a follow-up...")
    if not chat_input:
        return

    user_text = chat_input.strip()
    if not user_text:
        return

    st.session_state.followup_chat_history.append({"role": "user", "content": user_text})
    client = create_client(
        base_url=settings["base_url"],
        model=settings["model"],
        api_key=settings.get("api_key", ""),
        timeout_seconds=settings["timeout_seconds"],
    )

    chat_messages = [
        {"role": "system", "content": FOLLOWUP_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Original question:\n{st.session_state.last_question}\n\n"
                f"Context:\n{st.session_state.last_context[:2000]}\n\n"
                f"Follow-up question:\n{user_text}"
            ),
        },
    ]

    try:
        with st.spinner("Getting follow-up response..."):
            reply = client.chat(
                messages=chat_messages,
                temperature=settings["temperature"],
                max_tokens=min(560, settings["max_tokens"]),
            )
    except Exception as exc:
        st.session_state.followup_chat_history.append({"role": "assistant", "content": f"The follow-up request failed: {exc}"})
    else:
        st.session_state.followup_chat_history.append({"role": "assistant", "content": reply})
    st.rerun()


init_state()
secrets = get_streamlit_secrets()
defaults = load_from_env(secrets)
inject_font_css()

with st.expander("Model Service", expanded=False):
    settings = build_runtime_settings(defaults)
    ready, status = get_backend_status(settings)
    if ready:
        st.success(status)
    else:
        st.warning(status)

st.title("Black Box Explainer")
st.caption("Explain model answers with evidence, context, and follow-up questions.")

with st.container(border=True):
    st.subheader("Input")
    question = st.text_input("Question", value=st.session_state.last_question, placeholder="Ask a specific question...")
    context = st.text_area("Context", value=st.session_state.last_context, height=260, placeholder="Paste source/context text here...")
    run = st.button("Explain", type="primary")

if run:
    if not question.strip():
        st.error("Question is required.")
    elif not context.strip():
        st.error("Context is required.")
    elif not ready:
        st.error("Model service is not ready. Check the server URL and model.")
    else:
        try:
            client = create_client(
                base_url=settings["base_url"],
                model=settings["model"],
                api_key=settings.get("api_key", ""),
                timeout_seconds=settings["timeout_seconds"],
            )
            pipeline = ExplainerPipeline(client)
            with st.spinner("Running explainer..."):
                result = pipeline.run(
                    question=question.strip(),
                    context=context,
                    temperature=settings["temperature"],
                    max_tokens=settings["max_tokens"],
                    critique_pass=settings["critique_pass"],
                )

            st.session_state.last_result = result
            st.session_state.last_question = question.strip()
            st.session_state.last_context = context
            st.session_state.followup_chat_history = []
        except Exception as exc:
            st.session_state.last_result = None
            st.error(f"Failed to run the explainer. {exc}")

if st.session_state.last_result is not None:
    st.divider()
    render_result(st.session_state.last_result)
    st.divider()
    if ready:
        render_chat(settings)
