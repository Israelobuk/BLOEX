import React, { useEffect, useMemo, useState } from "react";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");
const TABS = [
  { key: "answer", label: "Answer" },
  { key: "black_box_explanation", label: "Why The Model Said It" },
  { key: "evidence", label: "Supporting Context" },
  { key: "risks", label: "Gaps & Next Questions" },
];

function ResultTabs({ result }) {
  const [activeTab, setActiveTab] = useState("answer");

  const content = useMemo(() => {
    if (activeTab === "answer") {
      return (
        <div className="answer-tab-shell">
          <div className="section-block">
            <div className="metric-label">Answer</div>
            <div className="metric-body">{result.audit_verdict || result.answer || "No answer was returned."}</div>
          </div>
        </div>
      );
    }

    if (activeTab === "black_box_explanation") {
      return (
        <div className="section-block">
          <div className="metric-label">Why the model likely landed on this answer</div>
          <div className="metric-body">{result.black_box_explanation || "No model-behavior explanation was returned."}</div>
        </div>
      );
    }

    if (activeTab === "evidence") {
      return (
        <div className="evidence-list">
          {(result.evidence_claims || []).length ? (
            (result.evidence_claims || []).map((claim, index) => (
              <div key={`${claim.quote}-${index}`} className="evidence-card">
                <div className="evidence-title">{claim.claim || "Evidence"}</div>
                <blockquote>{claim.quote || "No quote extracted."}</blockquote>
                {claim.matched_context && claim.matched_context !== claim.quote ? (
                  <div className="metric-caption">Closest source support: "{claim.matched_context}"</div>
                ) : null}
                {claim.support_reason ? <div className="metric-caption">{claim.support_reason}</div> : null}
              </div>
            ))
          ) : (
            <div className="metric-caption">No evidence claims were returned.</div>
          )}
        </div>
      );
    }

    return (
      <div className="risk-grid">
        <div className="risk-column">
          <h4>Assumptions the answer depends on</h4>
          <ul>{(result.assumptions || []).map((item, index) => <li key={`a-${index}`}>{item}</li>)}</ul>

          <h4>Where the answer may still be weak</h4>
          <ul>{(result.uncertainty || []).map((item, index) => <li key={`u-${index}`}>{item}</li>)}</ul>
        </div>
        <div className="risk-column">
          <h4>Questions to ask next</h4>
          <ul>{(result.followups || []).map((item, index) => <li key={`f-${index}`}>{item}</li>)}</ul>

          <h4>Why the app scored it this way</h4>
          <p className="metric-body">{result.confidence_reason || "No confidence explanation returned."}</p>
        </div>
      </div>
    );
  }, [activeTab, result]);

  return (
    <div className="tabs-shell">
      <div className="tabs-row">
        {TABS.map((tab) => (
          <button
            key={tab.key}
            className={`tab-pill ${activeTab === tab.key ? "active" : ""}`}
            onClick={() => setActiveTab(tab.key)}
            type="button"
          >
            {tab.label}
          </button>
        ))}
      </div>
      <div className="tab-panel">{content}</div>
    </div>
  );
}

function ChatPanel({ ready, selectedModel, question, modelAnswer, context }) {
  const [messages, setMessages] = useState([]);
  const [draft, setDraft] = useState("");
  const [loading, setLoading] = useState(false);

  async function sendFollowup(event) {
    event.preventDefault();
    const cleaned = draft.trim();
    if (!cleaned || !ready) return;

    setLoading(true);
    setMessages((current) => [...current, { role: "user", content: cleaned }]);
    setDraft("");

    try {
      const response = await fetch(`${API_BASE_URL}/api/followup`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          model_answer: modelAnswer,
          context,
          followup: cleaned,
          model: selectedModel,
        }),
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "Follow-up request failed.");
      }
      setMessages((current) => [...current, { role: "assistant", content: payload.reply }]);
    } catch (error) {
      setMessages((current) => [
        ...current,
        { role: "assistant", content: error instanceof Error ? error.message : "Follow-up request failed." },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="panel-shell">
      <div className="section-title-row">
        <h2>Pressure-test this answer</h2>
        {messages.length ? (
          <button className="ghost-button" type="button" onClick={() => setMessages([])}>
            Clear chat
          </button>
        ) : null}
      </div>
      <p className="results-subtitle compact">
        Ask the explainer to challenge one claim, rewrite a weak sentence, or explain why a source snippet does or does not support the answer.
      </p>
      <div className="chat-stack">
        {messages.map((message, index) => (
          <div key={`${message.role}-${index}`} className={`chat-bubble ${message.role}`}>
            {message.content}
          </div>
        ))}
      </div>
      <form className="chat-form" onSubmit={sendFollowup}>
        <textarea
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          placeholder="Ask the explainer to check a claim, rewrite a sentence, or suggest a stronger follow-up..."
          rows={3}
          autoComplete="off"
          autoCorrect="off"
          autoCapitalize="off"
          spellCheck={false}
          data-gramm="false"
          data-gramm_editor="false"
          data-enable-grammarly="false"
          data-lt-active="false"
        />
        <button className="primary-button" type="submit" disabled={!ready || loading}>
          {loading ? "Thinking..." : "Send"}
        </button>
      </form>
    </section>
  );
}

async function readJsonResponse(response) {
  const rawText = await response.text();
  let payload = {};

  if (rawText.trim()) {
    try {
      payload = JSON.parse(rawText);
    } catch {
      payload = { detail: rawText.trim() };
    }
  }

  return payload;
}

function TitlePage({ status, providerTone, providerLabel, onStart }) {
  return (
    <section className="title-fullscreen">
      <div className="ml-atmosphere" aria-hidden="true">
        <div className="ml-gradient-flow" />
        <div className="ml-nebula n1" />
        <div className="ml-nebula n2" />
        <div className="ml-corner-slash beam-a" />
        <div className="ml-vignette" />
        <div className="ml-light-ray ray-a" />
        <div className="ml-light-ray ray-b" />
      </div>

      <header className="ml-topbar">
        <div className="ml-logo">
          <span className="ml-logo-mark" aria-hidden="true" />
          <span>BLOEX</span>
        </div>
        <button className="ml-enter" type="button" onClick={onStart}>
          Launch
        </button>
      </header>

      <div className="ml-hero-shell">
        <div className="ml-copy-column">
          <p className="ml-kicker ml-kicker-slash">BLACK BOX EXPLAINER</p>
          <p className="ml-subcopy ml-subcopy-slash">
            Audit generated answers against evidence, surface weak claims, and ship responses with a clear reliability score.
          </p>
        </div>
      </div>

    </section>
  );
}
export default function App() {
  const [view, setView] = useState("home");
  const [question, setQuestion] = useState("");
  const [modelAnswer, setModelAnswer] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeModel, setActiveModel] = useState("tinyllama:latest");
  const [status, setStatus] = useState({ ok: false, status: "Checking backend..." });
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;

    async function bootstrap() {
      try {
        const healthResponse = await fetch(`${API_BASE_URL}/api/health`);
        const health = await readJsonResponse(healthResponse);

        if (cancelled) {
          return;
        }

        if (health.selectedModel) {
          setActiveModel(health.selectedModel);
        }
        setStatus({
          ok: Boolean(health.ok),
          status: health.status || "Backend unavailable.",
        });
      } catch (fetchError) {
        if (cancelled) {
          return;
        }
        setStatus({
          ok: false,
          status: "Backend is waking up or unreachable right now.",
        });
      }
    }

    bootstrap();

    const timer = window.setInterval(bootstrap, 15000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  async function handleExplain(event) {
    event.preventDefault();
    setError("");

    if (!question.trim()) {
      setError("The original question is required.");
      return;
    }
    if (!modelAnswer.trim()) {
      setError("The model answer is required.");
      return;
    }
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/explain`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: question.trim(),
          model_answer: modelAnswer,
          model: activeModel,
        }),
      });
      const payload = await readJsonResponse(response);
      if (!response.ok) {
        throw new Error(payload.detail || "Failed to run the auditor.");
      }
      setResult(payload);
      if (payload.selected_model) {
        setActiveModel(payload.selected_model);
      }
      setStatus({
        ok: true,
        status: "Backend responded successfully.",
      });
    } catch (runError) {
      const message = runError instanceof Error ? runError.message : "Failed to run the auditor.";
      setError(message);
      setStatus({
        ok: false,
        status: message,
      });
    } finally {
      setLoading(false);
    }
  }

  function openEvaluateView() {
    setView("evaluate");
  }

  function openHomeView() {
    setView("home");
  }

  const providerTone = status.ok ? "ok" : "warn";
  const providerLabel = status.ok ? "AI Provider Active" : "AI Provider Unavailable";

  if (view === "home") {
    return <TitlePage status={status} providerTone={providerTone} providerLabel={providerLabel} onStart={openEvaluateView} />;
  }

  return (
    <div className="page-shell audit-page">
      <div className="audit-homepage-bg" aria-hidden="true">
        <div className="ml-vignette" />
        <div className="ml-light-ray ray-a" />
        <div className="ml-light-ray ray-b" />
      </div>
      <div className="audit-workspace">
        <div className="workspace-nav">
          <button className="audit-home-logo" type="button" onClick={openHomeView} aria-label="Back to home">
            <span className="audit-home-mark" aria-hidden="true" />
            <span>BLOEX</span>
          </button>
          <div className={`status-pill ${providerTone}`}>{providerLabel}</div>
        </div>

        <section className="panel-shell audit-form-panel">
          <form onSubmit={handleExplain} className="input-form">
            <div className="audit-chat-header">
              <div>
                <h2>Start an audit</h2>
                <p>Share the prompt and the answer exactly as it was written.</p>
              </div>
              <div className={`backend-chip ${status.ok ? "ok" : "warn"}`}>
                <span aria-hidden="true" />
                {status.ok ? "Backend online" : "Backend unavailable"}
              </div>
            </div>

            <div className="audit-chat-thread">
              <div className="audit-input-grid">
                <label className="audit-field-card">
                  <span>Original question</span>
                  <input
                    value={question}
                    onChange={(event) => setQuestion(event.target.value)}
                    placeholder="What was the user asking the model to answer?"
                    autoComplete="off"
                    autoCorrect="off"
                    autoCapitalize="off"
                    spellCheck={false}
                    data-gramm="false"
                    data-gramm_editor="false"
                    data-enable-grammarly="false"
                    data-lt-active="false"
                  />
                </label>

                <label className="audit-field-card large">
                  <span>LLM answer to analyze</span>
                  <textarea
                    value={modelAnswer}
                    onChange={(event) => setModelAnswer(event.target.value)}
                    placeholder="Paste the exact answer the LLM gave you..."
                    rows={7}
                    autoComplete="off"
                    autoCorrect="off"
                    autoCapitalize="off"
                    spellCheck={false}
                    data-gramm="false"
                    data-gramm_editor="false"
                    data-enable-grammarly="false"
                    data-lt-active="false"
                  />
                </label>
              </div>
            </div>

            {error ? <div className="status-banner warn">{error}</div> : null}

            <div className="audit-composer-bar">
              <span>{status.status}</span>
              <button className="primary-button" type="submit" disabled={loading}>
                {loading ? "Explaining..." : "Explain"}
              </button>
            </div>
          </form>
        </section>

        {result ? (
          <>
            <section className="results-header">
              <h2>Result</h2>
              <p className="results-subtitle">
                This workspace is for understanding the answer, what the model likely relied on, and where its reasoning may still be shaky.
              </p>
            </section>

          {result.fallback_mode ? (
              <section className="fallback-notice">
                <div className="fallback-icon" aria-hidden="true">!</div>
                <div>
                  <strong>Fallback review is showing</strong>
                  <p>{result.fallback_error || "The model did not return a full generation, so BLOEX used a lightweight local review."}</p>
                </div>
              </section>
            ) : null}

            <section className="panel-shell">
              <ResultTabs result={result} />
            </section>

            <ChatPanel
              ready={status.ok}
              selectedModel={activeModel}
              question={question}
              modelAnswer={modelAnswer}
              context=""
            />
          </>
        ) : null}
      </div>
    </div>
  );
}

