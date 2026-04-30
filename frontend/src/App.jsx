import React, { useEffect, useMemo, useState } from "react";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8001").replace(/\/$/, "");
const TABS = [
  { key: "answer", label: "Answer" },
  { key: "black_box_explanation", label: "Why The Model Said It" },
  { key: "evidence", label: "Supporting Context" },
  { key: "risks", label: "Gaps & Risks" },
];


function ResultTabs({ result, onReset }) {
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
          <h4>Why the app scored it this way</h4>
          <p className="metric-body">{result.confidence_reason || "No confidence explanation returned."}</p>
        </div>
      </div>
    );
  }, [activeTab, result]);

  return (
    <div className="tabs-shell">
      <div className="tabs-row">
        <div className="tabs-row-pills">
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
        <button className="ghost-button new-chat-button" type="button" onClick={onReset}>
          New chat
        </button>
      </div>
      <div className="tab-panel">{content}</div>
    </div>
  );
}
function ResultPanel({ result, onReset }) {
  return (
    <section className="panel-shell audit-form-panel result-combined-panel">
      <div className="audit-chat-thread result-thread">
        {result.fallback_mode ? (
          <section className="fallback-notice in-panel">
            <div className="fallback-icon" aria-hidden="true">!</div>
            <div>
              <strong>Fallback review is showing</strong>
              <p>{result.fallback_error || "The model did not return a full generation, so BLOEX used a lightweight local review."}</p>
            </div>
          </section>
        ) : null}

        <ResultTabs result={result} onReset={onReset} />
      </div>
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

function TitlePage({ onStart }) {
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
  const [context, setContext] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeModel, setActiveModel] = useState("phi3:mini");
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
          status: health.ok ? "Black box linked. Ready to explain." : (health.status || "Backend unavailable."),
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

    if (!modelAnswer.trim()) {
      setError("The model answer is required.");
      return;
    }
    if (!question.trim()) {
      setError("Fix the missing field and run the audit again.");
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
          context,
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
        status: "Black box linked. Ready to explain.",
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

  function resetAudit() {
    setResult(null);
    setQuestion("");
    setModelAnswer("");
    setContext("");
    setError("");
  }

  if (view === "home") {
    return <TitlePage onStart={openEvaluateView} />;
  }

  return (
    <div className="audit-page">
      <div className="audit-homepage-bg" aria-hidden="true">
        <div className="ml-vignette" />
        <div className="ml-light-ray ray-a" />
        <div className="ml-light-ray ray-b" />
      </div>
      <header className="ml-topbar">
        <button className="ml-logo ml-logo-button" type="button" onClick={openHomeView} aria-label="Back to home">
          <span className="ml-logo-mark" aria-hidden="true" />
          <span>BLOEX</span>
        </button>
      </header>
      <div className="audit-workspace">
        {!result ? (
        <section className="panel-shell audit-form-panel">
          <form onSubmit={handleExplain} className="input-form">
            <div className="audit-chat-header">
              <div>
                <h2>Start an audit</h2>
                <p>Share the prompt and the answer exactly as it was written.</p>
              </div>
            </div>

            <div className="audit-chat-thread">
              <div className="audit-input-grid">
                <label className="audit-field-card">
                  <span>Question</span>
                  <input
                    value={question}
                    onChange={(event) => setQuestion(event.target.value)}
                    placeholder="Enter the user’s original question."
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
                  <span>Model answer</span>
                  <textarea
                    value={modelAnswer}
                    onChange={(event) => setModelAnswer(event.target.value)}
                    placeholder="Paste the model’s answer."
                    rows={7}
                    style={{ overflow: "hidden", resize: "none" }}
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

            <div className="audit-composer-bar">
              <div className="audit-composer-status">
                <span>{error || status.status}</span>
              </div>
              <button className="primary-button" type="submit" disabled={loading}>
                {loading ? "Explaining..." : "Explain"}
              </button>
            </div>
          </form>
        </section>
        ) : (
          <ResultPanel
            result={result}
            onReset={resetAudit}
          />
        )}
      </div>
    </div>
  );
}

