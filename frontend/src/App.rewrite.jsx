yimport { useEffect, useMemo, useState } from "react";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
const TABS = [
  { key: "answer", label: "Audit" },
  { key: "black_box_explanation", label: "Why The Model Said It" },
  { key: "evidence", label: "Supporting Context" },
  { key: "context", label: "Source View" },
  { key: "risks", label: "Gaps & Next Questions" },
];

function getGroundedClaims(result) {
  return (result.evidence_claims || []).filter(
    (claim) =>
      claim.question_relevance === "relevant" &&
      (claim.verified || ["direct", "partial"].includes(claim.context_support_label)),
  );
}

function getStrongestClaim(result) {
  return getGroundedClaims(result)[0] || (result.evidence_claims || [])[0] || null;
}

function getWeakestSentence(result) {
  return (result.answer_support || []).find((item) => ["overreach", "unsupported"].includes(item.label))
    || (result.answer_support || []).find((item) => item.label === "inferred")
    || null;
}

function getReuseVerdict(result) {
  const groundedClaims = getGroundedClaims(result).length;
  const claimCount = (result.evidence_claims || []).length;
  const flaggedSentences = (result.answer_support || []).filter((item) => ["overreach", "unsupported"].includes(item.label)).length;
  const confidence = result.confidence || "low";

  if (!groundedClaims || confidence === "low") {
    return {
      status: "High risk",
      headline: "Do not reuse this answer without checking the source.",
      detail: `Only ${groundedClaims}/${claimCount} extracted support points are grounded in the source, so the answer still needs human review.`,
    };
  }

  if (flaggedSentences || confidence === "medium") {
    return {
      status: "Needs review",
      headline: "Useful draft, but check the weak parts before reusing it.",
      detail: `The answer has some real support, but ${flaggedSentences} sentence${flaggedSentences === 1 ? "" : "s"} still look too strong or weakly supported.`,
    };
  }

  return {
    status: "Grounded",
    headline: "This answer is mostly usable, but still worth a quick skim.",
    detail: "The strongest claims are grounded in the source and no major overreach was flagged.",
  };
}

function formatSentenceLabel(label) {
  if (!label) return "inferred";
  return label.replaceAll("_", " ");
}

function AuditSummary({ result }) {
  const groundedClaims = getGroundedClaims(result);
  const strongestClaim = getStrongestClaim(result);
  const weakestSentence = getWeakestSentence(result);
  const verdict = getReuseVerdict(result);

  return (
    <section className="panel-shell audit-summary-shell">
      <div className="audit-summary-main">
        <div className="audit-status-row">
          <span className="audit-status-pill">{verdict.status}</span>
          <span className="metric-caption">Confidence: {(result.confidence || "low").toUpperCase()}</span>
        </div>
        <h2>{verdict.headline}</h2>
        <p className="audit-summary-copy">{result.audit_verdict || result.answer || "No audit verdict was returned."}</p>
        <p className="metric-caption">{verdict.detail}</p>
      </div>

      <div className="audit-summary-grid">
        <div className="summary-block">
          <div className="metric-label">What the model seems to be doing</div>
          <div className="metric-body">{result.black_box_explanation || "No model-behavior readout was returned."}</div>
        </div>
        <div className="summary-block">
          <div className="metric-label">Best source support</div>
          <div className="metric-body">
            {strongestClaim?.matched_context || strongestClaim?.quote || "No clearly grounded supporting snippet was found."}
          </div>
        </div>
        <div className="summary-block">
          <div className="metric-label">What to check first</div>
          <div className="metric-body">
            {weakestSentence?.sentence || (result.uncertainty || [])[0] || "No single weak sentence was isolated."}
          </div>
          {weakestSentence?.reason ? <div className="metric-caption">{weakestSentence.reason}</div> : null}
        </div>
        <div className="summary-block">
          <div className="metric-label">Grounding snapshot</div>
          <div className="summary-stat-row">
            <span>Grounded support</span>
            <strong>{groundedClaims.length}/{(result.evidence_claims || []).length}</strong>
          </div>
          <div className="summary-stat-row">
            <span>Flagged answer sentences</span>
            <strong>{(result.answer_support || []).filter((item) => ["overreach", "unsupported"].includes(item.label)).length}</strong>
          </div>
          <div className="summary-stat-row">
            <span>Main model confidence</span>
            <strong>{(result.confidence || "low").toUpperCase()}</strong>
          </div>
        </div>
      </div>
    </section>
  );
}

function ResultTabs({ result }) {
  const [activeTab, setActiveTab] = useState("answer");

  const content = useMemo(() => {
    if (activeTab === "answer") {
      return (
        <div className="answer-tab-shell">
          <div className="section-block">
            <div className="metric-label">Model answer being audited</div>
            <div className="metric-body">{result.audited_answer || "No model answer was captured for this run."}</div>
          </div>

          <div className="section-block">
            <div className="metric-label">Audit verdict</div>
            <div className="metric-body">{result.audit_verdict || result.answer || "No audit verdict was returned."}</div>
          </div>

          <div className="detector-shell">
            <div className="detector-header">
              <h4>Sentence-by-sentence check</h4>
              <span className={`detector-badge ${(result.overreach_flags || []).length ? "flagged" : "clean"}`}>
                {(result.overreach_flags || []).length ? "Needs review" : "No major overreach"}
              </span>
            </div>
            <p className="detector-summary">
              This checks each sentence in the model answer to see whether it stays grounded in the source or stretches beyond it.
            </p>
            <div className="detector-list">
              {(result.answer_support || []).length ? (
                (result.answer_support || []).map((item, index) => (
                  <div key={`${item.sentence}-${index}`} className={`detector-item ${item.label || "inferred"}`}>
                    <div className="detector-item-top">
                      <span className="detector-item-label">{formatSentenceLabel(item.label)}</span>
                      <span className="detector-item-score">Support: {item.support_ratio ?? 0}</span>
                    </div>
                    <div className="detector-item-sentence">{item.sentence}</div>
                    {item.matched_context ? (
                      <div className="metric-caption">Closest source support: "{item.matched_context}"</div>
                    ) : null}
                    <div className="metric-caption">{item.reason}</div>
                  </div>
                ))
              ) : (
                <div className="metric-caption">No sentence-level audit was returned.</div>
              )}
            </div>
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

    if (activeTab === "context") {
      return (
        <div
          className="context-panel"
          dangerouslySetInnerHTML={{ __html: result.highlighted_context || "" }}
        />
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
        Ask the auditor to challenge one claim, rewrite a weak sentence, or explain why a source snippet does or does not support the model answer.
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
          placeholder="Ask the auditor to check a claim, rewrite a sentence, or suggest a stronger follow-up..."
          rows={3}
        />
        <button className="primary-button" type="submit" disabled={!ready || loading}>
          {loading ? "Thinking..." : "Send"}
        </button>
      </form>
    </section>
  );
}

export default function App() {
  const [question, setQuestion] = useState("");
  const [modelAnswer, setModelAnswer] = useState("");
  const [context, setContext] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeModel, setActiveModel] = useState("phi3:mini");
  const [status, setStatus] = useState({ ok: false, status: "Checking backend..." });
  const [error, setError] = useState("");

  useEffect(() => {
    async function bootstrap() {
      try {
        const healthResponse = await fetch(`${API_BASE_URL}/api/health`);
        const health = await healthResponse.json();

        if (health.selectedModel) {
          setActiveModel(health.selectedModel);
        }
        setStatus({
          ok: Boolean(health.ok),
          status: health.status || "Backend unavailable.",
        });
      } catch (fetchError) {
        setStatus({
          ok: false,
          status: fetchError instanceof Error ? fetchError.message : "Failed to reach backend.",
        });
      }
    }

    bootstrap();
  }, []);

  async function handleExplain(event) {
    event.preventDefault();
    setError("");

    if (!question.trim()) {
      setError("The original question is required.");
      return;
    }
    if (!modelAnswer.trim()) {
      setError("The model answer to audit is required.");
      return;
    }
    if (!context.trim()) {
      setError("Source context is required.");
      return;
    }
    if (!status.ok) {
      setError("Model service is not ready. Check the backend deployment.");
      return;
    }

    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/explain`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: question.trim(),
          model_answer: modelAnswer.trim(),
          context,
          model: activeModel,
        }),
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "Failed to run the auditor.");
      }
      setResult(payload);
    } catch (runError) {
      setError(runError instanceof Error ? runError.message : "Failed to run the auditor.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page-shell">
      <header className="hero-shell">
        <h1>AI Answer Auditor</h1>
        <p>
          Paste the original question, the model's answer, and the source text. The app shows whether the answer is grounded, where it stretches, and what a human should verify before trusting it.
        </p>
        <div className="hero-meta-row">
          <div className="hero-chip">Catch unsupported claims</div>
          <div className="hero-chip">See what the model leaned on</div>
          <div className="hero-chip">Find the first thing to verify</div>
        </div>
      </header>

      <section className="panel-shell">
        <div className={`status-banner ${status.ok ? "ok" : "warn"}`}>{status.status}</div>
      </section>

      <section className="panel-shell">
        <form onSubmit={handleExplain} className="input-form">
          <h2>Run an audit</h2>

          <label>
            <span>Original question</span>
            <input
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="What was the user asking the model to answer?"
            />
          </label>

          <label>
            <span>Model answer to audit</span>
            <textarea
              value={modelAnswer}
              onChange={(event) => setModelAnswer(event.target.value)}
              placeholder="Paste the exact answer another LLM gave you..."
              rows={7}
            />
          </label>

          <label>
            <span>Source / context</span>
            <textarea
              value={context}
              onChange={(event) => setContext(event.target.value)}
              placeholder="Paste the source text, retrieved chunks, resume content, policy text, notes, or anything the model should have relied on..."
              rows={10}
            />
          </label>

          {error ? <div className="status-banner warn">{error}</div> : null}

          <button className="primary-button" type="submit" disabled={loading}>
            {loading ? "Running audit..." : "Run audit"}
          </button>
        </form>
      </section>

      {result ? (
        <>
          <section className="results-header">
            <h2>Audit result</h2>
            <p className="results-subtitle">
              This workspace is for deciding whether an AI answer is safe to trust, edit, or reuse, and for showing what the model likely relied on when it produced that answer.
            </p>
          </section>

          <AuditSummary result={result} />

          <section className="panel-shell">
            <ResultTabs result={result} />
          </section>

          <ChatPanel
            ready={status.ok}
            selectedModel={activeModel}
            question={question}
            modelAnswer={modelAnswer}
            context={context}
          />
        </>
      ) : null}
    </div>
  );
}
