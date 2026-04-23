# Black Box Explainer

Inspect AI answers before you trust them.

Black Box Explainer is a product-focused analysis layer for LLM outputs. It takes a user question and model response, then returns a structured breakdown of what was said, why it was likely said, where confidence is weak, and what should be checked next.

## Product Overview

Black Box Explainer helps teams move from "this sounds right" to "this is reviewable."

Core product goals:

- make AI answers inspectable and auditable
- highlight assumptions and weak reasoning quickly
- surface follow-up questions that improve decision quality
- provide a consistent review framework across prompts and models

## Core Experience

Every analysis is organized into clear product views:

- `Answer`: what the response is claiming
- `Why The Model Said It`: likely reasoning path and emphasis
- `Supporting Context`: extracted support signals and evidence hints
- `Gaps & Next Questions`: uncertainty, assumptions, and next checks

## Who This Is For

- teams using LLMs in research, operations, or decision support
- builders who want explainability baked into AI UX
- reviewers who need faster quality checks on generated content
- educators and learners comparing response quality across prompts

## Why Teams Use It

- reduces over-trust in polished but weak responses
- creates a repeatable QA process for AI-generated answers
- improves follow-up prompting and downstream decisions
- makes model behavior easier to communicate to non-technical users

## Product Status

Black Box Explainer is actively evolving as a practical explainability product with a React frontend and Python backend.

Primary app modules:

- `frontend/`
- `backend/`
- `explain/`
- `llm/`
- `utils/`

## UI Update (In Progress)

The UI is being updated to be cleaner, more intuitive, and fully responsive, with improved visual hierarchy and a clearer step-by-step analysis flow across desktop and mobile.

## Repository Layout

```text
blackbox_explainer/
  frontend/     Product UI
  backend/      API and orchestration
  explain/      Core explanation pipeline
  llm/          Model client interfaces
  utils/        Shared utilities
  config.py     Shared configuration
```

