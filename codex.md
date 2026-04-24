# Codex UI and Debug Policy

## Global UI Workflow Policy

Use these defaults for UI tasks across projects in this workspace:

1. `@frontend-skill` (primary art direction and interaction quality)
2. `@build-web-apps:frontend-skill` (web composition and layout polish)
3. `@build-web-apps:frontend-app-builder` (UI build execution and implementation fidelity)
4. `@build-web-apps` (implementation workflow support)
5. `@remotion` (motion language and animation structure when needed)
## Interaction and Motion Requirements

- Keep first-screen hierarchy clear and easy to scan.
- Favor fast, smooth, restrained animations over heavy effects.
- Use motion to improve orientation and affordance, not decoration.
- Preserve performance on desktop and mobile.
- Respect `prefers-reduced-motion`.

## Design Guardrails

- Avoid generic layouts and unnecessary UI clutter.
- Prefer one dominant visual idea per section.
- Keep copy concise and utility-focused for product surfaces.
- Keep visual systems consistent (spacing, type scale, accent usage).

## Execution Notes

- Apply these defaults unless a project-level `CODEX.md` overrides them.
- If a named skill is unavailable, use the closest available workflow and keep the same design intent.

## Superpowers (Debugging and Fixing Code)

1. Reproduce first, then fix: identify failure paths before changing behavior.
2. Keep fixes modular and local; avoid broad refactors unless required.
3. Protect existing working paths with backward-compatible interfaces.
4. Add clear error handling for provider/network/storage failures.
5. Prefer deterministic logic for scoring and verification; AI should assist explanation, not final judgment.
6. Log actionable diagnostics for backend operations while keeping user-facing errors clean.
7. Validate structured outputs strictly and fail gracefully on schema mismatches.
8. Confirm changes with targeted tests or smoke checks for affected flows.

## Enforcement

- This file is the canonical project instruction source.
- Keep a single `codex.md` per project root.
- Remove nested duplicates and point back to this file when needed.

