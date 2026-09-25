# Claude Code Instructions for E3-SSE

Read `AGENTS.md` before substantial work. It is the canonical project policy.

Use the project skills under `.claude/skills/` for repeatable scientific
workflows.

Use project subagents under `.claude/agents/` when specialist isolation is
useful.

Critical rules:

- Never destructively modify `data/` or `/srv/ben/e3sse/data`.
- Treat `data/raw/` as immutable.
- Never fabricate scientific values.
- Inspect dataset schemas before writing loaders.
- Preserve split and leakage rules.
- Keep DFT, raw MLIP, corrected, and fitted quantities distinct.
- Put reusable code under `src/e3sse/`.
- Run `bash .agents/scripts/validate_repo.sh` before declaring code complete.
- Record provenance for production scientific runs.
- Prefer minimal, reviewable changes over broad unsolicited rewrites.

For complex work, first identify the relevant skill and specialist agent.
