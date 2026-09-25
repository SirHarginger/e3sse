# E3-SSE Agent Toolkit

## Shared skills

- acquire-codebase-knowledge
- poka-yoke
- dataset-audit
- data-loader
- scientific-python
- mlip-inference
- estimate-softening
- barrier-analysis
- transport-analysis
- conformal-selection
- experiment-run
- results-validation
- reproducibility-audit
- code-review
- prepare-pr

## Specialist agents

- research-architect
- data-engineer
- mlip-specialist
- statistical-validator
- code-reviewer

Claude Code:
`.claude/agents/`

GitHub Copilot:
`.github/agents/`

Codex uses the root `AGENTS.md`, shared `.agents/skills/`, and may delegate
focused work using its own subagent runtime.

## Typical workflow

Repository onboarding:
    acquire-codebase-knowledge

Dataset work:
    dataset-audit -> data-loader

Softening work:
    mlip-inference -> estimate-softening -> results-validation

Barrier work:
    barrier-analysis -> results-validation

Transport work:
    transport-analysis -> results-validation

Screening:
    conformal-selection -> results-validation

Production execution:
    experiment-run -> reproducibility-audit

Before merge:
    code-review -> prepare-pr
